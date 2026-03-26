# This file contains the Streamlit frontend for the AutoQuizzer application.
# It allows users to input a topic and receive a quiz on it.
# Users can submit their answers to receive a score and per-question feedback.
#
# Created: 2026-03-25
# Author: Devon Vanaenrode
# Updated: 2026-03-26

# --- Imports ---
import streamlit as st
import sys
import os

# Adjust path to import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RAG_Pipeline.rag_chain import get_quiz_generation_chain

# --- Globals ---
CORRECT_ICON = "✅"
INCORRECT_ICON = "❌"

# --- Helpers ---
def resolve_correct_answer_text(question):
    """
    Returns the correct answer as display text.
    Handles both key-based answers (e.g. 'A') and full-text answers.
    """
    options_dict = question.get('options', {})
    answer = question.get('answer', '')

    if isinstance(options_dict, dict) and answer in options_dict:
        return options_dict[answer]
    return answer  # Fallback: answer is already the full text

def get_options_list(options_dict):
    """Extracts a flat list of option texts from the options field."""
    if isinstance(options_dict, dict):
        return list(options_dict.values())
    if isinstance(options_dict, list):
        return [v for opt in options_dict if isinstance(opt, dict) for v in opt.values()]
    return []

def score_quiz(questions):
    """
    Computes the number of correct answers given session state radio selections.
    Returns (score, total).
    """
    correct = sum(
        1 for q in questions
        if isinstance(q, dict) and
        st.session_state.get(f"q_{q.get('question_number') or id(q)}") == resolve_correct_answer_text(q)
    )
    return correct, len(questions)

# --- Main loop ---
st.markdown("""
<style>
    div[role="radiogroup"] label p {
        font-size: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("AutoQuizzer")

topic = st.text_input("Enter the topic you want to be quizzed on:")

if 'quiz' not in st.session_state:
    st.session_state.quiz = None

if 'submitted' not in st.session_state:
    st.session_state.submitted = False

if st.button("Generate Quiz"):
    if topic:
        with st.spinner("Generating your quiz..."):
            try:
                quiz_chain = get_quiz_generation_chain()
                if quiz_chain:
                    st.session_state.quiz = quiz_chain.invoke(topic)
                    st.session_state.submitted = False
                else:
                    st.error("Failed to initialize the quiz generation chain. Please check the logs.")
            except Exception as e:
                st.error(f"An error occurred while generating the quiz: {e}")
    else:
        st.warning("Please enter a topic first.")

# Display the quiz if it exists in the session state
if st.session_state.quiz:
    st.subheader("Here is your quiz!")
    quiz_data = st.session_state.quiz

    if isinstance(quiz_data, list):
        quiz_data = quiz_data[0] if quiz_data else {}

    if isinstance(quiz_data, dict):
        if 'quiz_title' in quiz_data:
            st.title(quiz_data['quiz_title'])

        questions = quiz_data.get('questions', [])
        if isinstance(questions, list) and questions:
            for question in questions:
                if not isinstance(question, dict):
                    st.warning(f"Found a question item that is not a dictionary: {question}")
                    continue

                question_text = question.get('question', 'No question text provided.')
                question_number = question.get('question_number')
                options_dict = question.get('options')
                options_list = get_options_list(options_dict)
                q_key = f"q_{question_number or id(question)}"

                if not (question_text and options_list):
                    continue

                question_header = f"### {question_number}. {question_text}" if question_number else f"### {question_text}"
                st.markdown(question_header)

                st.radio(
                    "Options",
                    options=options_list,
                    key=q_key,
                    label_visibility="hidden",
                    disabled=st.session_state.submitted
                )

                # Show per-question feedback after submission
                if st.session_state.submitted:
                    selected = st.session_state.get(q_key)
                    correct_text = resolve_correct_answer_text(question)
                    explanation = question.get('explanation', '')

                    if selected == correct_text:
                        st.success(f"{CORRECT_ICON} Correct!")
                    else:
                        st.error(f"{INCORRECT_ICON} Incorrect. The correct answer is: **{correct_text}**")

                    if explanation:
                        st.info(f"**Explanation:** {explanation}")

            # Submit button and score display
            if not st.session_state.submitted:
                if st.button("Submit Answers"):
                    st.session_state.submitted = True
                    st.rerun()
            else:
                score, total = score_quiz(questions)
                st.divider()
                st.subheader(f"Your Score: {score} / {total}")
                if score == total:
                    st.balloons()
                if st.button("Retake Quiz"):
                    st.session_state.submitted = False
                    st.rerun()
        else:
            st.error("Could not find 'questions' in the quiz, or it's not a list.")
            st.json(quiz_data)
    else:
        st.error("Generated quiz is not in the expected dictionary format.")
        st.json(quiz_data)
