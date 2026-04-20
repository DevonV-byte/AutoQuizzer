// Quiz screen: drives the study loop — topic/difficulty setup, one question at a time
// via /encounter/batch (pre-generated pool of 25), immediate feedback via /answer,
// per-quiz score (5 questions each) and level-complete detection at 80% pass rate.
// State machine: setup → loading → question → answered → [quiz_complete →] loading (next batch) → …
// Player ID is persisted in localStorage so progress survives page refreshes.
//
// Created: 2026-04-10
// Updated: 2026-04-20 — switched to /encounter/batch; player ID; question queue;
//                        quiz_complete / level_complete phases; quiz/question progress in scorebar
// Author: Devon Vanaenrode

import { useState } from 'react'
import { encounterBatch, submitAnswer, ApiError } from '@/api/grimoire'
import type { Difficulty, EncounterResponse, AnswerResponse } from '@/api/grimoire'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Phase = 'setup' | 'loading' | 'question' | 'answered' | 'quiz_complete'

export type TopicScores = Record<string, { correct: number; total: number }>

interface QuizState {
  question: EncounterResponse
  result: AnswerResponse | null
  selected: string | null
}

interface Props {
  zones: string[]
  onResults: (breakdown: TopicScores) => void
  onRestart: () => void
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DIFFICULTIES: { key: Difficulty; label: string }[] = [
  { key: 'easy',   label: 'Easy'   },
  { key: 'medium', label: 'Medium' },
  { key: 'hard',   label: 'Hard'   },
]

const QUIZ_SIZE  = 5  // must match backend QUIZ_SIZE
const N_QUIZZES  = 5  // must match backend POOL_SIZE / QUIZ_SIZE

// ---------------------------------------------------------------------------
// Player ID — persisted in localStorage so progress survives refreshes
// ---------------------------------------------------------------------------

function getOrCreatePlayerId(): string {
  const key = 'grimoire_player_id'
  let id = localStorage.getItem(key)
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem(key, id)
  }
  return id
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function QuizScreen({ zones, onResults, onRestart }: Props) {
  const [phase, setPhase]                     = useState<Phase>('setup')
  const [selectedZone, setSelectedZone]       = useState<string>(zones[0] ?? '')
  const [difficulty, setDifficulty]           = useState<Difficulty>('medium')
  const [quiz, setQuiz]                       = useState<QuizState | null>(null)
  const [questionQueue, setQuestionQueue]     = useState<EncounterResponse[]>([])
  const [quizNumber, setQuizNumber]           = useState(1)
  const [lastQuizScore, setLastQuizScore]     = useState<number | null>(null)
  const [levelComplete, setLevelComplete]     = useState(false)
  const [topicScores, setTopicScores]         = useState<TopicScores>({})
  const [error, setError]                     = useState<string | null>(null)

  // Stable player ID for this browser
  const [playerId] = useState(getOrCreatePlayerId)

  // Derived overall score across all topics
  const totalCorrect  = Object.values(topicScores).reduce((s, t) => s + t.correct, 0)
  const totalAnswered = Object.values(topicScores).reduce((s, t) => s + t.total,   0)

  // Pop from queue if available; otherwise fetch a new batch from the backend
  async function fetchNextQuestion() {
    if (questionQueue.length > 0) {
      const [next, ...rest] = questionQueue
      setQuestionQueue(rest)
      setQuiz({ question: next, result: null, selected: null })
      setPhase('question')
      return
    }

    setPhase('loading')
    setError(null)
    try {
      const batch = await encounterBatch({ player_id: playerId, zone: selectedZone, difficulty })
      setQuizNumber(batch.quiz_number)
      const [first, ...rest] = batch.questions
      setQuestionQueue(rest)
      setQuiz({ question: first, result: null, selected: null })
      setPhase('question')
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Failed to load questions.'
      setError(msg)
      setPhase('setup')
    }
  }

  // Submit the user's chosen option and record the result
  async function handleAnswer(letter: string) {
    if (!quiz || phase !== 'question') return
    const { question } = quiz

    setQuiz((prev) => prev && { ...prev, selected: letter })

    try {
      const result = await submitAnswer({
        player_id:      playerId,
        zone:           question.zone,
        question:       question.question,
        correct_answer: question.answer,
        player_answer:  letter,
        difficulty:     question.difficulty,
        explanation:    question.explanation,
      })

      setTopicScores((prev) => {
        const current = prev[question.zone] ?? { correct: 0, total: 0 }
        return {
          ...prev,
          [question.zone]: {
            correct: current.correct + (result.correct ? 1 : 0),
            total:   current.total + 1,
          },
        }
      })
      setQuiz((prev) => prev && { ...prev, result })

      if (result.quiz_complete) {
        setLastQuizScore(result.quiz_score)
        setLevelComplete(result.level_complete)
        setPhase('quiz_complete')
      } else {
        setPhase('answered')
      }
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Failed to submit answer.'
      setError(msg)
    }
  }

  function handleChangeTopic() {
    setQuiz(null)
    setQuestionQueue([])
    setLastQuizScore(null)
    setLevelComplete(false)
    setError(null)
    setPhase('setup')
    // topicScores preserved so the results screen has the full session history
  }

  // ── Setup phase ────────────────────────────────────────────────────────────

  if (phase === 'setup') {
    return (
      <div className="screen">
        <h1 className="quiz__title">Grimoire</h1>
        <p className="quiz__subtitle">Choose a topic and difficulty to begin.</p>

        <div className="quiz__setup">
          <section className="quiz__section">
            <h2 className="quiz__section-label">Topic</h2>
            <div className="quiz__options-row">
              {zones.map((zone) => (
                <button
                  key={zone}
                  className={`quiz__chip ${selectedZone === zone ? 'quiz__chip--active' : ''}`}
                  onClick={() => setSelectedZone(zone)}
                >
                  {zone}
                </button>
              ))}
            </div>
          </section>

          <section className="quiz__section">
            <h2 className="quiz__section-label">Difficulty</h2>
            <div className="quiz__options-row">
              {DIFFICULTIES.map((d) => (
                <button
                  key={d.key}
                  className={`quiz__chip ${difficulty === d.key ? 'quiz__chip--active' : ''}`}
                  onClick={() => setDifficulty(d.key)}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </section>

          {error && <p className="quiz__error">{error}</p>}

          <button className="quiz__btn-primary" onClick={fetchNextQuestion}>
            {totalAnswered > 0 ? 'Continue' : 'Start Quiz'}
          </button>

          <div className="quiz__actions">
            {totalAnswered > 0 && (
              <button className="quiz__btn-ghost" onClick={() => onResults(topicScores)}>
                See Results
              </button>
            )}
            <button className="quiz__btn-ghost" onClick={onRestart}>
              Upload new documents
            </button>
          </div>
        </div>
      </div>
    )
  }

  // ── Loading phase ──────────────────────────────────────────────────────────

  if (phase === 'loading') {
    return (
      <div className="screen">
        <p className="loading__text">Generating questions…</p>
      </div>
    )
  }

  // ── Question / Answered / Quiz-complete phase ──────────────────────────────

  if (!quiz) return null
  const { question, result, selected } = quiz
  const answered     = phase === 'answered' || phase === 'quiz_complete'
  const quizComplete = phase === 'quiz_complete'

  return (
    <div className="screen">
      {/* Score bar */}
      <div className="quiz__scorebar">
        <span className="quiz__zone-tag">{question.zone}</span>
        <span className="quiz__progress">
          Quiz {quizNumber}/{N_QUIZZES} &middot; Q{QUIZ_SIZE - questionQueue.length}/{QUIZ_SIZE}
        </span>
        <span className="quiz__score">{totalCorrect} / {totalAnswered} correct</span>
      </div>

      {/* Question card */}
      <div className="quiz__card">
        <p className="quiz__question">{question.question}</p>

        <div className="quiz__answers">
          {Object.entries(question.options).map(([letter, text]) => {
            let cls = 'quiz__answer'
            if (answered) {
              if (letter === question.answer) cls += ' quiz__answer--correct'
              else if (letter === selected)   cls += ' quiz__answer--wrong'
            }
            return (
              <button
                key={letter}
                className={cls}
                disabled={answered}
                onClick={() => handleAnswer(letter)}
              >
                <span className="quiz__answer-letter">{letter}</span>
                {text}
              </button>
            )
          })}
        </div>

        {/* Per-question feedback */}
        {answered && result && (
          <div className={`quiz__feedback ${result.correct ? 'quiz__feedback--correct' : 'quiz__feedback--wrong'}`}>
            <p className="quiz__feedback-verdict">
              {result.correct ? 'Correct!' : 'Incorrect.'}
            </p>
            <p className="quiz__feedback-explanation">{result.explanation}</p>
          </div>
        )}

        {/* Quiz-complete banner */}
        {quizComplete && (
          <div className={`quiz__quiz-banner ${levelComplete ? 'quiz__quiz-banner--level' : ''}`}>
            {levelComplete ? (
              <p className="quiz__quiz-banner-text">
                Level complete! You have mastered {question.zone} ({difficulty}).
              </p>
            ) : (
              <p className="quiz__quiz-banner-text">
                Quiz {quizNumber} done — {lastQuizScore}/{QUIZ_SIZE}
                {lastQuizScore !== null && lastQuizScore >= 4 ? ' Passed!' : ' Keep going!'}
              </p>
            )}
          </div>
        )}

        {error && <p className="quiz__error">{error}</p>}
      </div>

      {/* Actions */}
      <div className="quiz__actions">
        {answered && (
          <button className="quiz__btn-primary" onClick={fetchNextQuestion}>
            {quizComplete ? 'Next Quiz' : 'Next Question'}
          </button>
        )}
        <button className="quiz__btn-ghost" onClick={handleChangeTopic}>
          Change Topic
        </button>
        {answered && (
          <button className="quiz__btn-ghost" onClick={() => onResults(topicScores)}>
            See Results
          </button>
        )}
      </div>
    </div>
  )
}
