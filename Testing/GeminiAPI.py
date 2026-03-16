# This script provides a simple test for the Gemini API to ensure
# that the GOOGLE_API_KEY is correctly configured and that the API
# can be successfully called.
#
# Created: 2026-03-16
# Author: Devon Vanaenrode

# --- Imports ---

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Globals ---

# Load environment variables from .env file in the parent directory
# It's recommended to set the GOOGLE_API_KEY as an environment variable
# for security reasons. This script will read it from the .env file.
dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")

# --- Helpers ---

def test_gemini_api():
    """
    Tests the ChatGoogleGenerativeAI model to ensure it's working with the API key.
    """
    try:
        # 1. Instantiate the model
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=API_KEY)

        # 2. Invoke the model with a simple prompt
        prompt = "Hello, what is the capital of Finland?"
        result = llm.invoke(prompt)

        # 3. Get the content of the response
        response_content = result.text

        # 4. Assert that the response is a non-empty string
        assert isinstance(response_content, str), "Response is not a string."
        assert len(response_content.strip()) > 0, "Response is empty."

        # 5. Assert that the response contains the expected answer
        assert "helsinki" in response_content.lower(), "Response does not contain the expected answer."

        print("ChatGoogleGenerativeAI test passed!")
        print(f"Prompt: {prompt}")
        print(f"Response: {response_content}")

    except Exception as e:
        print(f"ChatGoogleGenerativeAI test failed: {e}")

# --- Main loop ---

if __name__ == "__main__":
    test_gemini_api()
