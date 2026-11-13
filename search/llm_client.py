import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

# "gemini-flash-latest" always points to Google's current Flash model, so the project
# keeps working when older versions are retired. Pin a version with GEMINI_MODEL in .env.
load_dotenv()
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
# Used automatically by the LangChain pipeline when the main model is overloaded.
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-flash-lite-latest")


def get_gemini_client() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    return genai.Client(api_key=api_key)


def generate(prompt: str, model: str = DEFAULT_MODEL) -> str:
    client = get_gemini_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS),
    )
    return response.text.strip()