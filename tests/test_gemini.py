import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

from google import genai

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model='gemma-3-27b-it', contents='"Explain retrieval-augmented generation in one short paragraph."'
)
print(response.text)
print("Prompt tokens:", response.usage_metadata.prompt_token_count)
print("Response tokens:", response.usage_metadata.candidates_token_count)

