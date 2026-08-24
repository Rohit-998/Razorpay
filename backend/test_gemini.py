"""Quick test: does the Gemini API key work?"""
import os, sys
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

key = os.getenv("GEMINI_API_KEY", "")
print(f"Key found: {'YES' if key else 'NO'}  (first 10 chars: {key[:10]}...)")

try:
    from google import genai
    from google.genai.types import GenerateContentConfig

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="Say hello in one sentence.",
        config=GenerateContentConfig(max_output_tokens=50),
    )
    print(f"\nGemini responded: {response.text}")
except Exception as e:
    print(f"\nGemini error: {type(e).__name__}: {e}")
