import argparse
import mimetypes

from google.genai import types

from search.llm_client import DEFAULT_MODEL, get_gemini_client


def main():
    parser = argparse.ArgumentParser(description="Multimodal image query rewriting CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--query", type=str, required=True, help="Text query to rewrite based on the image")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        img = f.read()

    client = get_gemini_client()

    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""

    parts = [
        system_prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=parts,
    )

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()