import base64
import json
from pathlib import Path
from typing import Optional

from openai import OpenAI

from ..config import settings
from ..models.notes import Note, ExtractedMusic, NoteName, Accidental


class AIVisionOMR:
    """Optical Music Recognition using OpenAI's Vision API."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL

    def extract_notes(self, image_path: str) -> ExtractedMusic:
        """Use GPT Vision to extract notes from sheet music."""

        # Load and encode image
        image_data = self._encode_image(image_path)
        media_type = self._get_media_type(image_path)

        prompt = """Analyze this sheet music image and extract the melody notes.

Return a JSON object with this exact structure:
{
    "title": "Song title if visible, or null",
    "key_signature": "Key signature if visible (e.g., 'C major', 'G major'), or null",
    "notes": [
        {"name": "E", "octave": 4, "accidental": "natural"},
        {"name": "G", "octave": 4, "accidental": "sharp"},
        ...
    ]
}

Rules:
1. Focus on the MELODY line only (top staff if multiple staves, or the main vocal/instrument line)
2. Use note names: C, D, E, F, G, A, B (uppercase)
3. Use octave numbers where middle C = C4
4. For accidentals use: "natural", "sharp", or "flat"
5. List notes in order of appearance (left to right, following the music)
6. If this is a lead sheet with chord symbols, extract the melody notes from the staff, not the chord symbols
7. Include all notes, even repeated ones

Return ONLY the JSON object, no other text or markdown formatting."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
            max_completion_tokens=4096,
        )

        response_text = response.choices[0].message.content
        return self._parse_response(response_text)

    def _encode_image(self, image_path: str) -> str:
        """Read and base64 encode an image file."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _get_media_type(self, image_path: str) -> str:
        """Determine MIME type from file extension."""
        path = Path(image_path)
        ext = path.suffix.lower()
        if ext == ".png":
            return "image/png"
        elif ext in [".jpg", ".jpeg"]:
            return "image/jpeg"
        elif ext == ".gif":
            return "image/gif"
        elif ext == ".webp":
            return "image/webp"
        return "image/png"

    def _parse_response(self, response_text: str) -> ExtractedMusic:
        """Parse the AI response into structured data."""
        # Clean up response (remove markdown code blocks if present)
        text = response_text.strip()
        if text.startswith("```"):
            # Remove code block markers
            lines = text.split("\n")
            # Skip first line (```json) and last line (```)
            text = "\n".join(lines[1:-1])

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse AI response as JSON: {e}")

        notes = []
        for n in data.get("notes", []):
            try:
                note_name = n.get("name", "").upper()
                if note_name not in ["C", "D", "E", "F", "G", "A", "B"]:
                    continue

                accidental_str = n.get("accidental", "natural").lower()
                if accidental_str == "sharp":
                    accidental = Accidental.SHARP
                elif accidental_str == "flat":
                    accidental = Accidental.FLAT
                else:
                    accidental = Accidental.NATURAL

                notes.append(Note(
                    name=NoteName(note_name),
                    octave=int(n.get("octave", 4)),
                    accidental=accidental
                ))
            except (ValueError, KeyError):
                continue

        return ExtractedMusic(
            notes=notes,
            title=data.get("title"),
            key_signature=data.get("key_signature"),
            confidence=0.85
        )
