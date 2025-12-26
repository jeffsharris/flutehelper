"""
AI-powered Optical Music Recognition (OMR) service.

This module uses OpenAI's Vision API to extract musical notes from
images of sheet music. It handles the image encoding, API communication,
and parsing of the AI response into structured Note objects.

The extracted notes can then be:
- Displayed directly with standard import mode
- Passed to the AI suggestion service for intelligent arrangement

Key Classes:
    AIVisionOMR: Main service class for extracting notes from images

Usage:
    from app.services.ai_vision import AIVisionOMR

    omr = AIVisionOMR()
    extracted = omr.extract_notes("sheet_music.png")
    for note in extracted.notes:
        print(f"{note.name}{note.octave}")  # E4, F#4, etc.
"""

import json
import re
from typing import Optional

import openai
from openai import OpenAI

from ..config import settings
from ..models.notes import Note, ExtractedMusic, NoteName, Accidental
from ..utils.image import encode_image_base64, get_media_type


# Prompt for the vision model
OMR_PROMPT = """Analyze this sheet music image and extract the melody notes.

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


class OMRServiceError(RuntimeError):
    """Raised when the OMR service cannot complete a request."""


class AIVisionOMR:
    """
    Optical Music Recognition using OpenAI's Vision API.

    This service takes an image of sheet music and extracts the notes
    using GPT's vision capabilities. The notes are returned as structured
    data that can be used for fingering lookup or AI arrangement.

    Attributes:
        client: OpenAI client instance
        model: Model to use for vision requests (from settings)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OMR service.

        Args:
            api_key: OpenAI API key. If not provided, uses settings.OPENAI_API_KEY
        """
        self.client = OpenAI(
            api_key=api_key or settings.OPENAI_API_KEY,
            timeout=settings.OPENAI_TIMEOUT_SECONDS,
            max_retries=0,
        )
        self.model = settings.OPENAI_MODEL

    def extract_notes(self, image_path: str) -> ExtractedMusic:
        """
        Extract notes from a sheet music image.

        Uses GPT Vision to analyze the image and identify musical notes.
        The model focuses on the melody line and returns structured data.

        Args:
            image_path: Path to the image file (PNG, JPG, GIF, or WebP)

        Returns:
            ExtractedMusic containing:
            - notes: List of Note objects in order of appearance
            - title: Song title if detected
            - key_signature: Key signature if detected
            - confidence: Confidence score (currently fixed at 0.85)

        Raises:
            ValueError: If the AI response cannot be parsed as JSON
            FileNotFoundError: If the image file doesn't exist
        """
        # Encode the image
        image_data = encode_image_base64(image_path)
        media_type = get_media_type(image_path)

        # Call the API
        try:
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
                                "text": OMR_PROMPT
                            }
                        ],
                    }
                ],
                max_completion_tokens=4096,
            )
        except Exception as exc:
            raise OMRServiceError(self._format_error_message(exc)) from exc

        response_text = response.choices[0].message.content
        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> ExtractedMusic:
        """
        Parse the AI response into structured ExtractedMusic.

        Handles common response formats:
        - Plain JSON
        - JSON wrapped in markdown code blocks
        - JSON embedded in other text

        Args:
            response_text: Raw text response from the API

        Returns:
            ExtractedMusic with parsed notes

        Raises:
            ValueError: If no valid JSON can be extracted
        """
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        # Try to parse as JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            # Try to extract JSON from mixed content
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse AI response as JSON: {e}")

        # Parse notes
        notes = []
        for n in data.get("notes", []):
            try:
                note = self._parse_single_note(n)
                if note:
                    notes.append(note)
            except (ValueError, KeyError):
                continue  # Skip malformed notes

        return ExtractedMusic(
            notes=notes,
            title=data.get("title"),
            key_signature=data.get("key_signature"),
            confidence=0.85
        )

    def _parse_single_note(self, note_data: dict) -> Optional[Note]:
        """
        Parse a single note from the AI response.

        Args:
            note_data: Dictionary with 'name', 'octave', and optional 'accidental'

        Returns:
            Note object, or None if the note data is invalid
        """
        note_name = note_data.get("name", "").upper()
        if note_name not in ["C", "D", "E", "F", "G", "A", "B"]:
            return None

        accidental_str = note_data.get("accidental", "natural").lower()
        if accidental_str == "sharp":
            accidental = Accidental.SHARP
        elif accidental_str == "flat":
            accidental = Accidental.FLAT
        else:
            accidental = Accidental.NATURAL

        return Note(
            name=NoteName(note_name),
            octave=int(note_data.get("octave", 4)),
            accidental=accidental
        )

    def _format_error_message(self, exc: Exception) -> str:
        """Return a user-facing error message for vision failures."""
        if hasattr(openai, "APITimeoutError") and isinstance(exc, openai.APITimeoutError):
            return (
                f"AI vision request timed out after {settings.OPENAI_TIMEOUT_SECONDS}s "
                "while extracting notes."
            )
        if hasattr(openai, "RateLimitError") and isinstance(exc, openai.RateLimitError):
            return "AI vision request was rate limited. Please try again in a moment."
        if hasattr(openai, "APIConnectionError") and isinstance(exc, openai.APIConnectionError):
            return "Could not connect to the AI vision service. Please check your connection and try again."
        if hasattr(openai, "APIError") and isinstance(exc, openai.APIError):
            return "AI vision service returned an error. Please try again."
        return f"AI vision request failed: {exc}"
