"""AI-powered suggestion service for optimizing sheet music for a specific flute."""
import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from openai import OpenAI

from ..config import settings
from ..models.notes import ExtractedMusic


@dataclass
class NoteSuggestion:
    """Suggestion for a single note."""
    original: str  # Original note from sheet music (e.g., "C4")
    transposed: str  # Note after transposition (e.g., "D4")
    suggested: str  # Final suggested note to play (e.g., "D4")
    playable: bool  # Whether the suggested note is playable on this flute
    substitution_reason: Optional[str] = None  # Reason if substituted


@dataclass
class AISuggestion:
    """AI-generated suggestion for arranging a song for a specific flute."""
    recommended_transposition: int  # Semitones (positive = up, negative = down)
    transposition_reasoning: str  # Why this transposition was chosen
    note_mappings: List[NoteSuggestion]  # Mapping for each note
    musical_notes: str  # Overall notes about the arrangement
    original_key: Optional[str] = None  # Original key signature
    suggested_key: Optional[str] = None  # Key after transposition
    ocr_corrections: Optional[str] = None  # Description of any OCR errors corrected


class AISuggestionService:
    """Service for generating AI-powered arrangement suggestions."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL

    def get_suggestions(
        self,
        extracted_music: ExtractedMusic,
        profile_fingerings: Dict[str, Any],
        image_path: Optional[str] = None
    ) -> AISuggestion:
        """
        Get AI suggestions for the best way to play this music on the given flute.

        Args:
            extracted_music: The notes extracted from sheet music
            profile_fingerings: Dict of available fingerings on the user's flute
                               Keys are note names like "E4", "F#4", etc.
            image_path: Path to the original sheet music image for visual verification

        Returns:
            AISuggestion with recommended transposition and note mappings
        """
        # Build list of extracted notes
        note_list = []
        for note in extracted_music.notes:
            note_key = f"{note.name.value}"
            if note.accidental.value == "sharp":
                note_key += "#"
            elif note.accidental.value == "flat":
                note_key += "b"
            note_key += str(note.octave)
            note_list.append(note_key)

        # Get available notes on this flute
        available_notes = sorted(profile_fingerings.keys())

        prompt = self._build_prompt(
            note_list,
            available_notes,
            extracted_music.key_signature,
            extracted_music.title
        )

        # Build message content - include image if available
        user_content = []

        if image_path:
            # Add the original image for visual verification
            image_data = self._encode_image(image_path)
            media_type = self._get_media_type(image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_data}"
                }
            })

        user_content.append({
            "type": "text",
            "text": prompt
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a music arrangement expert specializing in Native American flute. You help adapt sheet music to work within the limited note range of these instruments while preserving the musical character of the piece."
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            max_completion_tokens=4096,
        )

        response_text = response.choices[0].message.content
        return self._parse_response(response_text, extracted_music.key_signature)

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

    def _build_prompt(
        self,
        note_list: List[str],
        available_notes: List[str],
        key_signature: Optional[str],
        song_title: Optional[str] = None
    ) -> str:
        """Build the prompt for the AI model."""
        notes_str = ", ".join(note_list)
        available_str = ", ".join(available_notes) if available_notes else "No fingerings recorded yet"
        key_info = f"Original key: {key_signature}" if key_signature else "Key signature not detected"

        # Include song title context
        song_context = ""
        if song_title:
            song_context = f"""
SONG TITLE: {song_title}
"""

        return f"""You are helping arrange sheet music for a Native American flute with limited notes.

IMPORTANT: I've attached the original sheet music image. Please look at it directly to read the notes - DO NOT rely solely on the OCR extraction below, which may contain errors. Use the image as your primary source of truth.

{key_info}
{song_context}
OCR-EXTRACTED MELODY (for reference only - may contain errors):
{notes_str}

AVAILABLE NOTES ON THIS FLUTE:
{available_str}

TASK:
1. LOOK AT THE SHEET MUSIC IMAGE and read the actual notes. Compare with the OCR extraction above and note any discrepancies.
2. If you recognize the song title, use your knowledge of the melody to verify the notes are correct.
3. Determine the best transposition (in semitones) to maximize playability on this specific flute.
4. Apply the transposition to all notes.
5. For any notes that are STILL not playable after transposition, suggest the closest musically appropriate substitute from the available notes.
6. Your goal is to create an arrangement that sounds correct and musical on this flute.

IMPORTANT RULES:
- TRUST THE IMAGE over the OCR extraction - read the notes yourself from the sheet music
- The Native American flute typically spans about 1.5 octaves
- Prefer transpositions that keep the melody within the flute's natural range
- When substituting notes, prefer notes that maintain the melodic contour (direction of movement)
- For passing tones, a nearby available note is usually acceptable
- For important melodic notes (like phrase endings), try harder to find a good substitute

Return ONLY a JSON object with this exact structure:
{{
  "recommended_transposition": 0,
  "transposition_reasoning": "Explanation of why this transposition was chosen...",
  "suggested_key": "The new key after transposition (e.g., 'E minor')",
  "ocr_corrections": "Description of any OCR errors you noticed and corrected, or null if none",
  "note_mappings": [
    {{"original": "C4", "transposed": "C4", "suggested": "C4", "playable": true}},
    {{"original": "Bb4", "transposed": "Bb4", "suggested": "B4", "playable": true, "substitution_reason": "Bb4 not available, using B4 as nearest option"}}
  ],
  "musical_notes": "Overall notes about this arrangement and any performance suggestions..."
}}

Return ONLY the JSON object, no other text or markdown formatting."""

    def _parse_response(
        self,
        response_text: str,
        original_key: Optional[str]
    ) -> AISuggestion:
        """Parse the AI response into structured data."""
        # Clean up response (remove markdown code blocks if present)
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # Return a fallback response
                return AISuggestion(
                    recommended_transposition=0,
                    transposition_reasoning="Could not parse AI response",
                    note_mappings=[],
                    musical_notes="Error processing AI suggestions. Please try again.",
                    original_key=original_key,
                    suggested_key=original_key
                )

        # Parse note mappings
        note_mappings = []
        for mapping in data.get("note_mappings", []):
            note_mappings.append(NoteSuggestion(
                original=mapping.get("original", ""),
                transposed=mapping.get("transposed", mapping.get("original", "")),
                suggested=mapping.get("suggested", mapping.get("transposed", "")),
                playable=mapping.get("playable", False),
                substitution_reason=mapping.get("substitution_reason")
            ))

        return AISuggestion(
            recommended_transposition=data.get("recommended_transposition", 0),
            transposition_reasoning=data.get("transposition_reasoning", ""),
            note_mappings=note_mappings,
            musical_notes=data.get("musical_notes", ""),
            original_key=original_key,
            suggested_key=data.get("suggested_key"),
            ocr_corrections=data.get("ocr_corrections")
        )


# Singleton instance
ai_suggestion_service = AISuggestionService()
