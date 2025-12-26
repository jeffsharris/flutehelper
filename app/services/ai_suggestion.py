"""
AI-powered music arrangement service for Native American flute.

This module provides intelligent suggestions for adapting sheet music to work
within the limited note range of Native American flutes. It uses the OpenAI
Responses API with reasoning capabilities to:

1. Analyze extracted notes from sheet music (with optional image verification)
2. Recommend optimal transposition for the user's specific flute
3. Suggest note substitutions when notes aren't playable
4. Provide musical context and arrangement advice

Key Classes:
    StreamEvent: Event emitted during streaming for real-time updates
    NoteSuggestion: Mapping from original note to suggested playable note
    AISuggestion: Complete arrangement suggestion with all mappings
    AISuggestionService: Main service class for generating suggestions

Usage:
    from app.services.ai_suggestion import ai_suggestion_service

    # Non-streaming (waits for complete response)
    suggestion = ai_suggestion_service.get_suggestions(
        extracted_music, profile_fingerings, image_path="song.png"
    )

    # Streaming (yields events as AI processes)
    for event in ai_suggestion_service.get_suggestions_streaming(...):
        if event.type == "reasoning_delta":
            print(event.data, end="")  # Real-time reasoning
        elif event.type == "complete":
            result = event.final_response
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

import openai
from openai import OpenAI

from ..config import settings
from ..models.notes import ExtractedMusic
from ..utils.image import encode_image_base64, get_media_type


# ============================================================
# Data Classes
# ============================================================

@dataclass
class StreamEvent:
    """
    Event emitted during streaming AI responses.

    Used to provide real-time updates to the frontend as the AI
    processes the arrangement request.

    Attributes:
        type: Event type - one of:
            - 'status': Step-level progress update
            - 'reasoning_delta': Chunk of reasoning summary text
            - 'text_delta': Chunk of output text (JSON response)
            - 'complete': Final parsed response ready
            - 'error': Error occurred during processing
        data: Event payload (text chunk, status dict, or error message)
        final_response: Complete AISuggestion (only for 'complete' type)
    """
    type: str
    data: Any
    final_response: Optional['AISuggestion'] = None


@dataclass
class NoteSuggestion:
    """
    Mapping from an original note to a suggested playable note.

    Attributes:
        original: Original note from sheet music (e.g., "C4")
        transposed: Note after transposition (e.g., "D4")
        suggested: Final note to play, may differ if substituted (e.g., "D4")
        playable: Whether the suggested note is playable on this flute
        substitution_reason: Explanation if a substitution was made
    """
    original: str
    transposed: str
    suggested: str
    playable: bool
    substitution_reason: Optional[str] = None


@dataclass
class AISuggestion:
    """
    Complete AI-generated arrangement suggestion.

    Contains all the information needed to display and save an
    AI-arranged version of a song for a specific flute.

    Attributes:
        recommended_transposition: Semitones to shift (+ = up, - = down)
        transposition_reasoning: AI's explanation for the transposition choice
        note_mappings: List of NoteSuggestion for each note in the song
        musical_notes: General arrangement advice and performance tips
        original_key: Original key signature from sheet music
        suggested_key: New key after transposition
        ocr_corrections: Description of OCR errors the AI corrected
        reasoning_summary: AI's reasoning process summary (from extended thinking)

        Debug fields (for troubleshooting):
        debug_raw_response: Raw text response from the AI
        debug_parse_error: Any JSON parsing error that occurred
        debug_model_used: Model that generated the response
        debug_input_notes: Notes sent to the AI
        debug_available_notes: Available fingerings on the flute
        debug_request: Full API request payload
    """
    recommended_transposition: int
    transposition_reasoning: str
    note_mappings: List[NoteSuggestion]
    musical_notes: str
    original_key: Optional[str] = None
    suggested_key: Optional[str] = None
    ocr_corrections: Optional[str] = None
    reasoning_summary: Optional[str] = None
    # Debug fields
    debug_raw_response: Optional[str] = None
    debug_parse_error: Optional[str] = None
    debug_model_used: Optional[str] = None
    debug_input_notes: Optional[List[str]] = None
    debug_available_notes: Optional[List[str]] = None
    debug_request: Optional[Dict[str, Any]] = None


# ============================================================
# Constants
# ============================================================

# System instructions for the AI model
SYSTEM_INSTRUCTIONS = """You are a music arrangement expert specializing in Native American flute. \
You help adapt sheet music to work within the limited note range of these instruments \
while preserving the musical character of the piece."""

# Reasoning configuration - use medium effort for balance of quality and speed
REASONING_CONFIG = {"effort": "medium", "summary": "detailed"}

# Max visible output tokens
MAX_OUTPUT_TOKENS = 4096

# JSON schema for structured output
AI_SUGGESTION_SCHEMA = {
    "name": "ai_suggestion",
    "schema": {
        "type": "object",
        "properties": {
            "recommended_transposition": {"type": "integer"},
            "transposition_reasoning": {"type": "string"},
            "suggested_key": {"type": ["string", "null"]},
            "ocr_corrections": {"type": ["string", "null"]},
            "note_mappings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "original": {"type": "string"},
                        "transposed": {"type": "string"},
                        "suggested": {"type": "string"},
                        "playable": {"type": "boolean"},
                        "substitution_reason": {"type": ["string", "null"]},
                    },
                    "required": ["original", "transposed", "suggested", "playable"],
                    "additionalProperties": False,
                },
            },
            "musical_notes": {"type": "string"},
        },
        "required": [
            "recommended_transposition",
            "transposition_reasoning",
            "suggested_key",
            "ocr_corrections",
            "note_mappings",
            "musical_notes",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}

TEXT_FORMAT = {"format": {"type": "json_schema", "json_schema": AI_SUGGESTION_SCHEMA}}


# ============================================================
# Service Class
# ============================================================

class AISuggestionService:
    """
    Service for generating AI-powered arrangement suggestions.

    This service uses the OpenAI Responses API with vision and reasoning
    to analyze sheet music and suggest optimal arrangements for Native
    American flute.

    The service supports both synchronous and streaming modes:
    - get_suggestions(): Blocks until complete response
    - get_suggestions_streaming(): Yields events for real-time updates
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AI suggestion service.

        Args:
            api_key: OpenAI API key. If not provided, uses settings.OPENAI_API_KEY
        """
        self.client = OpenAI(
            api_key=api_key or settings.OPENAI_API_KEY,
            timeout=settings.OPENAI_TIMEOUT_SECONDS,
            max_retries=0,
        )
        self.model = settings.OPENAI_MODEL

    def get_suggestions(
        self,
        extracted_music: ExtractedMusic,
        profile_fingerings: Dict[str, Any],
        image_path: Optional[str] = None
    ) -> AISuggestion:
        """
        Get AI suggestions for arranging music for a specific flute.

        This is the synchronous version that waits for the complete response.
        Use get_suggestions_streaming() for real-time updates.

        Args:
            extracted_music: Notes extracted from sheet music via OCR
            profile_fingerings: Dict mapping note names (e.g., "E4") to fingering data
            image_path: Optional path to original image for AI visual verification

        Returns:
            AISuggestion with recommended transposition and note mappings
        """
        # Prepare the request
        note_list = self._extract_note_list(extracted_music)
        available_notes = sorted(profile_fingerings.keys())
        prompt = self._build_prompt(note_list, available_notes, extracted_music)
        input_content = self._build_input_content(prompt, image_path)
        request_params = self._build_request_params(input_content)
        request_payload = self._build_request_payload(input_content)

        # Build debug info
        debug_info = {
            "raw_response": "",
            "model_used": self.model,
            "input_notes": note_list,
            "available_notes": available_notes,
            "request": request_payload,
        }

        try:
            # Call the API
            response = self.client.responses.create(**request_params)
        except Exception as exc:
            error_message = self._format_error_message(exc)
            debug_info["raw_response"] = ""
            return self._create_fallback_response(
                extracted_music.key_signature,
                reasoning_summary=None,
                parse_error=error_message,
                debug_info=debug_info,
            )

        # Extract reasoning summary
        reasoning_summary = self._extract_reasoning_summary(response)

        # Get text response (may be empty if model only produced reasoning)
        response_text = response.output_text or ""

        # Build debug info
        debug_info["raw_response"] = response_text

        return self._parse_response(
            response_text, extracted_music.key_signature, reasoning_summary, debug_info
        )

    def get_suggestions_streaming(
        self,
        extracted_music: ExtractedMusic,
        profile_fingerings: Dict[str, Any],
        image_path: Optional[str] = None
    ) -> Generator[StreamEvent, None, None]:
        """
        Stream AI suggestions with real-time reasoning display.

        Yields StreamEvent objects as the AI processes the request,
        allowing the frontend to show reasoning in real-time.

        Args:
            extracted_music: Notes extracted from sheet music via OCR
            profile_fingerings: Dict mapping note names to fingering data
            image_path: Optional path to original image for verification

        Yields:
            StreamEvent objects:
            - type='status': Step-level progress update
            - type='reasoning_delta': Partial reasoning text
            - type='text_delta': Partial output text (usually not displayed)
            - type='complete': Final parsed AISuggestion
            - type='error': Error message if something fails
        """
        # Prepare the request
        note_list = self._extract_note_list(extracted_music)
        available_notes = sorted(profile_fingerings.keys())
        prompt = self._build_prompt(note_list, available_notes, extracted_music)
        input_content = self._build_input_content(prompt, image_path)
        request_params = self._build_request_params(input_content)
        request_payload = self._build_request_payload(input_content)

        # Debug info (updated as we stream)
        debug_info = {
            "raw_response": "",
            "model_used": self.model,
            "input_notes": note_list,
            "available_notes": available_notes,
            "request": request_payload,
        }

        # Collect streamed content
        reasoning_parts: List[str] = []
        text_parts: List[str] = []
        status_emitted: set[str] = set()
        saw_reasoning = False
        saw_output = False

        try:
            status = self._build_status_event(
                "request_prepared", "Prepared AI request", status_emitted
            )
            if status:
                yield status

            status = self._build_status_event(
                "request_sent", "Request sent to model", status_emitted
            )
            if status:
                yield status

            with self.client.responses.stream(**request_params) as stream:
                for event in stream:
                    if event.type in ("response.created", "response.in_progress"):
                        status = self._build_status_event(
                            "model_processing", "Model is processing", status_emitted
                        )
                        if status:
                            yield status
                    elif event.type == "response.completed":
                        status = self._build_status_event(
                            "model_completed", "Model completed response", status_emitted
                        )
                        if status:
                            yield status
                    elif event.type == "response.reasoning_summary_text.delta":
                        if not saw_reasoning:
                            saw_reasoning = True
                            status = self._build_status_event(
                                "reasoning_streaming",
                                "Receiving reasoning summary",
                                status_emitted,
                            )
                            if status:
                                yield status
                        reasoning_parts.append(event.delta)
                        yield StreamEvent(type="reasoning_delta", data=event.delta)
                    elif event.type == "response.output_text.delta":
                        if not saw_output:
                            saw_output = True
                            status = self._build_status_event(
                                "response_streaming",
                                "Receiving response JSON",
                                status_emitted,
                            )
                            if status:
                                yield status
                        text_parts.append(event.delta)
                        yield StreamEvent(type="text_delta", data=event.delta)
                    elif event.type in ("response.failed", "response.incomplete", "response.error"):
                        error_message = "AI response was interrupted. Please try again."
                        yield StreamEvent(type="error", data=error_message)
                        return

                # Get final response for any additional data
                final_response = stream.get_final_response()

            # Assemble complete texts
            full_reasoning = "".join(reasoning_parts) or self._extract_reasoning_summary(
                final_response
            )
            full_text = "".join(text_parts)
            debug_info["raw_response"] = full_text

            status = self._build_status_event(
                "parsing_response", "Parsing AI response", status_emitted
            )
            if status:
                yield status

            # Parse and yield final result
            result = self._parse_response(
                full_text, extracted_music.key_signature, full_reasoning, debug_info
            )
            yield StreamEvent(type="complete", data="", final_response=result)

        except Exception as e:
            yield StreamEvent(type="error", data=self._format_error_message(e))

    # ============================================================
    # Private Helper Methods
    # ============================================================

    def _extract_note_list(self, extracted_music: ExtractedMusic) -> List[str]:
        """Convert ExtractedMusic notes to string list like ['E4', 'F#4', ...]."""
        note_list = []
        for note in extracted_music.notes:
            note_key = note.name.value
            if note.accidental.value == "sharp":
                note_key += "#"
            elif note.accidental.value == "flat":
                note_key += "b"
            note_key += str(note.octave)
            note_list.append(note_key)
        return note_list

    def _build_input_content(
        self, prompt: str, image_path: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Build the input content array for the API request."""
        content: List[Dict[str, Any]] = []

        if image_path:
            image_data = encode_image_base64(image_path)
            media_type = get_media_type(image_path)
            content.append({
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{image_data}"
            })

        content.append({"type": "input_text", "text": prompt})
        return content

    def _build_request_params(
        self, input_content: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build request parameters for the API call."""
        return {
            "model": self.model,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": [{"role": "user", "content": input_content}],
            "reasoning": REASONING_CONFIG,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "text": TEXT_FORMAT,
        }

    def _build_request_payload(
        self, input_content: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build request payload for debug logging (with truncated images)."""
        sanitized = self._sanitize_for_debug(input_content)
        return self._build_request_params(sanitized)

    def _sanitize_for_debug(
        self, input_content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Truncate base64 images in input content for readable debug output."""
        sanitized = []
        for item in input_content:
            if item.get("type") == "input_image":
                url = item.get("image_url", "")
                if url.startswith("data:") and len(url) > 200:
                    sanitized.append({
                        "type": "input_image",
                        "image_url": f"{url[:100]}...[truncated, {len(url)} chars]"
                    })
                else:
                    sanitized.append(item)
            else:
                sanitized.append(item)
        return sanitized

    def _extract_reasoning_summary(self, response) -> Optional[str]:
        """Extract reasoning summary from API response output items."""
        for item in response.output:
            if item.type == "reasoning" and hasattr(item, 'summary') and item.summary:
                texts = [s.text for s in item.summary if hasattr(s, 'text') and s.text]
                if texts:
                    return "\n".join(texts)
        return None

    def _build_status_event(
        self, stage: str, message: str, status_emitted: set[str]
    ) -> Optional[StreamEvent]:
        """Build a status event only once per stage."""
        if stage in status_emitted:
            return None
        status_emitted.add(stage)
        return StreamEvent(type="status", data={"stage": stage, "message": message})

    def _format_error_message(self, exc: Exception) -> str:
        """Return a user-facing error message for OpenAI failures."""
        if hasattr(openai, "APITimeoutError") and isinstance(exc, openai.APITimeoutError):
            return (
                f"AI request timed out after {settings.OPENAI_TIMEOUT_SECONDS}s. "
                "Please try again."
            )
        if hasattr(openai, "RateLimitError") and isinstance(exc, openai.RateLimitError):
            return "AI request was rate limited. Please try again in a moment."
        if hasattr(openai, "APIConnectionError") and isinstance(exc, openai.APIConnectionError):
            return "Could not connect to the AI service. Please check your connection and try again."
        if hasattr(openai, "APIError") and isinstance(exc, openai.APIError):
            return "AI service returned an error. Please try again."
        return f"AI request failed: {exc}"

    def _build_prompt(
        self,
        note_list: List[str],
        available_notes: List[str],
        extracted_music: ExtractedMusic
    ) -> str:
        """Build the prompt for the AI model."""
        notes_str = ", ".join(note_list) if note_list else "No notes extracted"
        available_str = ", ".join(available_notes) if available_notes else "No fingerings recorded yet"
        key_info = f"Original key: {extracted_music.key_signature}" if extracted_music.key_signature else "Key signature not detected"
        song_context = f"\nSONG TITLE: {extracted_music.title}\n" if extracted_music.title else ""

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
        original_key: Optional[str],
        reasoning_summary: Optional[str] = None,
        debug_info: Optional[Dict[str, Any]] = None
    ) -> AISuggestion:
        """
        Parse the AI response text into a structured AISuggestion.

        Handles various edge cases:
        - Empty responses
        - Markdown code blocks around JSON
        - Partial JSON extraction from mixed content
        """
        debug_info = debug_info or {}
        parse_error = None

        # Handle empty response
        text = (response_text or "").strip()
        if not text:
            parse_error = "Empty response from AI model"
            return self._create_fallback_response(
                original_key, reasoning_summary, parse_error, debug_info
            )

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Skip first line (```json) and last line (```)
            text = "\n".join(lines[1:-1] if len(lines) > 2 else lines)

        # Try to parse JSON
        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            parse_error = f"JSON decode error: {e}"
            # Try to extract JSON object from mixed content
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    parse_error = None  # Successfully recovered
                except json.JSONDecodeError as e2:
                    parse_error = f"JSON extraction failed: {e2}"

        if data is None:
            return self._create_fallback_response(
                original_key, reasoning_summary, parse_error, debug_info
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
            ocr_corrections=data.get("ocr_corrections"),
            reasoning_summary=reasoning_summary,
            debug_raw_response=debug_info.get("raw_response"),
            debug_parse_error=parse_error,
            debug_model_used=debug_info.get("model_used"),
            debug_input_notes=debug_info.get("input_notes"),
            debug_available_notes=debug_info.get("available_notes"),
            debug_request=debug_info.get("request"),
        )

    def _create_fallback_response(
        self,
        original_key: Optional[str],
        reasoning_summary: Optional[str],
        parse_error: str,
        debug_info: Dict[str, Any]
    ) -> AISuggestion:
        """Create a fallback response when parsing fails."""
        return AISuggestion(
            recommended_transposition=0,
            transposition_reasoning="Could not parse AI response",
            note_mappings=[],
            musical_notes="Error processing AI suggestions. Please try again.",
            original_key=original_key,
            suggested_key=original_key,
            reasoning_summary=reasoning_summary,
            debug_raw_response=debug_info.get("raw_response"),
            debug_parse_error=parse_error,
            debug_model_used=debug_info.get("model_used"),
            debug_input_notes=debug_info.get("input_notes"),
            debug_available_notes=debug_info.get("available_notes"),
            debug_request=debug_info.get("request"),
        )


# Singleton instance for use throughout the application
ai_suggestion_service = AISuggestionService()
