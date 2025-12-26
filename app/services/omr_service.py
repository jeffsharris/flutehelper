"""
OMR (Optical Music Recognition) orchestration service.

This module provides a high-level interface for extracting musical notes
from sheet music images. It abstracts the underlying OCR implementation,
allowing for easy switching between different recognition approaches.

Currently uses AI Vision (GPT) for recognition, but could be extended
to support other OCR engines or hybrid approaches.

Usage:
    from app.services.omr_service import OMRService

    omr = OMRService()
    music = await omr.process_image("sheet_music.png")
    print(f"Found {len(music.notes)} notes")
"""

from typing import Optional

from ..models.notes import ExtractedMusic
from .ai_vision import AIVisionOMR


class OMRService:
    """
    High-level orchestrator for Optical Music Recognition.

    This service provides a unified interface for extracting notes
    from sheet music images, abstracting the underlying implementation.

    Currently delegates to AIVisionOMR for all recognition tasks.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OMR service.

        Args:
            api_key: OpenAI API key for the underlying AI vision service
        """
        self.ai_omr = AIVisionOMR(api_key=api_key)

    async def process_image(self, image_path: str) -> ExtractedMusic:
        """
        Process a sheet music image and extract notes.

        Args:
            image_path: Path to the sheet music image file

        Returns:
            ExtractedMusic containing the extracted notes, title,
            key signature, and confidence score. Returns empty
            result if extraction fails.
        """
        result = self.ai_omr.extract_notes(image_path)

        if result and result.notes:
            return result

        # Return empty result if extraction fails
        return ExtractedMusic(notes=[], confidence=0.0)
