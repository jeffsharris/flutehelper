from typing import Optional

from ..models.notes import ExtractedMusic
from .ai_vision import AIVisionOMR


class OMRService:
    """Orchestrates OMR processing."""

    def __init__(self, api_key: Optional[str] = None):
        self.ai_omr = AIVisionOMR(api_key=api_key)

    async def process_image(self, image_path: str) -> ExtractedMusic:
        """Process sheet music image and extract notes."""
        # Use AI Vision for OMR
        result = self.ai_omr.extract_notes(image_path)

        if result and result.notes:
            return result

        # Return empty result if extraction fails
        return ExtractedMusic(notes=[], confidence=0.0)
