"""
Image utility functions for Flute Helper.

This module provides utilities for working with images, particularly
for preparing them to be sent to the OpenAI Vision API.

Functions:
    encode_image_base64: Convert an image file to base64 string
    get_media_type: Determine MIME type from file extension
"""

import base64
from pathlib import Path


def encode_image_base64(image_path: str) -> str:
    """
    Read and base64 encode an image file.

    Args:
        image_path: Path to the image file on disk

    Returns:
        Base64-encoded string of the image contents

    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If the file can't be read
    """
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_media_type(image_path: str) -> str:
    """
    Determine the MIME type of an image from its file extension.

    Args:
        image_path: Path to the image file

    Returns:
        MIME type string (e.g., "image/png", "image/jpeg")
        Defaults to "image/png" for unknown extensions
    """
    ext = Path(image_path).suffix.lower()

    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }

    return mime_types.get(ext, "image/png")
