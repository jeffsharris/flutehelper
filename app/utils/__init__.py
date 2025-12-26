"""
Utility modules for Flute Helper.

This package contains shared utility functions used across the application:
- image: Image encoding and media type detection
- storage: JSON file persistence for profiles, songs, and settings
"""

from .image import encode_image_base64, get_media_type
from .storage import (
    load_json, save_json,
    load_profiles, save_profiles,
    load_songs, save_songs,
    get_settings, save_settings,
    DATA_DIR, SONGS_DIR
)

__all__ = [
    'encode_image_base64',
    'get_media_type',
    'load_json',
    'save_json',
    'load_profiles',
    'save_profiles',
    'load_songs',
    'save_songs',
    'get_settings',
    'save_settings',
    'DATA_DIR',
    'SONGS_DIR',
]
