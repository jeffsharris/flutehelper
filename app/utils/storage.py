"""
Data persistence utilities for Flute Helper.

This module handles all JSON file-based storage for the application,
including flute profiles, saved songs, and user settings.

The data is stored in the app/data/ directory as JSON files:
- profiles.json: User's flute profiles with custom fingerings
- songs.json: Saved song arrangements
- settings.json: Application settings (e.g., last used profile)

Functions:
    load_json / save_json: Generic JSON file operations
    load_profiles / save_profiles: Flute profile persistence
    load_songs / save_songs: Song library persistence
    get_settings / save_settings: Application settings
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


# Directory paths
APP_DIR = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"
SONGS_DIR = DATA_DIR / "songs"

# Ensure directories exist on module load
DATA_DIR.mkdir(parents=True, exist_ok=True)
SONGS_DIR.mkdir(parents=True, exist_ok=True)

# File paths
PROFILES_FILE = DATA_DIR / "profiles.json"
SONGS_FILE = DATA_DIR / "songs.json"
SETTINGS_FILE = DATA_DIR / "settings.json"


def load_json(filepath: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        filepath: Path to the JSON file
        default: Default value to return if file doesn't exist

    Returns:
        Parsed JSON data as a dictionary, or the default value
    """
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)
    return default or {}


def save_json(filepath: Path, data: Dict[str, Any]) -> None:
    """
    Save data to a JSON file with pretty formatting.

    Args:
        filepath: Path to the JSON file
        data: Dictionary to save
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================
# Profiles: Store flute configurations and custom fingerings
# ============================================================

def load_profiles() -> Dict[str, Any]:
    """
    Load all flute profiles.

    Returns:
        Dictionary mapping profile_id to profile data.
        Each profile contains:
        - name: Display name for the flute
        - a4_frequency: Tuning reference (default 440Hz)
        - fingerings: Dict mapping note names to fingering patterns
        - created_at: ISO timestamp
    """
    return load_json(PROFILES_FILE, {})


def save_profiles(profiles: Dict[str, Any]) -> None:
    """Save all flute profiles to disk."""
    save_json(PROFILES_FILE, profiles)


# ============================================================
# Songs: Store saved song arrangements
# ============================================================

def load_songs() -> Dict[str, Any]:
    """
    Load all saved songs.

    Returns:
        Dictionary mapping song_id to song data.
        Each song contains:
        - title: Song title
        - profile_id: Associated flute profile
        - notes: List of note mappings
        - key_signature: Musical key (optional)
        - created_at: ISO timestamp
    """
    return load_json(SONGS_FILE, {})


def save_songs(songs: Dict[str, Any]) -> None:
    """Save all songs to disk."""
    save_json(SONGS_FILE, songs)


# ============================================================
# Settings: Store application preferences
# ============================================================

def get_settings() -> Dict[str, Any]:
    """
    Get application settings.

    Returns:
        Settings dictionary containing:
        - last_profile: ID of the last used flute profile
    """
    return load_json(SETTINGS_FILE, {"last_profile": None})


def save_settings(settings_data: Dict[str, Any]) -> None:
    """Save application settings to disk."""
    save_json(SETTINGS_FILE, settings_data)
