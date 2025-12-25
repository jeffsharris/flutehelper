"""
6-hole Native American Flute Fingering Chart for Key of E

Holes: (Top/Left hand: 1,2,3) (Bottom/Right hand: 4,5,6)
True = covered (closed), False = open

The flute's fundamental (all holes closed) = E4
Standard minor pentatonic: E - G - A - B - D - E (high)

Cross-fingerings allow chromatic notes but may require more skill.
"""

from typing import Optional, Tuple, Dict

# Fingering definitions: note_name -> (hole1, hole2, hole3, hole4, hole5, hole6)
# Holes 1-3 are top (left hand), 4-6 are bottom (right hand)
E_FLUTE_FINGERINGS: Dict[str, Tuple[bool, bool, bool, bool, bool, bool]] = {
    # First octave (fundamental)
    "E4": (True, True, True, True, True, True),      # All closed - fundamental
    "F#4": (True, True, True, True, True, False),    # Bottom hole open
    "G4": (True, True, True, True, False, False),    # Pentatonic note
    "G#4": (True, True, True, False, True, False),   # Cross-fingering
    "A4": (True, True, True, False, False, False),   # Pentatonic note
    "A#4": (True, True, False, True, True, False),   # Cross-fingering
    "B4": (True, True, False, False, False, False),  # Pentatonic note
    "C5": (True, False, True, False, False, False),  # Cross-fingering
    "C#5": (True, False, False, False, False, False),
    "D5": (False, True, False, False, False, False), # Pentatonic note
    "D#5": (False, False, True, False, False, False),# Cross-fingering

    # Second octave (overblown - same fingerings, more breath pressure)
    "E5": (True, True, True, True, True, True),      # Overblow
    "F#5": (True, True, True, True, True, False),
    "G5": (True, True, True, True, False, False),
    "G#5": (True, True, True, False, True, False),
    "A5": (True, True, True, False, False, False),
    "A#5": (True, True, False, True, True, False),
    "B5": (True, True, False, False, False, False),
}

# Flat equivalents (for lookup convenience)
FLAT_TO_SHARP = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
}


def normalize_note_name(note_name: str, octave: int) -> str:
    """Convert flats to sharps and build lookup key."""
    # Handle flats
    for flat, sharp in FLAT_TO_SHARP.items():
        if flat in note_name:
            note_name = note_name.replace(flat, sharp)
            break

    # Build key
    if "#" in note_name or len(note_name) == 1:
        return f"{note_name}{octave}"
    return f"{note_name}{octave}"


def get_fingering(note_name: str, octave: int) -> Optional[Tuple[bool, bool, bool, bool, bool, bool]]:
    """Get fingering for a specific note."""
    key = normalize_note_name(note_name, octave)
    return E_FLUTE_FINGERINGS.get(key)


def is_in_range(note_name: str, octave: int) -> bool:
    """Check if note is within the flute's playable range (E4 to B5)."""
    midi_base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

    # Extract base note
    base = note_name[0]
    if base not in midi_base:
        return False

    midi = (octave + 1) * 12 + midi_base[base]

    # Handle accidentals
    if "#" in note_name:
        midi += 1
    elif "b" in note_name.lower():
        midi -= 1

    # E4 = 52, B5 = 71
    return 52 <= midi <= 71


def get_playable_range() -> str:
    """Return description of playable range."""
    return "E4 to B5 (with cross-fingerings for chromatic notes)"
