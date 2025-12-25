"""Music theory utilities for transposition and note analysis."""

from typing import Optional

# Chromatic scale for transposition
CHROMATIC_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Enharmonic equivalents (flats to sharps)
ENHARMONIC_MAP = {
    'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#',
    'Ab': 'G#', 'Bb': 'A#', 'Cb': 'B',
    # Double sharps/flats (rare but possible)
    'C##': 'D', 'D##': 'E', 'E##': 'F#', 'F##': 'G',
    'G##': 'A', 'A##': 'B', 'B##': 'C#',
}

# Semitone names for display
SEMITONE_NAMES = {
    -6: '-6 (tritone down)',
    -5: '-5 (4th down)',
    -4: '-4 (M3 down)',
    -3: '-3 (m3 down)',
    -2: '-2 (whole step down)',
    -1: '-1 (half step down)',
    0: 'Original key',
    1: '+1 (half step up)',
    2: '+2 (whole step up)',
    3: '+3 (m3 up)',
    4: '+4 (M3 up)',
    5: '+5 (4th up)',
    6: '+6 (tritone up)',
}


def parse_note_key(note_key: str) -> tuple[str, int]:
    """
    Parse a note key like 'C#4' into (note_name, octave).
    Returns ('C#', 4) for 'C#4'.
    """
    if not note_key:
        return ('C', 4)

    # Find where the octave starts (first digit or negative sign)
    octave_start = -1
    for i, char in enumerate(note_key):
        if char.isdigit() or (char == '-' and i > 0):
            octave_start = i
            break

    if octave_start == -1:
        return (note_key, 4)  # Default octave

    note_name = note_key[:octave_start]
    octave = int(note_key[octave_start:])

    return (note_name, octave)


def normalize_note(note_name: str) -> str:
    """Convert flats and enharmonics to sharp notation."""
    # Handle flats
    if 'b' in note_name:
        if note_name in ENHARMONIC_MAP:
            return ENHARMONIC_MAP[note_name]
        # Simple flat conversion
        base = note_name[0]
        idx = CHROMATIC_NOTES.index(base)
        return CHROMATIC_NOTES[(idx - 1) % 12]
    return note_name


def note_to_midi(note_key: str) -> int:
    """Convert a note key to MIDI number (C4 = 60)."""
    note_name, octave = parse_note_key(note_key)
    note_name = normalize_note(note_name)

    if note_name not in CHROMATIC_NOTES:
        # Handle edge cases
        note_name = note_name[0]  # Just use base note

    semitone = CHROMATIC_NOTES.index(note_name)
    return (octave + 1) * 12 + semitone


def midi_to_note(midi: int) -> str:
    """Convert MIDI number to note key (60 = C4)."""
    octave = (midi // 12) - 1
    semitone = midi % 12
    note_name = CHROMATIC_NOTES[semitone]
    return f"{note_name}{octave}"


def transpose_note(note_key: str, semitones: int) -> str:
    """Transpose a note by the given number of semitones."""
    midi = note_to_midi(note_key)
    new_midi = midi + semitones
    return midi_to_note(new_midi)


def transpose_notes(notes: list[dict], semitones: int) -> list[dict]:
    """Transpose a list of note dicts by the given semitones."""
    if semitones == 0:
        return notes

    result = []
    for note in notes:
        transposed = note.copy()
        transposed['note_key'] = transpose_note(note['note_key'], semitones)
        transposed['original_note_key'] = note.get('original_note_key', note['note_key'])
        result.append(transposed)
    return result


def analyze_playability(notes: list[dict], fingerings: dict) -> dict:
    """
    Analyze how many notes are playable with given fingerings.
    Returns stats dict with counts and details.
    """
    total = len(notes)
    playable = 0
    missing = []

    for note in notes:
        note_key = note.get('note_key', '')
        if note_key in fingerings:
            playable += 1
        else:
            if note_key not in missing:
                missing.append(note_key)

    return {
        'total': total,
        'playable': playable,
        'missing_count': total - playable,
        'missing_notes': missing,
        'percentage': round(playable / total * 100) if total > 0 else 0
    }


def find_optimal_transposition(notes: list[dict], fingerings: dict,
                                range_min: int = -6, range_max: int = 6) -> list[dict]:
    """
    Find the best transposition for maximum playability.
    Returns list of options sorted by playability.
    """
    options = []

    for semitones in range(range_min, range_max + 1):
        transposed = transpose_notes(notes, semitones)
        stats = analyze_playability(transposed, fingerings)

        options.append({
            'semitones': semitones,
            'label': SEMITONE_NAMES.get(semitones, f'{semitones:+d} semitones'),
            'playable': stats['playable'],
            'total': stats['total'],
            'missing_count': stats['missing_count'],
            'percentage': stats['percentage'],
            'missing_notes': stats['missing_notes']
        })

    # Sort by playable count (descending), then by absolute semitones (prefer smaller shifts)
    options.sort(key=lambda x: (-x['playable'], abs(x['semitones'])))

    return options


def find_nearest_playable(note_key: str, fingerings: dict, max_distance: int = 6) -> Optional[dict]:
    """
    Find the nearest playable note to the given note.
    Returns dict with suggestion info or None if nothing found.
    """
    midi = note_to_midi(note_key)

    # Search outward from the note
    for distance in range(1, max_distance + 1):
        # Check both directions
        for direction in [1, -1]:
            candidate_midi = midi + (distance * direction)
            candidate_note = midi_to_note(candidate_midi)

            if candidate_note in fingerings:
                return {
                    'original': note_key,
                    'suggestion': candidate_note,
                    'distance': distance * direction,
                    'direction': 'up' if direction > 0 else 'down',
                    'fingering': fingerings[candidate_note]['fingering']
                }

    return None


def get_key_name(original_key: str, semitones: int) -> str:
    """Get the new key name after transposition."""
    if not original_key or semitones == 0:
        return original_key or 'Unknown'

    # Parse the key (e.g., "C major" -> "C")
    parts = original_key.split()
    root = parts[0] if parts else 'C'
    mode = ' '.join(parts[1:]) if len(parts) > 1 else ''

    # Transpose the root
    new_root = transpose_note(root + '4', semitones)
    new_root = new_root[:-1]  # Remove octave

    return f"{new_root} {mode}".strip()
