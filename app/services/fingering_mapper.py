from typing import List, Optional

from ..models.notes import Note, ExtractedMusic, Accidental, NoteName
from ..models.fingerings import Fingering, FingeringResult, SubstituteNote
from ..data.fingering_charts import (
    E_FLUTE_FINGERINGS,
    get_fingering,
    is_in_range,
    normalize_note_name,
    FLAT_TO_SHARP,
)

# MIDI to note name mapping
MIDI_TO_NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


class FingeringMapper:
    """Maps musical notes to Native American flute fingerings."""

    def __init__(self, flute_key: str = "E"):
        self.flute_key = flute_key
        self.fingerings = E_FLUTE_FINGERINGS

    def map_notes(self, music: ExtractedMusic) -> List[FingeringResult]:
        """Map all extracted notes to flute fingerings."""
        results = []
        for note in music.notes:
            result = self._map_single_note(note)
            results.append(result)
        return results

    def _map_single_note(self, note: Note) -> FingeringResult:
        """Map a single note to its fingering."""
        note_key = self._build_note_key(note)

        # Check if in playable range
        note_name_with_acc = self._get_note_name_with_accidental(note)
        if not is_in_range(note_name_with_acc, note.octave):
            substitutes = self._find_substitute_notes(note)
            return FingeringResult(
                original_note=note,
                fingering=None,
                playable=False,
                transposition_note=self._suggest_transposition(note),
                substitute_notes=substitutes
            )

        # Look up fingering
        fingering_tuple = get_fingering(note_name_with_acc, note.octave)

        if fingering_tuple:
            fingering = Fingering(
                holes=fingering_tuple,
                note_name=note.display_name,
                is_primary=True
            )
            return FingeringResult(
                original_note=note,
                fingering=fingering,
                playable=True
            )

        # Note is in range but no direct fingering - find substitutes
        substitutes = self._find_substitute_notes(note)
        return FingeringResult(
            original_note=note,
            fingering=None,
            playable=False,
            transposition_note=f"No fingering for {note.display_name}",
            substitute_notes=substitutes
        )

    def _build_note_key(self, note: Note) -> str:
        """Build the lookup key for a note."""
        return note.display_name

    def _get_note_name_with_accidental(self, note: Note) -> str:
        """Get note name with accidental symbol."""
        if note.accidental == Accidental.SHARP:
            return f"{note.name.value}#"
        elif note.accidental == Accidental.FLAT:
            return f"{note.name.value}b"
        return note.name.value

    def _suggest_transposition(self, note: Note) -> str:
        """Suggest how to handle out-of-range notes."""
        midi = note.midi_number

        if midi < 52:  # Below E4
            octave_diff = (52 - midi) // 12 + 1
            return f"Transpose up {octave_diff} octave(s)"
        elif midi > 71:  # Above B5
            octave_diff = (midi - 71) // 12 + 1
            return f"Transpose down {octave_diff} octave(s)"

        return "Consider using a different key flute"

    def _find_substitute_notes(self, note: Note) -> List[SubstituteNote]:
        """Find nearby playable notes as substitutes."""
        substitutes = []
        original_midi = note.midi_number

        # Search within +/- 3 semitones for substitutes
        for offset in [-1, 1, -2, 2, -3, 3]:
            target_midi = original_midi + offset
            target_note_name = self._midi_to_note_name(target_midi)
            target_octave = (target_midi // 12) - 1

            # Extract just the note name part for lookup
            if '#' in target_note_name:
                lookup_name = target_note_name
            else:
                lookup_name = target_note_name

            fingering_tuple = get_fingering(lookup_name, target_octave)

            if fingering_tuple:
                fingering = Fingering(
                    holes=fingering_tuple,
                    note_name=f"{target_note_name}{target_octave}",
                    is_primary=False
                )
                substitutes.append(SubstituteNote(
                    note_name=f"{target_note_name}{target_octave}",
                    semitones_away=offset,
                    fingering=fingering
                ))

                # Only suggest up to 2 substitutes
                if len(substitutes) >= 2:
                    break

        return substitutes

    def _midi_to_note_name(self, midi: int) -> str:
        """Convert MIDI number to note name."""
        note_index = midi % 12
        return MIDI_TO_NOTE[note_index]
