from typing import List, Optional

from ..models.notes import Note, ExtractedMusic, Accidental
from ..models.fingerings import Fingering, FingeringResult
from ..data.fingering_charts import (
    E_FLUTE_FINGERINGS,
    get_fingering,
    is_in_range,
    normalize_note_name,
    FLAT_TO_SHARP,
)


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
            return FingeringResult(
                original_note=note,
                fingering=None,
                playable=False,
                transposition_note=self._suggest_transposition(note)
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

        # Note is in range but no fingering found (shouldn't happen with full chart)
        return FingeringResult(
            original_note=note,
            fingering=None,
            playable=False,
            transposition_note=f"No fingering available for {note.display_name}"
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
