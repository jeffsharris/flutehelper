from pydantic import BaseModel
from typing import Tuple, Optional, List
from .notes import Note


class Fingering(BaseModel):
    """
    Represents a fingering for a 6-hole Native American flute.
    Holes are numbered 1-6 from top (near mouthpiece) to bottom.
    True = covered (closed), False = open
    """
    holes: Tuple[bool, bool, bool, bool, bool, bool]
    note_name: str
    is_primary: bool = True
    technique_note: Optional[str] = None

    @property
    def as_symbols(self) -> str:
        """Return visual representation using filled/empty circles."""
        return "".join(["●" if h else "○" for h in self.holes])


class SubstituteNote(BaseModel):
    """A suggested substitute note when original isn't playable."""
    note_name: str
    semitones_away: int  # Positive = higher, negative = lower
    fingering: Optional[Fingering] = None


class FingeringResult(BaseModel):
    """Result of mapping a note to a flute fingering."""
    original_note: Note
    fingering: Optional[Fingering] = None
    playable: bool
    transposition_note: Optional[str] = None
    substitute_notes: List[SubstituteNote] = []

    class Config:
        arbitrary_types_allowed = True
