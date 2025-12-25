from pydantic import BaseModel
from typing import Tuple, Optional
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


class FingeringResult(BaseModel):
    """Result of mapping a note to a flute fingering."""
    original_note: Note
    fingering: Optional[Fingering] = None
    playable: bool
    transposition_note: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
