from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class NoteName(str, Enum):
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    A = "A"
    B = "B"


class Accidental(str, Enum):
    NATURAL = "natural"
    SHARP = "sharp"
    FLAT = "flat"


class Note(BaseModel):
    name: NoteName
    octave: int
    accidental: Accidental = Accidental.NATURAL
    duration: Optional[str] = None

    @property
    def midi_number(self) -> int:
        """Convert to MIDI number for comparison."""
        base_values = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        midi = (self.octave + 1) * 12 + base_values[self.name.value]
        if self.accidental == Accidental.SHARP:
            midi += 1
        elif self.accidental == Accidental.FLAT:
            midi -= 1
        return midi

    @property
    def display_name(self) -> str:
        """Return display name like E4, G#4, Bb4."""
        acc = ""
        if self.accidental == Accidental.SHARP:
            acc = "#"
        elif self.accidental == Accidental.FLAT:
            acc = "b"
        return f"{self.name.value}{acc}{self.octave}"


class ExtractedMusic(BaseModel):
    notes: List[Note]
    title: Optional[str] = None
    key_signature: Optional[str] = None
    time_signature: Optional[str] = None
    confidence: float = 0.0
