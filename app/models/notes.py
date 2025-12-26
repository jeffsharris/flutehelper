"""
Musical note data models.

This module defines the core data structures for representing musical notes
and extracted music from sheet music images.

Classes:
    NoteName: Enum of natural note names (C through B)
    Accidental: Enum for sharp/flat/natural
    Note: A single musical note with name, octave, and accidental
    ExtractedMusic: Collection of notes with metadata from OCR

Usage:
    from app.models.notes import Note, NoteName, Accidental, ExtractedMusic

    # Create a single note
    note = Note(name=NoteName.E, octave=4, accidental=Accidental.NATURAL)
    print(note.display_name)  # "E4"
    print(note.midi_number)   # 52

    # Create extracted music
    music = ExtractedMusic(
        notes=[note],
        title="Amazing Grace",
        key_signature="G major"
    )
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class NoteName(str, Enum):
    """
    Natural note names in Western music.

    Values are uppercase letter names that can be used directly
    in display and serialization.
    """
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    A = "A"
    B = "B"


class Accidental(str, Enum):
    """
    Accidental modifiers for notes.

    - NATURAL: No modification
    - SHARP: Raise pitch by one semitone
    - FLAT: Lower pitch by one semitone
    """
    NATURAL = "natural"
    SHARP = "sharp"
    FLAT = "flat"


class Note(BaseModel):
    """
    A single musical note.

    Represents a pitch with its name, octave, and any accidental.
    Uses scientific pitch notation where middle C = C4.

    Attributes:
        name: The note letter (C through B)
        octave: The octave number (middle C = 4)
        accidental: Sharp, flat, or natural
        duration: Optional duration value (for future use)
    """
    name: NoteName
    octave: int
    accidental: Accidental = Accidental.NATURAL
    duration: Optional[str] = None

    @property
    def midi_number(self) -> int:
        """
        Convert to MIDI number for pitch comparison.

        MIDI numbers are useful for transposition and range checking.
        Middle C (C4) = 60.

        Returns:
            Integer MIDI number (0-127 range for standard notes)
        """
        base_values = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        midi = (self.octave + 1) * 12 + base_values[self.name.value]

        if self.accidental == Accidental.SHARP:
            midi += 1
        elif self.accidental == Accidental.FLAT:
            midi -= 1

        return midi

    @property
    def display_name(self) -> str:
        """
        Get human-readable note name with octave.

        Examples: "E4", "G#4", "Bb5"

        Returns:
            String representation like "C#4" or "Bb5"
        """
        acc = ""
        if self.accidental == Accidental.SHARP:
            acc = "#"
        elif self.accidental == Accidental.FLAT:
            acc = "b"
        return f"{self.name.value}{acc}{self.octave}"


class ExtractedMusic(BaseModel):
    """
    Music extracted from a sheet music image.

    Contains all the notes from the melody line along with
    metadata detected by the OCR process.

    Attributes:
        notes: List of Note objects in order of appearance
        title: Song title if detected from the image
        key_signature: Key signature (e.g., "G major", "D minor")
        time_signature: Time signature (e.g., "4/4") - future use
        confidence: OCR confidence score (0.0 to 1.0)
    """
    notes: List[Note]
    title: Optional[str] = None
    key_signature: Optional[str] = None
    time_signature: Optional[str] = None
    confidence: float = 0.0
