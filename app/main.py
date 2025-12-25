import json
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Request, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .config import settings
from .services.omr_service import OMRService
from .services.fingering_mapper import FingeringMapper
from .services.ai_suggestion import ai_suggestion_service
from .services.music_utils import (
    transpose_notes, analyze_playability, find_optimal_transposition,
    find_nearest_playable, get_key_name
)

app = FastAPI(
    title="Flute Helper",
    description="Convert sheet music to Native American flute fingerings"
)

# Get the app directory
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
SONGS_DIR = DATA_DIR / "songs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SONGS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

# Initialize services
omr_service = OMRService(api_key=settings.OPENAI_API_KEY)

# ===== Data Storage =====

PROFILES_FILE = DATA_DIR / "profiles.json"
SONGS_FILE = DATA_DIR / "songs.json"
SETTINGS_FILE = DATA_DIR / "settings.json"


def load_json(filepath: Path, default: dict = None) -> dict:
    """Load JSON file or return default."""
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)
    return default or {}


def save_json(filepath: Path, data: dict):
    """Save data to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def get_settings() -> dict:
    """Get app settings."""
    return load_json(SETTINGS_FILE, {"last_profile": None})


def save_settings(settings_data: dict):
    """Save app settings."""
    save_json(SETTINGS_FILE, settings_data)


def load_profiles() -> dict:
    """Load all flute profiles."""
    return load_json(PROFILES_FILE, {})


def save_profiles(profiles: dict):
    """Save all flute profiles."""
    save_json(PROFILES_FILE, profiles)


def load_songs() -> dict:
    """Load all saved songs."""
    return load_json(SONGS_FILE, {})


def save_songs(songs: dict):
    """Save all songs."""
    save_json(SONGS_FILE, songs)


# ===== Pydantic Models =====

class ProfileCreate(BaseModel):
    name: str
    a4_frequency: int = 440


class ProfileUpdate(BaseModel):
    a4_frequency: Optional[int] = None
    volume_threshold: Optional[float] = None
    stability_threshold: Optional[int] = None


class FingeringInput(BaseModel):
    note: str
    fingering: list[int]
    frequency: float
    profile_id: str


class SaveSongInput(BaseModel):
    title: str
    profile_id: str
    notes: list[dict]
    key_signature: Optional[str] = None


# ===== Main Pages =====

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    profiles = load_profiles()
    app_settings = get_settings()
    songs = load_songs()

    # Get last used profile
    last_profile_id = app_settings.get("last_profile")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "profiles": profiles,
        "last_profile_id": last_profile_id,
        "songs": songs
    })


@app.post("/upload", response_class=HTMLResponse)
async def upload_sheet_music(
    request: Request,
    file: UploadFile = File(...),
    profile_id: str = Form(...),
    import_mode: str = Form("standard")
):
    """Process uploaded sheet music and return fingerings."""
    profiles = load_profiles()

    if not file.filename:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "error": "No file provided",
            "profiles": profiles
        })

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return templates.TemplateResponse("results.html", {
            "request": request,
            "error": "Please upload a PNG or JPG image",
            "profiles": profiles
        })

    if profile_id not in profiles:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "error": "Please select or create a flute profile first",
            "profiles": profiles
        })

    # Update last used profile
    app_settings = get_settings()
    app_settings["last_profile"] = profile_id
    save_settings(app_settings)

    # Save to temp file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Process with OMR
        extracted_music = await omr_service.process_image(tmp_path)

        # Get profile fingerings
        profile = profiles[profile_id]
        profile_fingerings = profile.get("fingerings", {})

        # AI-Assisted Import Mode
        if import_mode == "ai":
            # Get AI suggestions for optimal arrangement
            # Pass the original image so AI can verify OCR results visually
            ai_suggestion = ai_suggestion_service.get_suggestions(
                extracted_music, profile_fingerings, image_path=tmp_path
            )

            # Build results with AI suggestions
            ai_results = []
            for mapping in ai_suggestion.note_mappings:
                # Get fingering for the suggested note
                fingering_data = profile_fingerings.get(mapping.suggested)
                ai_results.append({
                    "original": mapping.original,
                    "transposed": mapping.transposed,
                    "suggested": mapping.suggested,
                    "playable": mapping.playable,
                    "substitution_reason": mapping.substitution_reason,
                    "fingering": fingering_data.get("fingering") if fingering_data else None
                })

            playable_count = sum(1 for r in ai_results if r["playable"])

            return templates.TemplateResponse("ai_results.html", {
                "request": request,
                "title": extracted_music.title or file.filename,
                "original_key": ai_suggestion.original_key,
                "suggested_key": ai_suggestion.suggested_key,
                "transposition": ai_suggestion.recommended_transposition,
                "transposition_reasoning": ai_suggestion.transposition_reasoning,
                "musical_notes": ai_suggestion.musical_notes,
                "ocr_corrections": ai_suggestion.ocr_corrections,
                "reasoning_summary": ai_suggestion.reasoning_summary,
                "results": ai_results,
                "confidence": extracted_music.confidence,
                "note_count": len(ai_results),
                "playable_count": playable_count,
                "profile_id": profile_id,
                "profile_name": profile.get("name", "Unknown"),
                "profiles": profiles
            })

        # Standard Import Mode
        results = []
        for note in extracted_music.notes:
            note_key = f"{note.name.value}"
            if note.accidental.value == "sharp":
                note_key += "#"
            elif note.accidental.value == "flat":
                note_key += "b"
            note_key += str(note.octave)

            fingering_data = profile_fingerings.get(note_key)

            results.append({
                "note_key": note_key,
                "fingering": fingering_data.get("fingering") if fingering_data else None,
                "playable": fingering_data is not None
            })

        playable_count = sum(1 for r in results if r["playable"])

        return templates.TemplateResponse("results.html", {
            "request": request,
            "title": extracted_music.title or file.filename,
            "key_signature": extracted_music.key_signature,
            "results": results,
            "confidence": extracted_music.confidence,
            "note_count": len(extracted_music.notes),
            "playable_count": playable_count,
            "profile_id": profile_id,
            "profile_name": profile.get("name", "Unknown"),
            "profiles": profiles
        })
    except Exception as e:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "error": f"Error processing image: {str(e)}",
            "profiles": profiles
        })
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/discover", response_class=HTMLResponse)
async def discover_page(request: Request):
    """Render the fingering discovery page."""
    profiles = load_profiles()
    app_settings = get_settings()
    return templates.TemplateResponse("discover.html", {
        "request": request,
        "profiles": profiles,
        "last_profile_id": app_settings.get("last_profile")
    })


@app.get("/library", response_class=HTMLResponse)
async def library_page(request: Request):
    """Render the song library page."""
    profiles = load_profiles()
    songs = load_songs()
    return templates.TemplateResponse("library.html", {
        "request": request,
        "profiles": profiles,
        "songs": songs
    })


@app.get("/song/{song_id}", response_class=HTMLResponse)
async def view_song(request: Request, song_id: str):
    """View a saved song."""
    songs = load_songs()
    profiles = load_profiles()

    if song_id not in songs:
        return RedirectResponse(url="/library")

    song = songs[song_id]
    profile = profiles.get(song.get("profile_id"), {})

    return templates.TemplateResponse("song.html", {
        "request": request,
        "song": song,
        "song_id": song_id,
        "profile": profile,
        "profiles": profiles
    })


# ===== Profile API =====

@app.get("/api/profiles")
async def list_profiles():
    """Get all profiles."""
    return load_profiles()


@app.post("/api/profiles")
async def create_profile(data: ProfileCreate):
    """Create a new flute profile (empty, no pre-loaded fingerings)."""
    profiles = load_profiles()

    profile_id = str(uuid.uuid4())[:8]
    profiles[profile_id] = {
        "name": data.name,
        "a4_frequency": data.a4_frequency,
        "volume_threshold": 0.05,
        "stability_threshold": 300,
        "fingerings": {},  # Empty! User builds this up
        "created_at": datetime.now().isoformat()
    }
    save_profiles(profiles)

    # Set as last used
    app_settings = get_settings()
    app_settings["last_profile"] = profile_id
    save_settings(app_settings)

    return {"success": True, "profile_id": profile_id}


@app.get("/api/profiles/{profile_id}")
async def get_profile(profile_id: str):
    """Get a specific profile."""
    profiles = load_profiles()
    if profile_id not in profiles:
        return {"error": "Profile not found"}
    return profiles[profile_id]


@app.put("/api/profiles/{profile_id}")
async def update_profile(profile_id: str, data: ProfileUpdate):
    """Update profile settings."""
    profiles = load_profiles()
    if profile_id not in profiles:
        return {"success": False, "error": "Profile not found"}

    if data.a4_frequency is not None:
        profiles[profile_id]["a4_frequency"] = data.a4_frequency
    if data.volume_threshold is not None:
        profiles[profile_id]["volume_threshold"] = data.volume_threshold
    if data.stability_threshold is not None:
        profiles[profile_id]["stability_threshold"] = data.stability_threshold

    save_profiles(profiles)
    return {"success": True}


@app.delete("/api/profiles/{profile_id}")
async def delete_profile(profile_id: str):
    """Delete a profile."""
    profiles = load_profiles()
    if profile_id in profiles:
        del profiles[profile_id]
        save_profiles(profiles)
    return {"success": True}


# ===== Fingerings API =====

@app.get("/api/fingerings/{profile_id}")
async def get_profile_fingerings(profile_id: str):
    """Get fingerings for a specific profile."""
    profiles = load_profiles()
    if profile_id not in profiles:
        return {}
    return profiles[profile_id].get("fingerings", {})


@app.post("/api/fingerings")
async def save_fingering(data: FingeringInput):
    """Save a fingering to a profile."""
    profiles = load_profiles()

    if data.profile_id not in profiles:
        return {"success": False, "error": "Profile not found"}

    if "fingerings" not in profiles[data.profile_id]:
        profiles[data.profile_id]["fingerings"] = {}

    profiles[data.profile_id]["fingerings"][data.note] = {
        "fingering": data.fingering,
        "frequency": data.frequency
    }

    save_profiles(profiles)
    return {"success": True, "note": data.note}


@app.delete("/api/fingerings/{profile_id}/{note}")
async def delete_fingering(profile_id: str, note: str):
    """Delete a fingering from a profile."""
    profiles = load_profiles()

    if profile_id in profiles and "fingerings" in profiles[profile_id]:
        if note in profiles[profile_id]["fingerings"]:
            del profiles[profile_id]["fingerings"][note]
            save_profiles(profiles)

    return {"success": True}


# ===== Songs API =====

@app.post("/api/songs")
async def save_song(data: SaveSongInput):
    """Save a converted song to the library."""
    songs = load_songs()

    song_id = str(uuid.uuid4())[:8]
    songs[song_id] = {
        "title": data.title,
        "profile_id": data.profile_id,
        "notes": data.notes,
        "key_signature": data.key_signature,
        "created_at": datetime.now().isoformat()
    }

    save_songs(songs)
    return {"success": True, "song_id": song_id}


@app.put("/api/songs/{song_id}/profile")
async def update_song_profile(song_id: str, profile_id: str = Form(...)):
    """Update the profile used for a song."""
    songs = load_songs()
    profiles = load_profiles()

    if song_id not in songs:
        return {"success": False, "error": "Song not found"}
    if profile_id not in profiles:
        return {"success": False, "error": "Profile not found"}

    songs[song_id]["profile_id"] = profile_id
    save_songs(songs)

    return {"success": True}


@app.delete("/api/songs/{song_id}")
async def delete_song(song_id: str):
    """Delete a song from the library."""
    songs = load_songs()
    if song_id in songs:
        del songs[song_id]
        save_songs(songs)
    return {"success": True}


# ===== Settings API =====

@app.post("/api/settings/last-profile")
async def set_last_profile(profile_id: str = Form(...)):
    """Set the last used profile."""
    app_settings = get_settings()
    app_settings["last_profile"] = profile_id
    save_settings(app_settings)
    return {"success": True}


# ===== Transposition API =====

@app.get("/api/songs/{song_id}/analyze/{profile_id}")
async def analyze_song_transposition(song_id: str, profile_id: str):
    """
    Analyze transposition options for a song with a given flute profile.
    Returns playability stats for each transposition option.
    """
    songs = load_songs()
    profiles = load_profiles()

    if song_id not in songs:
        return {"error": "Song not found"}
    if profile_id not in profiles:
        return {"error": "Profile not found"}

    song = songs[song_id]
    fingerings = profiles[profile_id].get("fingerings", {})

    # Get transposition options
    options = find_optimal_transposition(song["notes"], fingerings)

    # Get current (original) playability
    current_stats = analyze_playability(song["notes"], fingerings)

    return {
        "song_id": song_id,
        "profile_id": profile_id,
        "original_key": song.get("key_signature", "Unknown"),
        "current_stats": current_stats,
        "transposition_options": options,
        "best_option": options[0] if options else None
    }


@app.get("/api/songs/{song_id}/transposed/{profile_id}")
async def get_transposed_song(song_id: str, profile_id: str, semitones: int = 0):
    """
    Get song notes transposed by given semitones, with playability info.
    """
    songs = load_songs()
    profiles = load_profiles()

    if song_id not in songs:
        return {"error": "Song not found"}
    if profile_id not in profiles:
        return {"error": "Profile not found"}

    song = songs[song_id]
    fingerings = profiles[profile_id].get("fingerings", {})

    # Transpose notes
    transposed_notes = transpose_notes(song["notes"], semitones)

    # Add playability and suggestions to each note
    result_notes = []
    for note in transposed_notes:
        note_key = note["note_key"]
        original_key = note.get("original_note_key", note_key)

        note_result = {
            "note_key": note_key,
            "original_note_key": original_key if semitones != 0 else None,
            "playable": note_key in fingerings,
            "fingering": None,
            "suggestion": None
        }

        if note_key in fingerings:
            note_result["fingering"] = fingerings[note_key]["fingering"]
        else:
            # Find nearest playable note
            suggestion = find_nearest_playable(note_key, fingerings)
            if suggestion:
                note_result["suggestion"] = suggestion

        result_notes.append(note_result)

    # Calculate stats
    stats = analyze_playability(transposed_notes, fingerings)

    # Get new key name
    new_key = get_key_name(song.get("key_signature", ""), semitones)

    return {
        "song_id": song_id,
        "title": song["title"],
        "original_key": song.get("key_signature"),
        "transposed_key": new_key,
        "semitones": semitones,
        "notes": result_notes,
        "stats": stats
    }
