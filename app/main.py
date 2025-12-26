"""
Flute Helper - FastAPI Web Application.

This is the main entry point for the Flute Helper web application.
It provides a web interface for:

1. Converting sheet music images to Native American flute fingerings
2. Managing flute profiles with custom fingering mappings
3. Discovering and recording fingerings using audio input
4. Building a library of arranged songs

Architecture:
    The app uses HTMX for interactive updates without full page reloads.
    AI features use the OpenAI Responses API with vision and reasoning.

Routes:
    GET  /              - Home page with upload form
    POST /upload        - Process sheet music (standard or AI mode)
    POST /upload-stream - Process with streaming AI reasoning (SSE)
    GET  /discover      - Fingering discovery page
    GET  /library       - Saved songs library
    GET  /song/{id}     - View a saved song

    API Routes:
    /api/profiles/*     - CRUD operations for flute profiles
    /api/fingerings/*   - Manage fingerings within profiles
    /api/songs/*        - CRUD operations for saved songs
    /api/settings/*     - Application settings
"""

import json
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Request, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .config import settings
from .services.omr_service import OMRService
from .services.ai_suggestion import ai_suggestion_service, StreamEvent
from .services.music_utils import (
    transpose_notes, analyze_playability, find_optimal_transposition,
    find_nearest_playable, get_key_name
)
from .utils.storage import (
    load_profiles, save_profiles,
    load_songs, save_songs,
    get_settings, save_settings,
)


# ============================================================
# Application Setup
# ============================================================

app = FastAPI(
    title="Flute Helper",
    description="Convert sheet music to Native American flute fingerings",
    version="1.0.0"
)

# Directory paths
APP_DIR = Path(__file__).parent

# Static files and templates
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

# Initialize services
omr_service = OMRService(api_key=settings.OPENAI_API_KEY)


# ============================================================
# Request/Response Models
# ============================================================

class ProfileCreate(BaseModel):
    """Request model for creating a new flute profile."""
    name: str
    a4_frequency: int = 440


class ProfileUpdate(BaseModel):
    """Request model for updating profile settings."""
    a4_frequency: Optional[int] = None
    volume_threshold: Optional[float] = None
    stability_threshold: Optional[int] = None


class FingeringInput(BaseModel):
    """Request model for saving a fingering to a profile."""
    note: str
    fingering: list[int]
    frequency: float
    profile_id: str


class SaveSongInput(BaseModel):
    """Request model for saving a song to the library."""
    title: str
    profile_id: str
    notes: list[dict]
    key_signature: Optional[str] = None


# ============================================================
# Page Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the home page.

    Displays the main interface with:
    - Sheet music upload form
    - Profile selector
    - Recently saved songs
    """
    profiles = load_profiles()
    app_settings = get_settings()
    songs = load_songs()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "profiles": profiles,
        "last_profile_id": app_settings.get("last_profile"),
        "songs": songs
    })


@app.post("/upload", response_class=HTMLResponse)
async def upload_sheet_music(
    request: Request,
    file: UploadFile = File(...),
    profile_id: str = Form(...),
    import_mode: str = Form("standard")
):
    """
    Process uploaded sheet music and return fingerings.

    Supports two modes:
    - standard: Direct OCR → fingering lookup
    - ai: OCR → AI arrangement suggestions with transposition

    Args:
        file: Uploaded image file (PNG/JPG)
        profile_id: ID of the flute profile to use
        import_mode: "standard" or "ai"

    Returns:
        Rendered results.html or ai_results.html template
    """
    profiles = load_profiles()

    # Validation
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

    # Save to temp file for processing
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Extract notes using OCR
        extracted_music = await omr_service.process_image(tmp_path)

        profile = profiles[profile_id]
        profile_fingerings = profile.get("fingerings", {})

        # AI-Assisted Import Mode
        if import_mode == "ai":
            return await _handle_ai_import(
                request, extracted_music, profile_fingerings,
                tmp_path, file.filename, profile, profile_id, profiles
            )

        # Standard Import Mode
        return _handle_standard_import(
            request, extracted_music, profile_fingerings,
            file.filename, profile, profile_id, profiles
        )

    except Exception as e:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "error": f"Error processing image: {str(e)}",
            "profiles": profiles
        })
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def _handle_ai_import(request, extracted_music, profile_fingerings,
                            tmp_path, filename, profile, profile_id, profiles):
    """Handle AI-assisted import mode processing."""
    ai_suggestion = ai_suggestion_service.get_suggestions(
        extracted_music, profile_fingerings, image_path=tmp_path
    )

    # Build results with AI suggestions
    ai_results = []
    for mapping in ai_suggestion.note_mappings:
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
        "title": extracted_music.title or filename,
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
        "profiles": profiles,
        # Debug info
        "debug_raw_response": ai_suggestion.debug_raw_response,
        "debug_parse_error": ai_suggestion.debug_parse_error,
        "debug_model_used": ai_suggestion.debug_model_used,
        "debug_input_notes": ai_suggestion.debug_input_notes,
        "debug_available_notes": ai_suggestion.debug_available_notes,
        "debug_request": ai_suggestion.debug_request
    })


def _handle_standard_import(request, extracted_music, profile_fingerings,
                            filename, profile, profile_id, profiles):
    """Handle standard import mode processing."""
    results = []
    for note in extracted_music.notes:
        # Build note key string (e.g., "C#4")
        note_key = note.name.value
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
        "title": extracted_music.title or filename,
        "key_signature": extracted_music.key_signature,
        "results": results,
        "confidence": extracted_music.confidence,
        "note_count": len(extracted_music.notes),
        "playable_count": playable_count,
        "profile_id": profile_id,
        "profile_name": profile.get("name", "Unknown"),
        "profiles": profiles
    })


@app.post("/upload-stream")
async def upload_sheet_music_streaming(
    file: UploadFile = File(...),
    profile_id: str = Form(...)
):
    """
    Process sheet music with streaming AI reasoning.

    Returns Server-Sent Events (SSE) with real-time updates:
    - status: Processing stage updates
    - reasoning: AI reasoning text as it's generated
    - complete: Final results with all mappings
    - error: Error message if something fails

    This endpoint enables real-time display of the AI's thinking process.
    """
    profiles = load_profiles()

    # Validation with SSE error responses
    if not file.filename:
        return _sse_error("No file provided")

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return _sse_error("Please upload a PNG or JPG image")

    if profile_id not in profiles:
        return _sse_error("Please select or create a flute profile first")

    # Update last used profile
    app_settings = get_settings()
    app_settings["last_profile"] = profile_id
    save_settings(app_settings)

    # Save to temp file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    async def generate():
        """Generate SSE events as AI processes the request."""
        try:
            yield _sse_event("status", "Extracting notes from image...")
            extracted_music = await omr_service.process_image(tmp_path)

            profile = profiles[profile_id]
            profile_fingerings = profile.get("fingerings", {})

            yield _sse_event("status", "AI is analyzing the arrangement...")

            # Stream AI suggestions
            for event in ai_suggestion_service.get_suggestions_streaming(
                extracted_music, profile_fingerings, image_path=tmp_path
            ):
                if event.type == "reasoning_delta":
                    yield _sse_event("reasoning", event.data)
                elif event.type == "complete":
                    final_data = _build_streaming_result(
                        event.final_response, extracted_music,
                        profile_fingerings, profile, profile_id, file.filename
                    )
                    yield f"data: {json.dumps({'type': 'complete', 'data': final_data})}\n\n"
                elif event.type == "error":
                    yield _sse_event("error", event.data)

        except Exception as e:
            yield _sse_event("error", str(e))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


def _sse_event(event_type: str, data: str) -> str:
    """Format a Server-Sent Event."""
    return f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"


def _sse_error(message: str):
    """Return an SSE error response."""
    async def error_gen():
        yield _sse_event("error", message)
    return StreamingResponse(error_gen(), media_type="text/event-stream")


def _build_streaming_result(ai_suggestion, extracted_music, profile_fingerings,
                            profile, profile_id, filename):
    """Build the final result object for streaming response."""
    ai_results = []
    for mapping in ai_suggestion.note_mappings:
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

    return {
        "title": extracted_music.title or filename,
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
        # Debug info
        "debug_raw_response": ai_suggestion.debug_raw_response,
        "debug_parse_error": ai_suggestion.debug_parse_error,
        "debug_model_used": ai_suggestion.debug_model_used,
        "debug_input_notes": ai_suggestion.debug_input_notes,
        "debug_available_notes": ai_suggestion.debug_available_notes,
        "debug_request": ai_suggestion.debug_request
    }


@app.get("/discover", response_class=HTMLResponse)
async def discover_page(request: Request):
    """Render the fingering discovery page for recording new fingerings."""
    profiles = load_profiles()
    app_settings = get_settings()
    return templates.TemplateResponse("discover.html", {
        "request": request,
        "profiles": profiles,
        "last_profile_id": app_settings.get("last_profile")
    })


@app.get("/library", response_class=HTMLResponse)
async def library_page(request: Request):
    """Render the song library page showing all saved songs."""
    profiles = load_profiles()
    songs = load_songs()
    return templates.TemplateResponse("library.html", {
        "request": request,
        "profiles": profiles,
        "songs": songs
    })


@app.get("/song/{song_id}", response_class=HTMLResponse)
async def view_song(request: Request, song_id: str):
    """View a saved song with its fingerings."""
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


# ============================================================
# Profile API
# ============================================================

@app.get("/api/profiles")
async def list_profiles():
    """Get all flute profiles."""
    return load_profiles()


@app.post("/api/profiles")
async def create_profile(data: ProfileCreate):
    """
    Create a new flute profile.

    Profiles start empty - users build up fingerings through discovery.
    """
    profiles = load_profiles()

    profile_id = str(uuid.uuid4())[:8]
    profiles[profile_id] = {
        "name": data.name,
        "a4_frequency": data.a4_frequency,
        "volume_threshold": 0.05,
        "stability_threshold": 300,
        "fingerings": {},
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
    """Get a specific profile by ID."""
    profiles = load_profiles()
    if profile_id not in profiles:
        return {"error": "Profile not found"}
    return profiles[profile_id]


@app.put("/api/profiles/{profile_id}")
async def update_profile(profile_id: str, data: ProfileUpdate):
    """Update profile settings (tuning, thresholds)."""
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
    """Delete a flute profile."""
    profiles = load_profiles()
    if profile_id in profiles:
        del profiles[profile_id]
        save_profiles(profiles)
    return {"success": True}


# ============================================================
# Fingerings API
# ============================================================

@app.get("/api/fingerings/{profile_id}")
async def get_profile_fingerings(profile_id: str):
    """Get all fingerings for a profile."""
    profiles = load_profiles()
    if profile_id not in profiles:
        return {}
    return profiles[profile_id].get("fingerings", {})


@app.post("/api/fingerings")
async def save_fingering(data: FingeringInput):
    """Save a discovered fingering to a profile."""
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


# ============================================================
# Songs API
# ============================================================

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
    """Update which profile is used for a song."""
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


# ============================================================
# Settings API
# ============================================================

@app.post("/api/settings/last-profile")
async def set_last_profile(profile_id: str = Form(...)):
    """Set the last used profile (for auto-selection)."""
    app_settings = get_settings()
    app_settings["last_profile"] = profile_id
    save_settings(app_settings)
    return {"success": True}


# ============================================================
# Transposition API
# ============================================================

@app.get("/api/songs/{song_id}/analyze/{profile_id}")
async def analyze_song_transposition(song_id: str, profile_id: str):
    """
    Analyze transposition options for a song.

    Returns playability statistics for each possible transposition,
    helping users find the best key for their flute.
    """
    songs = load_songs()
    profiles = load_profiles()

    if song_id not in songs:
        return {"error": "Song not found"}
    if profile_id not in profiles:
        return {"error": "Profile not found"}

    song = songs[song_id]
    fingerings = profiles[profile_id].get("fingerings", {})

    options = find_optimal_transposition(song["notes"], fingerings)
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
    Get song notes transposed by given semitones.

    Includes playability info and suggestions for unplayable notes.
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

    # Build result with playability info
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
            suggestion = find_nearest_playable(note_key, fingerings)
            if suggestion:
                note_result["suggestion"] = suggestion

        result_notes.append(note_result)

    stats = analyze_playability(transposed_notes, fingerings)
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
