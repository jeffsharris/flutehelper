import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .services.omr_service import OMRService
from .services.fingering_mapper import FingeringMapper

app = FastAPI(
    title="Flute Helper",
    description="Convert sheet music to Native American flute fingerings"
)

# Get the app directory
APP_DIR = Path(__file__).parent

# Mount static files and templates
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

# Initialize services
omr_service = OMRService(api_key=settings.OPENAI_API_KEY)
fingering_mapper = FingeringMapper(flute_key="E")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with upload form."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_sheet_music(request: Request, file: UploadFile = File(...)):
    """Process uploaded sheet music and return fingerings."""

    # Validate file type
    if not file.filename:
        return templates.TemplateResponse(
            "results.html",
            {"request": request, "error": "No file provided"}
        )

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return templates.TemplateResponse(
            "results.html",
            {"request": request, "error": "Please upload a PNG or JPG image"}
        )

    # Save to temp file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Process with OMR
        extracted_music = await omr_service.process_image(tmp_path)

        # Map to fingerings
        fingering_results = fingering_mapper.map_notes(extracted_music)

        playable_count = sum(1 for r in fingering_results if r.playable)

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "title": extracted_music.title or "Untitled",
                "key_signature": extracted_music.key_signature,
                "results": fingering_results,
                "confidence": extracted_music.confidence,
                "note_count": len(extracted_music.notes),
                "playable_count": playable_count
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "results.html",
            {"request": request, "error": f"Error processing image: {str(e)}"}
        )
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/fingering/{note}")
async def get_fingering_api(note: str):
    """API endpoint to get fingering for a specific note."""
    from .data.fingering_charts import get_fingering, is_in_range

    # Parse note (e.g., "E4", "G#4")
    if len(note) < 2:
        return {"note": note, "error": "Invalid note format"}

    try:
        # Handle accidentals
        if note[-2] in "#b":
            note_name = note[:-1]
            octave = int(note[-1])
        else:
            note_name = note[:-1]
            octave = int(note[-1])

        if not is_in_range(note_name, octave):
            return {"note": note, "error": "Note out of range for E flute"}

        fingering = get_fingering(note_name, octave)
        if fingering:
            return {
                "note": note,
                "fingering": fingering,
                "symbols": "".join(["●" if h else "○" for h in fingering])
            }
    except (ValueError, IndexError):
        pass

    return {"note": note, "error": "Fingering not found"}
