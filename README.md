# Flute Helper

A web app that converts sheet music images into Native American flute fingerings for a 6-hole flute in the key of E.

## Features

- Upload sheet music images (PNG, JPG)
- Automatic note extraction using OpenAI Vision (GPT-5.2)
- Converts notes to 6-hole Native American flute fingerings
- Visual SVG fingering diagrams
- Supports both traditional sheet music and lead sheets

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/jeffsharris/flutehelper.git
   cd flutehelper
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

4. **Run the app**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Open** http://localhost:8000 in your browser

## Usage

1. Upload an image of sheet music
2. The app extracts melody notes using AI vision
3. Each note is mapped to a fingering diagram showing which holes to cover

## Fingering Chart (6-hole E Flute)

| Note | Fingering | Diagram |
|------|-----------|---------|
| E4   | All closed | ●●●●●● |
| F#4  | Bottom open | ●●●●●○ |
| G4   | Two open | ●●●●○○ |
| A4   | Three open | ●●●○○○ |
| B4   | Four open | ●●○○○○ |
| D5   | Special | ○●○○○○ |
| E5   | Overblow | ●●●●●● |

● = covered (closed), ○ = open

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML, CSS, HTMX
- **OMR**: OpenAI Vision API (GPT-5.2)

## License

MIT
