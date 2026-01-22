# 🎧 Audio or It Didn’t Happen  
`# pdf-to-speech`

Complete pipeline for converting PDFs to multi-voice audiobooks using LLM-powered dialogue detection and ElevenLabs TTS. 🎧📖

## Features

- **PDF extraction** → Markdown using `glossapi` (`Corpus.extract`)
- **Text cleanup** → Removes control chars, page numbers, running headers, fixes hyphenated line breaks
- **Plain text output** → Markdown → `.txt` via `pandoc` (better for TTS)
- **LLM-powered XML conversion** → Dialogue detection, speaker identification, and audio tag insertion using OpenAI
- **ElevenLabs XML/SSML output** → `<voice name="...">...</voice>` segments ready for multi-voice TTS
- **Optional MP3 generation** → Converts XML/TXT to audiobook using ElevenLabs API with parallel processing
- **Content validation** → Warns when XML appears to drop content beyond tolerance

## Installation

### Windows (recommended: use the setup script)

```powershell
.\setup.ps1
```

### Manual (any OS)

```bash
python -m venv .venv
```

```bash
# Windows:
.\.venv\Scripts\activate
```

```bash
# macOS/Linux:
source .venv/bin/activate
```

```bash
pip install -U pip
pip install -e .
```

```bash
# Install Pandoc (required for Markdown -> TXT)
# Windows (via winget):
winget install JohnMacFarlane.Pandoc

# macOS (via Homebrew):
brew install pandoc

# Linux (via apt):
sudo apt-get install pandoc
```

```bash
# Install ffmpeg (required for pydub/audio processing)
# Windows (via winget):
winget install ffmpeg

# macOS (via Homebrew):
brew install ffmpeg

# Linux (via apt):
sudo apt-get install ffmpeg
```

## Usage

### Basic: PDF to XML

```bash
python main.py
```

This processes all PDFs in `pdfs/` directory and outputs:
- Cleaned text files in `output/tts_txt/`
- XML files with speaker tags in `output/elevenlabs_xml/`

### With TTS: PDF to MP3 Audiobook

```bash
python main.py --tts
```

This runs the full pipeline including MP3 generation from XML files.

### Options

```bash
python main.py --help
```

```bash
# Custom input/output directories
python main.py --input-dir /path/to/pdfs --output-dir /path/to/output

# Skip XML conversion (only generate cleaned text)
python main.py --no-xml

# Generate TTS from TXT files instead of XML (narrator only)
python main.py --tts --tts-format txt
```
## Example Output

🎧 **Sample Audiobook**  
▶️ Listen here:  
https://github.com/Herc-Ch/Audio-Or-It-Didnt-Happen/releases/download/v0.1/v0.1 – Sample Audiobook.mp3




## Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key-here
ELEVENLABS_API_KEY=your-elevenlabs-api-key-here
```

**Required:**
- `OPENAI_API_KEY` - For LLM-based dialogue detection and audio tagging
- `ELEVENLABS_API_KEY` - For TTS generation (only needed if using `--tts`)

## Architecture

```mermaid
flowchart LR
  A["PDF files (pdfs/)"] -- "extract" --> B["GlossAPI (Corpus.extract)"]
  B -- "writes" --> C["Markdown (output/markdown/)"]
  C -- "clean" --> D["Cleaner (src/clean_text.py)"]
  D -- "convert" --> E["Pandoc (Markdown -> TXT)"]
  E -- "writes" --> F["Text (output/tts_txt/)"]
  F -- "optional LLM" --> G["XML converter (src/xml_converter.py)"]
  G -- "writes" --> H["ElevenLabs XML (output/elevenlabs_xml/)"]
  H -- "optional TTS" --> I["TTS converter (src/tts_converter.py)"]
  I -- "writes" --> J["MP3 Audiobook (output/audiobooks/)"]
```

## Project Structure

```plaintext
my_pdf_to_speech/
├── pdfs/                    # Input PDF files
├── output/
│   ├── markdown/           # Extracted markdown from PDFs
│   ├── tts_txt/            # Cleaned plain text files
│   ├── elevenlabs_xml/     # XML files with speaker tags
│   └── audiobooks/         # Generated MP3 files (if --tts used)
├── src/
│   ├── clean_text.py       # Text cleaning utilities
│   ├── processor.py        # PDF processing pipeline
│   ├── xml_converter.py    # LLM-based XML conversion
│   └── tts_converter.py    # ElevenLabs TTS conversion
├── main.py                 # Main entry point
├── pyproject.toml          # Python dependencies
├── setup.ps1               # Setup script (Windows)
└── README.md               # This file
```

## Voice Configuration

Character voices are configured in `src/tts_converter.py` in the `VOICE_DICT` mapping. Update voice IDs to match your ElevenLabs voice library.

## Notes

- XML conversion uses OpenAI LLM (default: `gpt-5.2`) for dialogue detection
- TTS generation processes segments in parallel (max 5 concurrent requests)
- Short narrator segments are automatically merged with character segments to avoid jarring voice switches
- Content validation warns if >5% content is missing, errors if >20% missing
