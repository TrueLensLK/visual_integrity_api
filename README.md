# DeepFake Detection System

A multi-layered AI-powered deepfake detection system using various analysis techniques including metadata analysis, digital physics, facial consistency, visual AI, frequency spectrum analysis, and optical physics (eye reflection analysis).

## Features

- **Layer 0: C2PA** - Verifies cryptographic content credentials
- **Layer 1: Triage** - Quick file validation
- **Layer 2: Metadata** - Detects AI generation tools and missing EXIF data
- **Layer 3: Physics** - ELA (Error Level Analysis) and noise consistency
- **Layer 3.5: Face** - Analyzes facial landmark and background consistency
- **Layer 4: Visual** - 5-Model Neural Ensemble (SDXL-Detector + ViT + SigLIP2 + ConvNeXt + Swin)
- **Layer 6: Spectrum** - Frequency domain analysis (FFT)
- **Layer 7: Eyes** - Optical physics and corneal reflection consistency
- **Layer 8: Watermark** - Detects invisible watermarks and text
- **Layer 8.5: PRNU** - Sensor fingerprint analysis
- **Layer 9: Context** - Reverse image search for provenance
- **Layer 10: Shadow** - Light source consistency analysis
- **Layer 12: Artifacts** - Spatial domain analysis for checkerboard/GAN traces
- **Layer 5: The Judge** - Master verdict system with "Redemption Logic"
- **Final Boss: LLM** - Single Gemini/OpenRouter call for gray zone cases
- **Final Boss: Adversarial Debate** - Prosecution vs Defense vs Convergence Judge for contradictions

## Installation

1. Clone the repository and move into it:
```bash
git clone <your-repo-url>
cd DeepFake_Detection
```

2. Create a virtual environment:

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies from the root requirements file:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```

On Windows, if `cp` is not available:
```powershell
Copy-Item .env.example .env
```

Edit `.env` and set at least one LLM provider key:
- `GOOGLE_AI_API_KEY` (or `GEMINI_API_KEY`)
- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`

Keep `ENABLE_LLM_JUDGE=true` to allow LLM-based judging.

## Usage

Run the FastAPI server from the repository root:
```bash
python -m uvicorn main:app --reload
```

Then open:
- API docs: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`

Example API call (PowerShell):
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/analyze -Method Post -Form @{
    file = Get-Item "path\to\image.jpg"
}
```

Example API call (curl):
```bash
curl -X POST "http://127.0.0.1:8000/analyze" -F "file=@path/to/image.jpg"
```

## Docker

You can run this API in Docker without installing Python locally.

1. Create `.env` from template and set your API keys:
```bash
cp .env.example .env
```

2. Build image:
```bash
docker build -t deepfake-detection:latest .
```

3. Run container:
```bash
docker run --rm -p 8000:8000 --env-file .env --name deepfake-api deepfake-detection:latest
```

4. Open:
- API docs: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`

Docker Compose alternative:
```bash
docker compose up --build
```

Stop Compose:
```bash
docker compose down
```

Notes:
- First run can be slow because Python ML/CV dependencies are heavy.
- LLM features require valid keys in `.env`.
- If port `8000` is busy, map another host port (for example `-p 8001:8000`).

Optional: configure AI metadata keywords (comma-separated) used by Layer 2:

Windows (PowerShell):
```powershell
$env:AI_METADATA_KEYWORDS="midjourney,stable diffusion,openai,firefly"
```

Windows (cmd):
```bash
set AI_METADATA_KEYWORDS=midjourney,stable diffusion,openai,firefly
```

macOS/Linux:
```bash
export AI_METADATA_KEYWORDS="midjourney,stable diffusion,openai,firefly"
```

## LLM Notes

- LLM is not called for every image.
- The system uses LLM mainly for ambiguous or contradictory cases.
- If no valid key is available, the app falls back to rule-based judging.

## API Endpoint

**POST** `/analyze`
- Upload an image file
- Returns detection results with confidence score and detailed analysis

## Project Structure

```
DeepFake_Detection/
├── main.py                          # FastAPI application entry point
├── requirements.txt                 # Python dependencies (install from this file)
├── .env.example                     # Environment variable template
├── temp_uploads/                    # Temporary upload directory
└── Universal_Detector/
    └── src/
        └── layers/
            ├── layer_2_metadata.py      # Metadata analysis
            ├── layer_3_physics.py       # Digital physics
            ├── layer_3_5_face.py        # Face consistency
            ├── layer_4_visual.py        # Visual AI model
            ├── layer_5_judge.py         # Final verdict aggregator
            ├── layer_6_spectrum.py      # Frequency analysis
            ├── layer_7_eyes.py          # Eye reflection analysis
            ├── debate/                   # Adversarial debate package
            │   ├── __init__.py          # Re-exports DebateOrchestrator
            │   ├── models.py            # Shared data classes & prompts
            │   ├── prosecution.py       # Prosecution agent (Gemini Vision)
            │   ├── defense.py           # Defense agent (OpenRouter Vision)
            │   ├── convergence.py       # Convergence judge (Groq text)
            │   └── orchestrator.py      # Debate flow controller
            └── (models auto-downloaded from HuggingFace)
```

## Requirements

- Python 3.8+
- FastAPI
- PyTorch
- OpenCV
- MediaPipe
- Pillow
- NumPy

## Troubleshooting

1. `LLM not working`
- Ensure `.env` exists in the project root.
- Ensure at least one API key is set and valid.
- Ensure `ENABLE_LLM_JUDGE=true`.
- Check terminal logs for provider errors like missing key, quota, or auth failures.

2. `Module/import errors`
- Confirm you installed from `requirements.txt` in the repository root.
- Confirm your virtual environment is activated before running `uvicorn`.

3. `Port already in use`
```bash
python -m uvicorn main:app --reload --port 8001
```

## License

MIT
