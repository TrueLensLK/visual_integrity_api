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

1. Clone the repository
2. Install dependencies:
```bash
pip install -r Universal_Detector/requirements.txt
```

## Usage

Run the FastAPI server:
```bash
python -m uvicorn main:app --reload
```

Optional: configure AI metadata keywords (comma-separated) used by Layer 2:
```bash
set AI_METADATA_KEYWORDS=midjourney,stable diffusion,openai,firefly
```

You can also copy .env.example to .env and set the value there.

Access the API documentation at: `http://127.0.0.1:8000/docs`

## API Endpoint

**POST** `/analyze`
- Upload an image file
- Returns detection results with confidence score and detailed analysis

## Project Structure

```
DeepFake_Detection/
├── main.py                          # FastAPI application entry point
├── temp_uploads/                    # Temporary upload directory
└── Universal_Detector/
    ├── requirements.txt
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

## License

MIT
