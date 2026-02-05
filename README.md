# DeepFake Detection System

A multi-layered AI-powered deepfake detection system using various analysis techniques including metadata analysis, digital physics, facial consistency, visual AI, frequency spectrum analysis, and optical physics (eye reflection analysis).

## Features

- **Layer 2: Metadata Analysis** - Detects AI generation tools and missing EXIF data
- **Layer 3: Digital Physics** - ELA (Error Level Analysis) and noise detection
- **Layer 3.5: Face Consistency** - Analyzes facial landmark consistency
- **Layer 4: Visual AI** - EfficientNet-based deep learning model
- **Layer 6: Spectrum Analysis** - Frequency domain analysis
- **Layer 7: Eye Analysis** - Optical physics and corneal reflection detection
- **Layer 5: The Judge** - Aggregates all layer scores for final verdict

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
            └── efficientnet_b0.pth      # Pre-trained model
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
