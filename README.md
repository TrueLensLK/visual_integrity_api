# Multi-Layer DeepFake Detection System

A high-precision, enterprise-grade forensic engine designed to detect AI-generated and manipulated imagery. This system employs a "Defense in Depth" strategy, stacking 15+ analysis layers ranging from cryptographic verification and digital physics to neural ensembles and adversarial LLM debates.

![System Status](https://img.shields.io/badge/Status-Active-success)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![API Type](https://img.shields.io/badge/API-FastAPI-009688)

## 🚀 Key Features

The system uses a consensus mechanism where multiple independent forensic layers vote on the image's authenticity.

### 🛡️ Layer 0: Cryptographic Truth
- **C2PA/Content Credentials**: Verifies digital signatures and provenance chains (Adobe, Microsoft, etc.).
- **Immediate Trust**: If valid C2PA credentials from a trusted issuer are found, the system can short-circuit to a "verified" verdict.

### 🔍 Low-Level Forensics (Digital Physics)
- **Layer 1: Triage**: Rapid file header analysis to detect obvious mismatches and broken files.
- **Layer 2: Metadata**: Scans EXIF/XMP for traces of AI tools (e.g., "Diffusers", "Midjourney") and missing camera data.
- **Layer 3: ELA & Noise**: Error Level Analysis (ELA) and noise print analysis to detect spliced areas or inconsistent compression.
- **Layer 6: Frequency Spectrum**: FFT (Fast Fourier Transform) analysis to detect grid-like artifacts common in GANs and early Diffusion models.
- **Layer 8: Watermarks**: Detects invisible signatures (Digimarc, SynthID traces) and visible AI tool watermarks.
- **Layer 8.5: PRNU**: Photo Response Non-Uniformity analysis to match images to specific camera sensor fingerprints.

### 🧠 Semantic & Physics Analysis
- **Layer 3.5: Face Consistency**: Compares foreground faces against the background for lighting and resolution mismatches.
- **Layer 7: Eye Physics**: Analyzes corneal reflections (gaze direction, light source consistency across both eyes).
- **Layer 10: Shadow Convergence**: Checks if shadows cast by objects converge to a single, consistent light source.
- **Layer 11: Physical Continuity**: Checks for vanishing point consistency and geometric logic.

### 🤖 Neural Detection Ensemble (Layer 4)
A voting block of 5 specialized computer vision models:
1. **SDXL-Detector**: Specialized for Stable Diffusion XL artifacts.
2. **ViT (Vision Transformer)**: General purpose anomaly detection.
3. **SigLIP 2**: Multimodal embedding analysis.
4. **ConvNeXt**: High-fidelity feature extraction.
5. **Swin Transformer**: Hierarchical visual processing.

### ⚖️ The "Final Boss" (Layers 5 & 12)
- **Layer 5: Standard Judge**: A weighted algorithm that aggregates all layer scores into a final probability 0-100.
- **LLM Judge**: For "Gray Zone" cases (score 35-65), a Vision LLM (Gemini/GPT-4o) acts as a human expert, reviewing the visual evidence and forensic logs.
- **Adversarial Debate**: In highly ambiguous cases, the system spawns two AI agents—a "**Prosecutor**" and a "**Defense Attorney**"—who argue over the evidence. A third "**Judge**" AI delivers the final verdict based on their debate.

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- Git
- (Optional) Docker for containerized deployment

### 1. Clone & Setup
```bash
git clone <your-repo-url>
cd DeepFake_Detection
```

### 2. Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```

**Required Keys for Full Functionality:**
```ini
# At least one LLM provider is needed for the "Final Boss" layers
GOOGLE_AI_API_KEY=your_gemini_key
# OR
OPENROUTER_API_KEY=your_openrouter_key
# OR
GROQ_API_KEY=your_groq_key

# System Flags
ENABLE_LLM_JUDGE=true
```

---

## 🏃 Usage

### Start API Server
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
- **Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health**: [http://localhost:8000/health](http://localhost:8000/health)

### API Endpoints

#### 1. Analyze Local File
`POST /analyze`
Uploads an image file for immediate analysis.
```bash
curl -X POST "http://localhost:8000/analyze" -F "file=@/path/to/image.jpg"
```

#### 2. Analyze URL
`POST /analyze-url`
Downloads and analyzes an image from a public URL (supports s3, http, https).
```json
{
  "s3_url": "https://example.com/image.jpg"
}
```

### Response Format
The API returns a detailed JSON report:
```json
{
  "final_score": 95,
  "verdict": "AI-GENERATED",
  "confidence": "HIGH",
  "technical_description": "...",
  "user_description": "...",
  "layer_scores": {
    "c2pa": 0,
    "metadata": 45,
    "physics": 20,
    "neural_network": 98,
    "...": "..."
  },
  "judge_source": "debate" // rule-based, llm, or debate
}
```

---

## 🐳 Docker Deployment

Run the system in a self-contained environment.

```bash
# 1. Build Image
docker build -t deepfake-detector .

# 2. Run Container (Map port 8000)
docker run -d -p 8000:8000 --env-file .env --name detector deepfake-detector
```
*Note: The first startup may take a few minutes to download the neural model weights.*

---

## 🧪 Validation & Testing

To test the system against a labeled dataset (Real vs Fake):

1. **Prepare Data**:
   Place images in `validation_dataset/real/` and `validation_dataset/fake/`.

2. **Run Validation Script**:
   ```bash
   python validate.py
   ```
   This will run the full pipeline on all images and generate a statistical report (`VALIDATION_REPORT.md`), including Accuracy, Precision, Recall, and F1-Score.

---

## 📂 Project Structure

```
DeepFake_Detection/
├── main.py                        # API Entry Point & Orchestrator
├── validate.py                    # Validation/Testing Suite
├── train_custom_model.py          # (Experimental) Custom Model Training
├── requirements.txt               # Dependency Locking
├── Universal_Detector/
│   └── src/
│       ├── layers/                # ALL Forensic Logic
│       │   ├── layer_0_c2pa.py    # Content Credentials
│       │   ├── layer_1_triage.py  # Validation
│       │   ├── layer_2_metadata.py
│       │   ├── layer_3_physics.py # Noise/ELA
│       │   ├── layer_4_visual.py  # Neural Ensemble
│       │   ├── layer_5_judge.py   # Rule-Based Verdicts
│       │   ├── layer_6_spectrum.py
│       │   ├── layer_7_eyes.py
│       │   ├── layer_10_Shadow_Convergence.py
│       │   ├── layer_12_artifacts.py
│       │   ├── forensic_case_builder.py
│       │   ├── llm_judge.py       # "Final Boss" Logic
│       │   └── debate/            # Adversarial Agents
│       └── utils/
└── temp_uploads/                  # Ephemeral storage for processing
```

## 🛠️ Troubleshooting

- **"OpenRouter API Key MISSING"**: The debate feature will be disabled. Set `OPENROUTER_API_KEY` in `.env`.
- **High Latency**: The first run downloads heavy model weights (~2GB). Subsequent runs are faster.
- **GPU Usage**: The system defaults to CPU. To use CUDA, ensure `torch` is installed with CUDA support.

## 📄 License
MIT License - Open for research and educational use.

