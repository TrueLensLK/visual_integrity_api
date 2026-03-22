# DeepFake Detection System Architecture

## 1. System Overview (Visual Graph)

```mermaid
graph TD
    User([User / Client]) -->|Upload Image| API[FastAPI Entry Point]
    API --> L0[Layer 0: C2PA Provenance]
    L0 -->|Valid Signature?| Verdict1{Checks}
    Verdict1 -->|Signed & Valid| EndReal([Verdict: REAL (Provenanced)])
    Verdict1 -->|No Sig / Unknown| L1[Layer 1: Triage]
    
    L1 -->|Valid File| Parallel[Parallel Forensic Analysis]
    
    subgraph "The Forensic Engine (Layers)"
        direction TB
        L2[L2: Metadata] -->|EXIF/Tags| J_Input
        L3[L3: Physics] -->|ELA/Noise| J_Input
        L35[L3.5: Face] -->|Consistency| J_Input
        L4[L4: Visual AI] -->|Artifacts| J_Input
        L6[L6: Spectrum] -->|Freq Grids| J_Input
        L7[L7: Eyes] -->|Reflections| J_Input
        L8[L8: Watermark] -->|SynthID/Stego| J_Input
        L85[L8.5: PRNU] -->|Sensor Noise| J_Input
        L10[L10: Shadow] -->|Geometry| J_Input
    end
    
    Parallel --> L2 & L3 & L35 & L4 & L6 & L7 & L8 & L85 & L10
    
    J_Input((Score Integration)) --> Judge[Layer 5: THE JUDGE]
    
    subgraph "The Judge's Logic"
        Judge --> T1[JPEG Dampening]
        T1 --> T2{Tier 1.5: Strong Watermark?}
        T2 -->|Yes (>99%)| V_AI([Verdict: AI-Generated])
        T2 -->|No| T3{Tier 2: Physical Impossibility?}
        T3 -->|Yes (Bad Eyes/Shadow)| V_AI
        T3 -->|No| T4[Tier 3: Voting Consensus]
        T4 --> Score[Calculate Integrity Score]
    end
    
    Score --> Final([Final Verdict: Real / Fake / Uncertain])
```

## 2. Verbal Explanation (How it Works)

The system operates like a digital detective squad, where 9 different specialists examine the same image simultaneously, and one "Judge" makes the final decision.

### **Phase 1: The Gatekeepers**
1.  **Layer 0 (Provenance)**: The file is checked for a "digital passport" (C2PA/Content Credentials). If a camera signed it (like a Sony/Leica) or Adobe confirms it, we trust it immediately.
2.  **Layer 1 (Triage)**: Filters out broken files, tiny thumbnails, or unsupported formats.

### **Phase 2: The Specialists (Parallel Layers)**
Each layer runs independently and produces a score (-50 for Fake to +50 for Real).

*   **L2 Metadata**: Reads hidden text tags. Looks for "Adobe Firefly" or "Midjourney" tags (Fake) vs. valid iPhone/Canon sensor data (Real).
*   **L3 Digital Physics**: Looks at the pixels. Detects if objects were "pasted" (ELA) or if the image is impossibly smooth (Denoising artifacts).
*   **L3.5 Face Consistency**: Checks if the face matches the background. (e.g., A high-def face on a blurry background is suspicious).
*   **L4 Visual AI**: A 6-model neural ensemble (SDXL, ViT, Ateeqq, ConvNeXt, Swin, Deepfake) trained to spot visual glitches humans miss (e.g., 6 fingers, melting ears).
*   **L6 Spectrum**: Uses math (FFT) to look at invisible frequencies. AI generators create perfect "grid" patterns; real cameras create messy noise.
*   **L7 Eyes**: Physics check. Do the reflections in the eyes match? (e.g., light coming from left in one eye, right in the other).
*   **L8 Watermark**: Scans for hidden AI barcodes (SynthID, Stable Diffusion invisible watermarks).
*   **L8.5 Sensor (PRNU)**: Checks for the unique "fingerprint" of the camera sensor.
*   **L10 Shadow**: Checks if shadows fall in the correct direction based on light sources.

### **Phase 3: The Judge (Decision Logic)**
The Judge (`Layer 5`) doesn't just add up points. It uses a **Tiered Logic System**:

1.  **Preprocessing (JPEG Dampening)**: If the image is a compressed JPEG, the Judge *reduces* the trust in Layer 6 (Spectrum) and Layer 8 (Watermark) because compression looks like "AI Artifacts."
2.  **Tier 1.5 (The "Smoking Gun")**: If Layer 8 finds a **definitive** AI Watermark (and it's not a False Positive), the verdict is immediately **AI**.
3.  **Tier 2 (The "Laws of Physics")**: If Layer 7 (Eyes) or Layer 10 (Shadow) proves the image is physically impossible, the verdict is **AI**.
4.  **Tier 3 (Consensus)**: If no "Smoking Gun" is found, the Judge counts how many layers vote "Real" vs "Fake."
    *   3+ Layers saying "Fake" = **AI Verdict**.
    *   Mixed signals = **Uncertain**.
    *   Mostly "Real" signals = **Real Verdict**.

### **Phase 4: The Final Boss (LLM + Adversarial Debate)**
If the Judge's verdict is ambiguous or forensic layers contradict each other, the system escalates:

1.  **Gray Zone (Score 35-65)**: A single LLM vision call (Gemini or OpenRouter) reviews the image and case file.
2.  **Contradiction Detected** (e.g., PRNU says Fake but Neural says Real): The **Adversarial Debate** activates:
    *   **Prosecution** (Gemini Vision): Argues the image is AI-generated, citing forensic evidence.
    *   **Defense** (OpenRouter Vision): Argues the image is real, challenging the prosecution's evidence.
    *   **Convergence Judge** (Groq text): Reads the debate transcript as a neutral observer and delivers the verdict.
    *   Max **3 rounds**. Image is only sent in Round 1; Rounds 2-3 are text-only rebuttals to control cost.
    *   Different LLM providers ensure genuine **epistemic diversity** (not same-model self-debate).

### **Output**
*   **Integrity Score**: A number from 0 to 100. (0 = Definite Deepfake, 100 = Definite Real).
*   **Verdict**: "AI-Generated", "Real", or "Uncertain".
*   **Explanation**: A human-readable summary (e.g., *"Artificial grid patterns detected, mismatched eye reflections"*).
