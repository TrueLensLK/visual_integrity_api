import numpy as np
import cv2
from scipy.fftpack import fft2, fftshift

def analyze_spectrum(image_path):
    """
    Layer 3: Frequency Domain Forensics (FFT)
    Detects the 'invisible grid' left by AI upsampling.
    """
    try:
        # 1. Load Image in Grayscale
        img = cv2.imread(image_path, 0)
        if img is None:
            return 0, "Error loading image"

        h, w = img.shape
        
        # 2. Compute Fast Fourier Transform (FFT)
        # This converts the image from "Pixels" to "Frequencies"
        f = fft2(img)
        fshift = fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

        # 3. Calculate Azimuthal Average (Radial Profile)
        # We average the energy at every radius from the center.
        # Real images have a smooth drop-off. AI images have "spikes".
        
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(int)

        # Sum energy per radius
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / (nr + 1e-8) # Average energy per radius

        # 4. Analyze the High Frequencies (The "Deepfake Zone")
        # AI struggles to generate realistic high-frequency noise (fine grain).
        # It usually drops off too fast or has spikes.
        
        # Normalize profile 
        radial_profile = radial_profile / np.max(radial_profile)
        
        # Check the last 20% of frequencies (High details)
        high_freq_energy = np.mean(radial_profile[-int(len(radial_profile)*0.2):])
        
        print(f"   [Spectrum] High Freq Energy: {high_freq_energy:.4f}")

        # 5. The Verdict Logic
        # Real photos (Noise) -> High energy at the end ( > 0.25 usually)
        # AI (Smooth/Upscaled) -> Low energy at the end ( < 0.15 usually)
        
        score = 0
        description = "Neutral"

        if high_freq_energy < 0.12:
            score = -40
            description = "Artificial High-Frequency Drop-off (AI Blur)"
        elif high_freq_energy < 0.20:
            score = -20
            description = "Suspiciously smooth texture"
        elif high_freq_energy > 0.35:
            score = 30
            description = "Natural Sensor Noise Detected"
        else:
            score = 0
            description = "Inconclusive Spectrum"

        print(f"   [Spectrum] Score: {score} ({description})")
        return score, description

    except Exception as e:
        print(f"   [Spectrum Error] {e}")
        return 0, "Error"