def calculate_integrity(meta_score, physics_score, visual_score, face_score, spectrum_score, eye_score):
    """
    Layer 5: The All-Seeing Judge (Final Version).
    Aggregates scores from Metadata, Digital Physics, Visual AI, Face Analysis, Spectrum, and Optical Physics (Eyes).
    
    Logic:
    - Base Score: 50 (Neutral)
    - "Resolution Trap": AI Dimensions -> Cap Score.
    - "Physics Veto": Impossible Eyes -> FAKE (Immediate).
    - "Web Amnesty": No Meta + Good Visuals -> REAL.
    """
    
    # 1. Log the inputs for debugging
    print(f"   [Judge] Evidence: Meta={meta_score}, Eye={eye_score}, Vis={visual_score}, Spec={spectrum_score}, Face={face_score}, Phys={physics_score}")

    # --- RULE 1: THE PHYSICS VETO (The "Soul" Check) ---
    # This is the strongest rule. If the eyes reflect different worlds, it is physically impossible.
    # No amount of metadata or visual quality can save it.
    if eye_score <= -40:
        print("   [Judge] 🚨 Physics Violation detected (Mismatched Eyes).")
        return -50, "FAKE", "Optical physics violation: Eyes reflect different light sources."

    # --- RULE 2: THE RESOLUTION TRAP ---
    # If Metadata found AI dimensions (e.g., 1024x1024), we cap the score.
    if meta_score < 0:
        print("   [Judge] 🚨 AI Resolution detected. Capping score.")
        return 35, "SUSPICIOUS", "Image dimensions match AI generation defaults."

    # --- RULE 3: The "Uncanny Veto" ---
    # If Visuals are HORRIBLE, it's fake.
    if visual_score <= -40:
        return -50, "FAKE", "Visual artifacts detected (High Confidence)."
    
    # --- RULE 4: Spectrum Forgiveness ---
    # Only Veto if Spectrum is CATASTROPHICALLY bad (-50).
    # Ignore minor compression artifacts (-20).
    if spectrum_score <= -50: 
        return -50, "FAKE", "Synthetic texture detected."

    # --- RULE 5: Weighted Average (Updated for 6 Layers) ---
    # We give high weight to Visuals (Smartest) and Eyes (Hardest to fake).
    weighted_score = (
        (visual_score * 0.35) +      # Visual Brain (Overall look)
        (eye_score * 0.25) +         # Optical Physics (The Soul)
        (spectrum_score * 0.15) +    # Frequency Analysis
        (face_score * 0.15) +        # Facial Consistency
        (meta_score * 0.05) +        # Metadata (Least trusted, easily stripped)
        (physics_score * 0.05)       # Digital Physics (ELA/Noise)
    )
    
    final_score = 50 + weighted_score

    # --- RULE 6: The "Web Image Amnesty" ---
    # If Metadata is missing (0) BUT Visuals are strong (>20) AND Eyes are valid (>0), 
    # we assume it's a real image from the internet.
    if meta_score == 0:
        if visual_score >= 20 and eye_score >= 0:
             # Boost the score slightly because the eyes and brain agree
            final_score += 10 
            print("   [Judge] 🌐 Web Image Detected (No Meta + Good Physics). Boosting score.")
        else:
            # If Visuals are weak AND no metadata, cap it.
            final_score = min(final_score, 65)

    # Clamp Score
    final_score = max(0, min(100, final_score))
    
    # Verdicts
    if final_score >= 75:
        verdict = "REAL"
        desc = "High authentic integrity. Physics and Visuals align."
    elif final_score >= 60:
        verdict = "LIKELY REAL"
        desc = "Consistent structure, likely authentic."
    elif final_score >= 40:
        verdict = "UNCERTAIN"
        desc = "Inconclusive signals."
    elif final_score >= 20:
        verdict = "SUSPICIOUS"
        desc = "Potential manipulation or AI Dimensions."
    else:
        verdict = "FAKE"
        desc = "Strong evidence of AI generation or Physics violation."
        
    return final_score, verdict, desc

# ==========================================
# LOCAL TESTER
# ==========================================
if __name__ == "__main__":
    print("Testing Judge with 6 Layers...")
    
    # Test 1: The "Rain Girl" (Midjourney)
    # Meta: -30 (1024px) | Eyes: -50 (Bad Reflections) | Visual: +40 (Looks real)
    s, v, d = calculate_integrity(-30, 0, 40, 0, 0, -50)
    print(f"Scenario 1 (Rain Girl): {s} - {v} (Expected: FAKE)")
    
    # Test 2: Real Google Image
    # Meta: 0 (Missing) | Eyes: +30 (Real) | Visual: +30 (Real)
    s2, v2, d2 = calculate_integrity(0, 10, 30, 10, -10, 30)
    print(f"Scenario 2 (Google Image): {s2} - {v2} (Expected: REAL)")