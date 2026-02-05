from PIL import Image
import os

# ==========================================
# LAYER 1: QUICK FORENSIC TRIAGE
# Purpose: Reject broken files, tiny icons, or "pixel bombs"
# ==========================================

def quick_check(file_path):
    """
    Analyzes file structure before deep processing.
    Returns: {'status': 'PASS'/'FAIL', 'reason': ...}
    """
    print(f"🔍 Layer 1 Analyzing: {file_path}...")
    
    try:
        # A. Check if it is a valid image format
        with Image.open(file_path) as img:
            img.verify()  # PIL built-in integrity check
        
        # B. Check Resolution Constraints
        # Re-open required because .verify() closes the file
        with Image.open(file_path) as img:
            width, height = img.size
            file_format = img.format
            
            # Reject Tiny Images (Icons/Thumbnails)
            if width < 100 or height < 100:
                return {"status": "FAIL", "reason": "Image too small (likely an icon)"}

            # Reject "Pixel Bombs" (Massive images meant to crash your server)
            if width > 6000 or height > 6000:
                return {"status": "FAIL", "reason": "Image resolution too high (DOS Protection)"}

        return {"status": "PASS", "details": "File structure is valid."}

    except Exception as e:
        return {"status": "FAIL", "reason": f"Corrupted or non-image file: {str(e)}"}

# ==========================================
# LOCAL TESTER (Only runs if you run this file directly)
# ==========================================
if __name__ == "__main__":
    # Create a dummy file to test
    print("🧪 Running Local Test...")
    with open("test.txt", "w") as f: f.write("Not an image")
    
    # Test the function
    print(quick_check("test.txt"))  # Should FAIL
    
    # Clean up
    if os.path.exists("test.txt"): os.remove("test.txt")