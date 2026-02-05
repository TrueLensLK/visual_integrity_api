import PIL.Image
import PIL.ExifTags

def analyze_metadata(image_path):
    """
    Layer 2: Metadata Analysis
    Checks for:
    1. AI Generation tools 
    2. Missing Exif data (common in AI images)
    
    Returns: (Score, Details_Dictionary) 
    """
    try:
        img = PIL.Image.open(image_path)
        exif_data = img._getexif()
        
        score = 0
        tags_found = {}

        # No Metadata at all (Suspicious for "Photography", normal for AI)
        if not exif_data:
            print("   [Metadata] No EXIF data found (Suspicious)")
            # We return the score AND an empty dict explanation
            return 0, {"info": "No EXIF data found"}

        # Scan for AI Keywords
        ai_keywords = ['midjourney', 'stable diffusion', 'dall-e', 'firefly', 'generated']
        
        for tag_id, value in exif_data.items():
            # Get the human-readable tag name 
            tag_name = PIL.ExifTags.TAGS.get(tag_id, tag_id)
            
            # Store it for the final report
            tags_found[str(tag_name)] = str(value)
            
            # Check if value contains AI terms
            value_str = str(value).lower()
            for keyword in ai_keywords:
                if keyword in value_str:
                    print(f"   [Metadata] AI Signature found: {keyword}")
                    return -50, {"detected_tool": keyword} # FAKE

        # Check for Editing Software
        if 'Software' in tags_found:
            software = tags_found['Software'].lower()
            if 'photoshop' in software:
                print("  [Metadata] Edited with Photoshop.")
                return -10, {"software": "Photoshop"} # Slight penalty

        # If we reach here, it seems clean
        return 20, {"status": "Clean metadata", "camera_info": tags_found.get("Model", "Unknown")} # REAL

    except Exception as e:
        print(f"   [Metadata Error] {e}")
        return 0, {"error": str(e)}

# ==========================================
# LOCAL TESTER
# ==========================================
if __name__ == "__main__":
    # Test with a dummy call to see if it returns two values
    print("To test Layer 2, place a .jpg file here and run this script.")