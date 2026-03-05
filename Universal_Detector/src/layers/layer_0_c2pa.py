import json
import os
import requests
from typing import Dict, List, Set
from pathlib import Path
from datetime import datetime, timedelta

#HARDCODED FALLBACKS
FALLBACK_GENERATORS = {
    "openai", "midjourney", "firefly", "sora", "gen-3", "kling", 
    "luma", "imagen", "gemini", "grok-3", "dall-e", "flux", 
    "sdxl", "leonardo", "runway", "ideogram", "black-forest-labs"
}

FALLBACK_IPTC_TYPES = {
    "trainedalgorithmicmedia", #IPTC's official term for AI-generated content
    "compositetrainedalgorithmicmedia", #IPTC's term for AI composites (e.g. human + AI elements)
    "algorithmicmedia", #General term for media created by algorithms (could be AI or procedural)
    "compositesynthetic", # IPTC's term for composites that may include synthetic elements
    "virtualreal" # IPTC's term for content that is entirely virtual or synthetic
}

class SignatureRegistry:
    # MULTIPLE SOURCES
    SOURCES = {
        "c2pa": "https://c2pa.org/conformance/products.json",
        "iptc": "https://cv.iptc.org/newscodes/digitalsourcetype/jsonld"
    }
    
    def __init__(self):
        # ABSOLUTE CACHE PATH
        self.cache_path = Path(__file__).parent.absolute() / "ai_registry_cache.json" #To download and store remote data for fallback and offline use
        self.cache_expiry = timedelta(hours=24) # Cache valid for 24 hours after that refresh from official sources 
        self.generators: Set[str] = set(FALLBACK_GENERATORS) 
        self.iptc_types: Set[str] = set(FALLBACK_IPTC_TYPES)
        self.last_updated = None # Timestamp of last successful update 
        
        self._load_from_cache() # Load from cache on initialization, fallback to hardcoded lists if cache is missing or corrupted

    def _load_from_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    data = json.load(f)
                    self.generators.update(data.get("generators", []))
                    self.iptc_types.update(data.get("iptc_types", []))
                    self.last_updated = datetime.fromisoformat(data["last_updated"])
            except Exception:
                pass # Fallback to hardcoded defaults on corruption

    def refresh_if_stale(self):
        # Updates from remote sources if the cache is >24h old.
        if not self.last_updated or (datetime.now() - self.last_updated) > self.cache_expiry:
            self._fetch_remotes()

    def _fetch_remotes(self):
        #Attempts to pull from official C2PA and IPTC registries
        updated = False
        try:
            # Sync C2PA Conforming Products
            resp = requests.get(self.SOURCES["c2pa"], timeout=5)
            if resp.status_code == 200:
                products = resp.json().get("products", [])
                new_gens = {p['name'].lower() for p in products if p.get('ai_gen') is True}
                self.generators.update(new_gens)
                updated = True
        except Exception: pass 

        if updated:
            self.last_updated = datetime.now()
            self._save_cache()

    def _save_cache(self):
        cache_data = {
            "generators": list(self.generators),
            "iptc_types": list(self.iptc_types),
            "last_updated": self.last_updated.isoformat()
        }
        with open(self.cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)

# LAZY INITIALIZATION ---
_REGISTRY_INSTANCE = None

def get_registry() -> SignatureRegistry:
    """Provides singleton access to the registry only when needed."""
    global _REGISTRY_INSTANCE
    if _REGISTRY_INSTANCE is None:
        _REGISTRY_INSTANCE = SignatureRegistry()
    return _REGISTRY_INSTANCE


def verify_c2pa(image_path: str) -> Dict:
    """Hardened C2PA verification with dynamic registry fallback."""
    try:
        from c2pa import Reader #C2PA is external lib so we import so system won't crash if it's missing, we just return an error message instead
    except ImportError:
        return {"status": "error", "message": "c2pa-python not installed"}

    # Trigger lazy registry & optional refresh
    registry = get_registry() # Gets the shared registry object
    registry.refresh_if_stale() #Gets the shared registry object

    try:
        reader = Reader(image_path)
        manifest_data = json.loads(reader.json())
        
        # Initialize Detection Flags
        is_ai = False
        findings = []
        
        for label, manifest in manifest_data.get("manifests", {}).items():
            result = _analyze_assertions(manifest, registry)
            if result["is_ai"]:
                is_ai = True
                findings.append(result["reason"])

        return {
            "status": "valid",
            "is_ai_flagged": is_ai,
            "forensic_matches": findings,
            "message": "AI Signature Detected" if is_ai else "Authentic Content"
        }
    except Exception as e:
        return {"status": "none", "is_ai_flagged": False, "message": str(e)}

def _analyze_assertions(manifest: Dict, registry: SignatureRegistry) -> Dict:
    """Uses registry to check both Generator names and IPTC types."""
    gen_name = manifest.get("claim_generator", "").lower()
    
    # Check Generator Names
    if any(sig in gen_name for sig in registry.generators):
        return {"is_ai": True, "reason": f"Generator Match: {gen_name}"}

    # Check IPTC DigitalSourceType 
    assertions = manifest.get("assertions", [])
    for asst in assertions:
        if "c2pa.actions" in asst.get("label", ""):
            for action in asst.get("data", {}).get("actions", []):
                stype = action.get("digitalSourceType", "").lower()
                if any(sig in stype for sig in registry.iptc_types):
                    return {"is_ai": True, "reason": f"IPTC Flag: {stype}"}
                    
    return {"is_ai": False, "reason": None}