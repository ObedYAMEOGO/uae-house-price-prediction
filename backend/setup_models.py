#!/usr/bin/env python
"""Pre-download models at container startup."""
from backend.model_loader import load_model, load_feature_engineer

if __name__ == "__main__":
    print("Downloading model...")
    load_model()
    print("Downloading feature engineer...")
    load_feature_engineer()
    print("✓ Models ready!")