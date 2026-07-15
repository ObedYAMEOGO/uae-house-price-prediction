#!/usr/bin/env python
"""Pre-download models at container startup."""
import sys
import logging

from backend.model_loader import load_feature_engineer, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    
    logger.info("Downloading model...")
    load_model()
    logger.info("✓ Model downloaded")
    
    logger.info("Downloading feature engineer...")
    load_feature_engineer()
    logger.info("✓ Feature engineer downloaded")
    
except Exception as e:
    logger.error(f"Failed to download models: {e}")
    sys.exit(1)