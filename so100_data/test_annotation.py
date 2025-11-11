#!/usr/bin/env python3
"""
Test script for VLM annotation system

This script tests the annotation system on a small subset of datasets
to verify everything works correctly before running the full batch.
"""

import json
import logging
import os
import sys
import traceback
from pathlib import Path

# Add the current directory to the path so we can import vlm_annot
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from so100_data.vlm_annot import VLMAnnotator

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_single_dataset():
    """Test annotation on a single dataset."""
    # Test with a small, simple dataset
    # test_repo_id = "jiuyal2/so100_marker"
    # test_repo_id = "samsam0510/tooth_extraction_4"
    # test_repo_id = "HITHY/so100_peach1"
    # test_repo_id = "AK51/so100_test6"
    test_repo_id = "Aravindh25/so100_orange_basket1"
    model_name = "Qwen/Qwen3-VL-8B-Instruct"

    logger.info(f"Testing annotation on dataset: {test_repo_id}")
    logger.info(f"Using model: {model_name}")

    # Initialize annotator with smaller model for testing
    annotator = VLMAnnotator(
        model_name=model_name,
        device="auto",
        batch_size=1,  # Smaller batch size for testing
        temperature=0.0,
        max_length=512,
        max_refinements=1,
    )

    # Create output directory
    output_dir = Path("so100_data/test_annotations") / test_repo_id.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run annotation with limited episodes for testing
        results = annotator.batch_annotate_hf_dataset(
            repo_id=test_repo_id,
            output_path=str(output_dir),
            episodes=[0, 1, 2],  # Only process first 3 episodes for testing
            random_seed=42,
            download_videos=True,
            force_cache_sync=False,
        )
        logger.info("Test completed successfully!")

        # Save test results
        test_results_path = output_dir / "test_results.json"
        with open(test_results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Test results saved to: {test_results_path}")

        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    """Main test function."""
    logger.info("Starting VLM annotation test...")

    success = test_single_dataset()

    if success:
        logger.info("✅ Test passed! The annotation system is working correctly.")
        logger.info("You can now run the full batch annotation script.")
    else:
        logger.error("❌ Test failed! Please check the error messages above.")
        logger.error("The annotation system needs to be fixed before running the full batch.")


if __name__ == "__main__":
    main()
