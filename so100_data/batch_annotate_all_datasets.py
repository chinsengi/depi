#!/usr/bin/env python3
"""
Batch annotation script for all datasets in dataset_list_valid.json

This script iterates through all datasets in the dataset list and runs VLM annotation
on each one, saving the results in an organized directory structure.
"""

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

# Add the current directory to the path so we can import vlm_annot
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vlm_annot import VLMAnnotator

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_dataset_list(file_path: str) -> List[str]:
    """Load the list of dataset repository IDs from the JSON file."""
    try:
        with open(file_path) as f:
            dataset_list = json.load(f)
        logger.info(f"Loaded {len(dataset_list)} datasets from {file_path}")
        return dataset_list
    except Exception as e:
        logger.error(f"Failed to load dataset list from {file_path}: {e}")
        raise


def create_output_directory(base_path: str, repo_id: str) -> Path:
    """Create output directory for a specific dataset."""
    # Clean repo_id for filesystem use
    safe_repo_id = repo_id.replace("/", "_")
    output_dir = Path(base_path) / safe_repo_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def annotate_single_dataset(
    annotator: VLMAnnotator,
    repo_id: str,
    output_base_path: str,
    max_episodes: Optional[int] = None,
    random_seed: int = 42,
) -> Dict:
    """Annotate a single dataset and return results."""
    try:
        logger.info(f"Starting annotation for dataset: {repo_id}")

        # Create output directory
        output_dir = create_output_directory(output_base_path, repo_id)

        # Run annotation
        results = annotator.batch_annotate_hf_dataset(
            repo_id=repo_id,
            output_path=str(output_dir),
            episodes=None
            if max_episodes is None
            else list(range(max_episodes)),  # Process all episodes unless max_episodes is specified
            random_seed=random_seed,
            download_videos=True,
            force_cache_sync=False,
        )

        # Add metadata
        results["repo_id"] = repo_id
        results["output_path"] = str(output_dir)
        results["timestamp"] = time.time()

        logger.info(
            f"Completed annotation for {repo_id}: {results['successful_annotations']}/{results['total_episodes']} successful"
        )

        return results

    except Exception as e:
        logger.error(f"Failed to annotate dataset {repo_id}: {e}")
        logger.error(traceback.format_exc())
        return {
            "repo_id": repo_id,
            "error": str(e),
            "successful_annotations": 0,
            "total_episodes": 0,
            "success_rate": 0.0,
            "average_confidence": 0.0,
            "timestamp": time.time(),
        }


def main():
    """Main function to batch annotate all datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch annotate all datasets in dataset_list_valid.json")
    parser.add_argument(
        "--dataset_list_path",
        type=str,
        default="data_ids/dataset_list_valid.json",
        help="Path to the dataset list JSON file",
    )
    parser.add_argument(
        "--output_base_path",
        type=str,
        default="so100_data/annotations",
        help="Base path for saving annotation results",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="VLM model name"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to run inference on")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--max_episodes", type=int, default=None, help="Maximum number of episodes to process per dataset"
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="Start from this dataset index (for resuming)"
    )
    parser.add_argument(
        "--end_index", type=int, default=None, help="End at this dataset index (for processing subset)"
    )
    parser.add_argument("--max_refinements", type=int, default=1, help="Maximum number of refinements")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the generated text")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    # Load dataset list
    dataset_list = load_dataset_list(args.dataset_list_path)

    # Apply index filtering
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index is not None else len(dataset_list)
    dataset_list = dataset_list[start_idx:end_idx]

    logger.info(f"Processing {len(dataset_list)} datasets (indices {start_idx} to {end_idx - 1})")

    # Create output base directory
    output_base_path = Path(args.output_base_path)
    output_base_path.mkdir(parents=True, exist_ok=True)

    # Initialize annotator
    logger.info(f"Initializing VLM annotator with model: {args.model_name}")
    annotator = VLMAnnotator(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_length=args.max_length,
        max_refinements=args.max_refinements,
    )

    # Process each dataset
    all_results = []
    successful_datasets = 0
    failed_datasets = 0

    for repo_id in tqdm(dataset_list, total=len(dataset_list), desc="Processing datasets"):
        # Check if annotation already exists
        # annotation_path = output_base_path / repo_id.replace("/", "_") / "annotations.json"
        # if annotation_path.exists():
        #     logger.info(f"Skipping {repo_id} - annotation already exists")
        #     continue
        try:
            result = annotate_single_dataset(
                annotator=annotator,
                repo_id=repo_id,
                output_base_path=str(output_base_path),
                max_episodes=args.max_episodes,
                random_seed=args.random_seed,
            )

            all_results.append(result)

            if "error" not in result:
                successful_datasets += 1
            else:
                failed_datasets += 1

        except Exception as e:
            logger.error(f"Unexpected error processing {repo_id}: {e}")
            failed_datasets += 1
            all_results.append(
                {
                    "repo_id": repo_id,
                    "error": str(e),
                    "successful_annotations": 0,
                    "total_episodes": 0,
                    "success_rate": 0.0,
                    "average_confidence": 0.0,
                    "timestamp": time.time(),
                }
            )

    # Save summary results
    summary = {
        "total_datasets": len(dataset_list),
        "successful_datasets": successful_datasets,
        "failed_datasets": failed_datasets,
        "success_rate": successful_datasets / (successful_datasets + failed_datasets)
        if (successful_datasets + failed_datasets) > 0
        else 0.0,
        "results": all_results,
        "parameters": vars(args),
    }

    summary_path = output_base_path / "batch_annotation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    logger.info("=" * 80)
    logger.info("BATCH ANNOTATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total datasets processed: {len(dataset_list)}")
    logger.info(f"Successful datasets: {successful_datasets}")
    logger.info(f"Failed datasets: {failed_datasets}")
    logger.info(f"Overall success rate: {summary['success_rate']:.2%}")
    logger.info(f"Results saved to: {summary_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
