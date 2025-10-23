"""
Advanced VLM Annotation System for Task Description Quality Improvement

This module provides a comprehensive system for generating high-quality task instructions
using Qwen2.5-VL-3B-Instruct with advanced prompting strategies, quality control,
and iterative refinement to improve annotation quality.
"""

import json
import logging
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.generation import GenerationConfig

# Import LeRobot dataset functionality
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
)  # Add filename to log format
logger = logging.getLogger(__name__)  # Line 16


class VLMAnnotator:
    """
    Advanced VLM-based task instruction annotator using Qwen2.5-VL-3B-Instruct.

    Features:
    - Multi-stage prompting for better quality
    - Quality control and validation
    - Iterative refinement
    - Batch processing with error handling
    - HuggingFace dataset integration
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "auto",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch_size: int = 4,
        max_refinements: int = 1,
    ):
        """
        Initialize the VLM annotator.

        Args:
            model_name: HuggingFace model name
            device: Device to run inference on ('auto', 'cuda', 'cpu')
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.max_refinements = max_refinements

        # Initialize model and processor
        self._load_model()

        # Quality control patterns
        self.quality_patterns = {
            "action_verbs": [
                "pick",
                "place",
                "move",
                "push",
                "pull",
                "grasp",
                "release",
                "rotate",
                "lift",
                "drop",
                "insert",
                "remove",
                "open",
                "close",
                "fold",
                "unfold",
                "pour",
                "scoop",
                "stir",
                "cut",
                "slice",
                "assemble",
                "disassemble",
            ],
            "object_indicators": ["the", "a", "an", "this", "that", "these", "those"],
            "spatial_indicators": [
                "on",
                "in",
                "at",
                "to",
                "from",
                "into",
                "onto",
                "under",
                "over",
                "beside",
                "next to",
                "between",
                "above",
                "below",
                "inside",
                "outside",
            ],
        }

        # Prompt templates for different stages
        self.prompt_templates = {
            "initial": self._get_initial_prompt(),
            "refinement": self._get_refinement_prompt(),
            "camera_pose": self._get_camera_pose_prompt(),
        }

    def _load_model(self):
        """Load the VLM model and processor."""
        try:
            logger.info(f"Loading model {self.model_name} on {self.device}")

            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            ).eval()
            self.model = torch.compile(self.model)

            # Set generation config
            if self.temperature == 0.0:
                self.generation_config = GenerationConfig(
                    max_new_tokens=self.max_length,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
            else:
                self.generation_config = GenerationConfig(
                    max_new_tokens=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _get_initial_prompt(self) -> str:
        """Generate the initial prompt for task instruction generation."""
        return """You are an expert robot arm task instruction generator. Here is a current task instruction: {original_instruction}. Given the demonstration videos, generate a clear imperative instruction describing the action performed by the robot arm. Carefully compare the starting state and the ending state of the object and infer how the robot arm achieve the goal. Pay attention to the color and the position of the object and the action. Make sure you get the color right and describe the destination of the object in detail, for example, if the object ends up in the box, include "put the object in the box" in the instruction.
        Start directly with an action verb."""

    def _get_refinement_prompt(self) -> str:
        """Generate the refinement prompt for improving instructions."""
        return """The following imperative robot arm task instruction may need improvement given the demonstration videos.

        Instruction to be improved: {current_instruction}

        Pay attention to the color, position of the object and the action. Make sure you get the color right and describe the end state of the object in detail. If the object is still with robot arm at the end, instruct the robot arm to hold the object. If the instruction is not correct, improve it.

        Does the instruction indicates the destination of the object? If not, improve it.

        Output the improved instruction directly without any other words."""

    def _get_camera_pose_prompt(self) -> str:
        """Generate the camera pose prompt for camera pose annotation."""
        return """Decide the camera pose given the demonstration videos. Choose from one of the two options:
                1. The fixed pose that is NOT mounted on the robot arm, so the view of the camera is fixed. For example, the camera is mounted on the table, so the view of the camera on the table is fixed. If you can see the base of the robot arm, it is the fixed pose: output "front" in this case.
                2. The wrist pose that is mounted on the wrist of the robot arm, so the view of the camera is moving. It is possible that you cannot see the robot arm. But if you can see the base of the robot arm, it is NOT the wrist pose: output "wrist" in this case.

                Output the camera pose directly without any other words."""

    def _generate_response(self, content_list: List[Dict[str, Any]]) -> str:
        """Generate response from the VLM using official Qwen2.5-VL guidance."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": content_list,
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Process vision info
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            inputs = inputs.to(self.device)

            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, generation_config=self.generation_config)

            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            return response.strip()

        except Exception as e:
            logger.error(f"description generation failed: {e}")
            logger.error(traceback.format_exc())
            return ""

    def generate_task_instruction(
        self,
        original_instruction: str,
        dataset: LeRobotDataset,
        episode_idx: int,
    ) -> Dict[str, Union[str, float, List[str]]]:
        """
        Generate a high-quality task instruction with iterative refinement.

        Args:
            image: Representative frame from robot demonstration
            original_instruction: Original instruction for context
            max_refinements: Maximum number of refinement iterations

        Returns:
            Dictionary containing:
            - instruction: Final task instruction
            - refinements: Number of refinements applied
        """
        try:
            # Initial generation
            initial_prompt = self.prompt_templates["initial"].format(
                original_instruction=original_instruction
            )

            # Prepare inputs using processor for multimodal model
            camera_keys = dataset.meta.camera_keys
            content_list = []
            for camera_key in camera_keys:
                video_path = dataset.root / dataset.meta.get_video_file_path(episode_idx, camera_key)
                video_path = str(video_path)  # Convert Path to string
                # For vision-language models, use image input
                content_list.append(
                    {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0}
                )
            content_list.append({"type": "text", "text": initial_prompt})
            current_instruction = self._generate_response(content_list)

            if not current_instruction:
                return {"instruction": original_instruction, "refinements": 0}

            # Iterative refinement
            refinements = 0

            for _ in range(self.max_refinements):
                # logger.info(f"Refinement {iteration+1} of {self.max_refinements}")
                # logger.info(f"Current instruction: {current_instruction}")
                # Refine instruction
                refinement_prompt = self.prompt_templates["refinement"].format(
                    current_instruction=current_instruction
                )
                content_list[-1]["text"] = refinement_prompt
                refined_instruction = self._generate_response(content_list)

                if refined_instruction and refined_instruction != current_instruction:
                    current_instruction = refined_instruction
                    refinements += 1
                else:
                    break
            # logger.info(f"Best instruction: {current_instruction}")

            return {"instruction": current_instruction, "refinements": refinements}

        except Exception as e:
            logger.error(f"Error generating task instruction: {e}")
            logger.error(traceback.format_exc())
            return {"instruction": "", "refinements": 0}

    def annotate_camera_pose(
        self, dataset: LeRobotDataset, episode_idx: int, camera_key: str
    ) -> Dict[str, List[str]]:
        """Annotate the camera pose for a given episode and camera key."""
        # Prepare inputs using processor for multimodal model
        camera_pose_prompt = self.prompt_templates["camera_pose"]
        content_list = []
        video_path = dataset.root / dataset.meta.get_video_file_path(episode_idx, camera_key)
        video_path = str(video_path)  # Convert Path to string
        # For vision-language models, use image input
        content_list.append({"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0})
        content_list.append({"type": "text", "text": camera_pose_prompt})
        camera_pose = self._generate_response(content_list)
        return camera_pose

    def batch_annotate_hf_dataset(
        self,
        repo_id: str,
        output_path: Union[str, Path],
        episodes: Optional[List[int]] = None,
        random_seed: int = 42,
        download_videos: bool = True,
        force_cache_sync: bool = False,
    ) -> Dict[str, Union[int, float, List[str]]]:
        """
        Batch annotate a HuggingFace LeRobot dataset with improved task instructions.

        Args:
            repo_id: HuggingFace dataset repository ID
            output_path: Path to save annotations
            episodes: List of episode indices to process (None for all episodes)
            random_seed: Random seed for reproducibility
            download_videos: Whether to download video files
            force_cache_sync: Whether to force cache synchronization

        Returns:
            Dictionary with annotation statistics
        """
        random.seed(random_seed)

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load LeRobot dataset
        logger.info(f"Loading dataset from HuggingFace: {repo_id}")
        try:
            dataset = LeRobotDataset(
                repo_id=repo_id,
                episodes=episodes,
                download_videos=download_videos,
                force_cache_sync=force_cache_sync,
            )
        except Exception as e:
            logger.error(f"Failed to load dataset {repo_id}: {e}")
            logger.error(traceback.format_exc())
            raise

        logger.info("Dataset loaded successfully:")
        logger.info(f"- Total episodes: {dataset.num_episodes}")
        logger.info(f"- Total frames: {dataset.num_frames}")
        logger.info(f"- Features: {list(dataset.features.keys())}")

        # Get camera keys for image processing
        camera_keys = dataset.meta.camera_keys
        if not camera_keys:
            raise ValueError(f"No camera keys found in dataset {repo_id}")

        logger.info(f"Camera keys: {camera_keys}")

        # Process episodes
        annotations = []
        total_episodes = dataset.num_episodes
        successful_annotations = 0

        logger.info(f"Processing {total_episodes} episodes...")

        for episode_idx in range(total_episodes):
            logger.info(f"Processing episode {episode_idx} ({episode_idx + 1}/{total_episodes})")

            # Get episode data
            episode_data = dataset.meta.episodes.get(episode_idx, {})
            episode_length = episode_data.get("length", 0)

            if episode_length == 0:
                logger.warning(f"No data found for episode {episode_idx}")
                continue

            # Get original instruction
            original_instruction = ""
            if "tasks" in episode_data and episode_data["tasks"]:
                original_instruction = episode_data["tasks"][0]

            # Get camera pose from dataset
            camera_pose = {}
            for camera_key in camera_keys:
                camera_pose[camera_key] = self.annotate_camera_pose(dataset, episode_idx, camera_key)

            # Generate improved instruction
            annotation_result = self.generate_task_instruction(original_instruction, dataset, episode_idx)

            if annotation_result["instruction"]:
                successful_annotations += 1

                # Save annotation
                annotation_data = {
                    "episode_index": episode_idx,
                    "original_instruction": original_instruction,
                    "improved_instruction": annotation_result["instruction"],
                    "refinements": annotation_result["refinements"],
                    "camera_pose": camera_pose,
                    "dataset_repo_id": repo_id,
                }

                annotations.append(annotation_data)

                logger.warning(
                    f"Episode {episode_idx}: {annotation_result['instruction']} Camera pose: {camera_pose}"
                )
            else:
                logger.warning(f"Failed to generate instruction for episode {episode_idx}")

        # Save results
        results = {
            "dataset_repo_id": repo_id,
            "total_episodes": total_episodes,
            "successful_annotations": successful_annotations,
            "success_rate": successful_annotations / total_episodes if total_episodes > 0 else 0.0,
            "annotations": annotations,
        }

        with open(output_path / "annotations.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Annotation complete. Success rate: {results['success_rate']:.2%}, ")

        return results


def main():
    """Main function for running the VLM annotation system."""
    import argparse

    parser = argparse.ArgumentParser(
        description="VLM Annotation System for Task Description Quality Improvement"
    )
    parser.add_argument(
        "--repo_id", type=str, default="HITHY/so100_peach1", help="HuggingFace dataset repository ID"
    )
    parser.add_argument(
        "--dataset_path", type=str, help="Path to local dataset directory (alternative to repo_id)"
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to save annotations")
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="VLM model name"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to run inference on")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--episodes", type=int, nargs="*", help="Specific episode indices to process")
    parser.add_argument("--download_videos", action="store_true", default=True, help="Download video files")
    parser.add_argument("--force_cache_sync", action="store_true", help="Force cache synchronization")
    parser.add_argument("--max_refinements", type=int, default=1, help="Maximum number of refinements")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum length of the generated text")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling")

    args = parser.parse_args()

    # Validate arguments
    if not args.repo_id and not args.dataset_path:
        parser.error("Either --repo_id or --dataset_path must be specified")

    # Initialize annotator
    annotator = VLMAnnotator(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_length=args.max_length,
        max_refinements=args.max_refinements,
    )

    # Run batch annotation
    if args.repo_id:
        # Use HuggingFace dataset
        results = annotator.batch_annotate_hf_dataset(
            repo_id=args.repo_id,
            output_path=args.output_path,
            episodes=args.episodes,
            random_seed=args.random_seed,
            download_videos=args.download_videos,
            force_cache_sync=args.force_cache_sync,
        )

    print("\nAnnotation Results:")
    print(f"Total episodes: {results['total_episodes']}")
    print(f"Successful annotations: {results['successful_annotations']}")
    print(f"Success rate: {results['success_rate']:.2%}")


if __name__ == "__main__":
    main()
