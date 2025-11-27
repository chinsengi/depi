#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Modified by Shirui Chen, 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LIBERO environment-specific processor step.

This processor handles LIBERO's observation format, adding dummy agent_pos
when obs_type="pixels" to maintain compatibility with preprocess_observation.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("libero_env_processor")
@dataclass
class LiberoEnvProcessorStep(ProcessorStep):
    """
    Processor step for LIBERO environments.

    Adds a dummy agent_pos (zeros) when it's missing from observations.
    This ensures compatibility with preprocess_observation which expects agent_pos.

    Attributes:
        agent_pos_dim: Dimensionality of the agent position (default: 8 for LIBERO)
    """

    agent_pos_dim: int = 8

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Adds dummy agent_pos to observations if missing.

        Args:
            transition: The input EnvTransition object.

        Returns:
            EnvTransition with agent_pos added if it was missing.
        """
        observation = transition.get(TransitionKey.OBSERVATION)

        if observation is None:
            return transition

        # Check if agent_pos is missing
        if "agent_pos" not in observation:
            # Add dummy agent_pos (zeros) to match expected format
            # Determine batch size from pixels if available
            if "pixels" in observation:
                pixels_dict = observation["pixels"]
                # Get any image to determine batch size
                if isinstance(pixels_dict, dict) and pixels_dict:
                    first_image = next(iter(pixels_dict.values()))
                    batch_size = first_image.shape[0] if hasattr(first_image, "shape") else 1
                else:
                    batch_size = 1
            else:
                batch_size = 1

            # Create dummy agent_pos with appropriate shape
            dummy_agent_pos = np.zeros((batch_size, self.agent_pos_dim), dtype=np.float64)

            # Add to observation
            new_observation = observation.copy()
            new_observation["agent_pos"] = dummy_agent_pos

            # Update transition
            new_transition = transition.copy()
            new_transition[TransitionKey.OBSERVATION] = new_observation
            return new_transition

        return transition

    def get_config(self) -> dict[str, Any]:
        """
        Returns the serializable configuration of the processor.

        Returns:
            A dictionary containing the agent_pos_dim setting.
        """
        return {"agent_pos_dim": self.agent_pos_dim}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Adding agent_pos doesn't alter the fundamental feature definitions.

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        return features
