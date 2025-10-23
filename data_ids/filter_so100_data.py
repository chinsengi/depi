"""
This script is used to aggregate so100 data from the huggingface.
"""

import json
import math

from datasets import concatenate_datasets, load_dataset, load_dataset_builder
from huggingface_hub import HfApi
from tqdm import tqdm

users = [
    "pierfabre",
    "Odog16",
    "Mwuqiu",
    "lerobot",
    "lirislab",
    "maximilienroberti",
]


dataset_id = [
    "youliangtan/so100_strawberry_grape",
    "cyoung96/so100_test",
    "Mwuqiu/pickup_ducks",
]


def main():
    api = HfApi()
    # list_datasets supports a `tags` parameter to filter by one or more tags
    datasets = api.list_datasets(tags="so100")
    dataset_list = []
    for dataset in datasets:
        print(f"processing dataset: {dataset.id}")
        dataset_id = dataset.id
        dataset_list.append(load_dataset(dataset_id)["train"])
    dataset = concatenate_datasets(dataset_list)
    dataset.push_to_hub("sengi/so100_pretraining")


def check_nan(sample):
    if isinstance(sample, dict):
        for key, value in sample.items():
            if isinstance(value, list) and any(not check_nan(x) for x in value):
                print(f"value error in {key=}")
                return False
    elif isinstance(sample, list):
        if any(not check_nan(x) for x in sample):
            return False
    else:
        if sample is None or not isinstance(sample, (float, int)) or math.isnan(sample):
            print(f"value error in {sample=}")
            return False
    return True

def get_repo_ids(
    tag: str = "so100",
    len_limit: int = None,
    load_from_cache: bool = True,
    dataset_list_path: str = "data_ids/dataset_list_valid.json",
) -> list[str]:
    """Get the repo ids of all single arm so-100 datasets for a given tag."""
    # check if the feature of the dataset contains required_features
    dataset_list = []
    if load_from_cache:
        with open(dataset_list_path) as f:
            dataset_list = json.load(f)
        print(f"Loaded {len(dataset_list)} datasets from cache")
        if len_limit is not None and len(dataset_list) >= len_limit:
            return dataset_list[:len_limit]
        return dataset_list
    dataset_list = set(dataset_list)
    required_features = [
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    api = HfApi()
    # list_datasets supports a `tags` parameter to filter by one or more tags
    datasets = api.list_datasets(tags=tag)
    datasets = list(datasets)
    print(f"Found {len(datasets)} datasets with tag so100")
    for dataset in tqdm(datasets, desc="Processing datasets"):
        if len_limit is not None and len(dataset_list) >= len_limit:
            break
        if dataset.id.startswith("Vacuame") or dataset.id in dataset_list:
            continue
        try:
            builder = load_dataset_builder(dataset.id)
            features = builder.info.features
            if features is None:
                continue
            feature_present_flag = set(required_features).issubset(features)
            single_arm_flag = (
                "action" in features
                and features.get("action").length == 6
                and features.get("observation.state").length == 6
            )
            if feature_present_flag and single_arm_flag:
                dataset_list.add(dataset.id)
                print(f"Found {len(dataset_list)} valid datasets")
        except Exception as e:
            print(f"Error processing dataset {dataset.id}: {e}")
            continue
    print(f"Found {len(dataset_list)} valid datasets")
    return dataset_list

if __name__ == "__main__":
    dataset_list_existing = get_repo_ids(tag="so100", load_from_cache=True)
    with open("bad_dataset_ids.json") as f:
        bad_dataset_ids = json.load(f)
    print(f"{len(dataset_list_existing)=}")
    for i in tqdm(range(len(dataset_list_existing))):
        dataset_id = dataset_list_existing[i]
        print(f"examining {dataset_id=}")
        try:
            ds = load_dataset(dataset_id)["train"]
            sample = ds[0]
            if not check_nan(sample):
                bad_dataset_ids.append(dataset_id)
                print(f"bad {dataset_id=}")
                continue
            features = ds.features
            if features["action"].length != 6 or features["observation.state"].length != 6:
                bad_dataset_ids.append(dataset_id)
                print(f"bad {dataset_id=}")
        except Exception as e:
            print(f"bad {dataset_id=}")
            print(f"{e=}")
            bad_dataset_ids.append(dataset_id)
    dataset_list_existing = list(set(dataset_list_existing) - set(bad_dataset_ids))
    bad_dataset_ids = list(bad_dataset_ids)
    print(f"{len(bad_dataset_ids)=}")
    print(f"{len(dataset_list_existing)=}")
    
    # save the dataset_list to a file
    with open("dataset_list_valid.json", "w") as f:
        json.dump(dataset_list_existing, f)
    with open("bad_dataset_ids.json", "w") as f:
        json.dump(bad_dataset_ids, f)
