from torch.utils.data import ConcatDataset

from data.loader import build_dataset_from_config


class SLAKEDataset:
    def __new__(cls, path: str = "BoKelvin/SLAKE", split: str = "train", num_samples=None, **_):
        return build_dataset_from_config(
            {
                "dataset": {
                    "name": "slake",
                    "type": "huggingface",
                    "path": path,
                    "split": split,
                    "num_samples": num_samples,
                },
                "fields": {
                    "image": "img_name",
                    "question": "question",
                    "answer": "answer",
                },
                "task_type": "open_ended",
                "max_samples": num_samples,
            }
        )


class PathVQADataset:
    def __new__(
        cls,
        path: str = "flaviagiammarino/path-vqa",
        split: str = "train",
        num_samples=None,
        **_,
    ):
        return build_dataset_from_config(
            {
                "dataset": {
                    "name": "pathvqa",
                    "type": "huggingface",
                    "path": path,
                    "split": split,
                    "num_samples": num_samples,
                },
                "fields": {
                    "image": "image",
                    "question": "question",
                    "answer": "answer",
                },
                "task_type": "open_ended",
                "max_samples": num_samples,
            }
        )


class MixedRLDataset:
    def __new__(
        cls,
        omnimedvqa_samples: int = 3000,
        slake_samples: int = 500,
        pathvqa_samples: int = 500,
        **_,
    ):
        return ConcatDataset(
            [
                build_dataset_from_config(
                    {
                        "dataset": {
                            "name": "omnimedvqa",
                            "type": "huggingface",
                            "path": "OpenGVLab/OmniMedVQA",
                            "split": "train",
                        },
                        "fields": {
                            "image": "image",
                            "question": "question",
                            "answer": "answer",
                            "options": "options",
                        },
                        "task_type": "closed_ended",
                        "max_samples": omnimedvqa_samples,
                    }
                ),
                SLAKEDataset(num_samples=slake_samples),
                PathVQADataset(num_samples=pathvqa_samples),
            ]
        )
