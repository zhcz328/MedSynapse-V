from data.loader import build_dataset_from_config


class OmniMedVQADataset:
    def __new__(
        cls,
        path: str = "OpenGVLab/OmniMedVQA",
        split: str = "train",
        num_samples=None,
        fields=None,
        **_,
    ):
        return build_dataset_from_config(
            {
                "dataset": {
                    "name": "omnimedvqa",
                    "type": "huggingface",
                    "path": path,
                    "split": split,
                    "num_samples": num_samples,
                },
                "fields": fields
                or {
                    "image": "image",
                    "question": "question",
                    "answer": "answer",
                    "options": "options",
                },
                "task_type": "closed_ended",
                "max_samples": num_samples,
            }
        )
