from data.loader import build_dataset_from_config


class GMAIMMBenchDataset:
    def __new__(
        cls,
        path: str = "GMAI-Lab/GMAI-MMBench",
        split: str = "test",
        num_samples=None,
        **_,
    ):
        return build_dataset_from_config(
            {
                "dataset": {
                    "name": "gmai_mmbench",
                    "type": "huggingface",
                    "path": path,
                    "split": split,
                },
                "task_type": "multi_choice",
                "max_samples": num_samples,
            }
        )


class PubMedVisionDataset:
    def __new__(
        cls,
        path: str = "FreedomIntelligence/PubMedVision",
        subset: str = "PubMedVision_Alignment",
        split: str = "train",
        num_samples=None,
        **_,
    ):
        return build_dataset_from_config(
            {
                "dataset": {
                    "name": "pubmedvision",
                    "type": "huggingface",
                    "path": path,
                    "subset": subset,
                    "split": split,
                },
                "fields": {
                    "image": "image",
                    "question": "question",
                    "answer": "answer",
                    "modality": "modality",
                },
                "task_type": "open_ended",
                "max_samples": num_samples,
            }
        )
