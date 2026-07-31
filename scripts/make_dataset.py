# Med-EASi dataset citation: Basu, C., Vasu, R., Yasunaga, M., & Yang, Q. (2023). "Med-EASi: Finely Annotated Dataset and Models for Controllable Simplification of Medical Texts." arXiv:2302.09155. HuggingFace: https://huggingface.co/datasets/cbasu/Med-EASi

"""Build OpenAI fine-tuning JSONL files from the Med-EASi dataset."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import DatasetDict, load_dataset


DATASET_NAME = "cbasu/Med-EASi"
SYSTEM_PROMPT = "You rewrite medical text in plain language a patient can understand."
SEED = 42
TRAIN_SIZE = 250
TEST_SIZE = 20
OUTPUT_DIR = Path("data/processed")


def download_dataset(dataset_name: str = DATASET_NAME) -> DatasetDict:
    """Download the Med-EASi dataset from HuggingFace."""
    return load_dataset(dataset_name)


def clean_text(value: object) -> str:
    """Normalize a dataset field into a stripped string."""
    if value is None:
        return ""
    return str(value).strip()


def extract_pairs(dataset: DatasetDict) -> List[Tuple[str, str]]:
    """Extract non-empty expert-to-simple text pairs from every dataset split."""
    pairs = []
    seen = set()

    for split in dataset.values():
        for row in split:
            expert = clean_text(row.get("Expert"))
            simple = clean_text(row.get("Simple"))
            key = (expert, simple)

            if not expert or not simple or key in seen:
                continue

            seen.add(key)
            pairs.append(key)

    return pairs


def split_pairs(
    pairs: List[Tuple[str, str]],
    train_size: int = TRAIN_SIZE,
    test_size: int = TEST_SIZE,
    seed: int = SEED,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Shuffle pairs with a fixed seed, then return training and held-out test pairs."""
    required = train_size + test_size
    if len(pairs) < required:
        raise ValueError(f"Need at least {required} clean pairs, found {len(pairs)}.")

    shuffled = list(pairs)
    shuffled_dataset = load_dataset_from_pairs(shuffled).shuffle(seed=seed)
    sampled = [(row["expert"], row["simple"]) for row in shuffled_dataset.select(range(required))]

    train_pairs = sampled[:train_size]
    test_pairs = sampled[train_size:]
    return train_pairs, test_pairs


def load_dataset_from_pairs(pairs: List[Tuple[str, str]]):
    """Create an in-memory HuggingFace dataset from text pairs for deterministic shuffling."""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "expert": [expert for expert, _ in pairs],
            "simple": [simple for _, simple in pairs],
        }
    )


def to_chat_example(expert: str, simple: str) -> Dict[str, List[Dict[str, str]]]:
    """Convert a text pair to the OpenAI chat fine-tuning message format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": expert},
            {"role": "assistant", "content": simple},
        ]
    }


def to_test_example(expert: str, simple: str) -> Dict[str, str]:
    """Convert a held-out text pair to a compact JSONL test record."""
    return {"expert": expert, "simple": simple}


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    """Write records to a JSONL file with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dataset_files(output_dir: Path = OUTPUT_DIR) -> Tuple[Path, Path]:
    """Download, clean, sample, and write train and test JSONL files."""
    dataset = download_dataset()
    pairs = extract_pairs(dataset)
    train_pairs, test_pairs = split_pairs(pairs)

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"

    write_jsonl(train_path, (to_chat_example(expert, simple) for expert, simple in train_pairs))
    write_jsonl(test_path, (to_test_example(expert, simple) for expert, simple in test_pairs))

    return train_path, test_path


def main() -> None:
    """Run the dataset build and print output paths."""
    train_path, test_path = build_dataset_files()
    print(f"Wrote {train_path}")
    print(f"Wrote {test_path}")


if __name__ == "__main__":
    main()
