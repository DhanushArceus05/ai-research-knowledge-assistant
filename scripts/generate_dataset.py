"""
Optionally expands the base labelled dataset into a larger synthetic dataset by
recombining topic sentences with simple templates. This is meant purely to give
the classifier more training rows for a more stable demo; it is NOT required to
run the project (a usable dataset is already committed at
data/dataset/document_classification.csv).

Usage:
    python scripts/generate_dataset.py --multiplier 4
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

TEMPLATES = [
    "{sentence}",
    "In this paper, we discuss how {sentence_lower}",
    "Recent research shows that {sentence_lower}",
    "According to the study, {sentence_lower}",
    "This section explains that {sentence_lower}",
]


def expand_dataset(input_csv: Path, output_csv: Path, multiplier: int) -> int:
    rows = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["text"].strip(), row["category"].strip()))

    expanded = []
    for text, category in rows:
        for i in range(multiplier):
            template = TEMPLATES[i % len(TEMPLATES)]
            sentence_lower = text[0].lower() + text[1:] if text else text
            new_text = template.format(sentence=text, sentence_lower=sentence_lower)
            expanded.append((new_text, category))

    random.shuffle(expanded)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "category"])
        writer.writerows(expanded)

    return len(expanded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a larger synthetic labelled dataset.")
    parser.add_argument("--input", default="data/dataset/document_classification.csv")
    parser.add_argument("--output", default="data/dataset/document_classification_synthetic.csv")
    parser.add_argument("--multiplier", type=int, default=4, help="How many variants to generate per source row.")
    args = parser.parse_args()

    count = expand_dataset(Path(args.input), Path(args.output), args.multiplier)
    print(f"Generated {count} synthetic rows -> {args.output}")
