import argparse
from collections import Counter


def read_groups(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    return [group.strip().split("\n") for group in content.split("\n\n") if group.strip()]


def print_unaugmented(groups):
    print("---- Unaugmented lines ----\n")
    count = 0

    for group in groups:
        if len(group) == 1:
            print(group[0])
            count += 1

    return count


def augmentation_stats(groups):
    counts = Counter(len(group) for group in groups)

    print("\n---- Variant distribution ----")
    for size in sorted(counts):
        print(f"{size} variant(s): {counts[size]}")


def main():
    parser = argparse.ArgumentParser(
        description="Count augmented and unaugmented subtitle groups."
    )
    parser.add_argument("data", help="JSONL file; a blank line separates groups")
    args = parser.parse_args()

    groups = read_groups(args.data)
    total_groups = len(groups)
    unaugmented = print_unaugmented(groups)

    print("\n---- Statistics ----")
    print("Total subtitle groups:", total_groups)
    print("Unaugmented groups:", unaugmented)
    print("Augmented groups:", total_groups - unaugmented)
    if total_groups:
        print(
            "Average variants per group:",
            sum(len(group) for group in groups) / total_groups,
        )

    augmentation_stats(groups)


if __name__ == "__main__":
    main()
