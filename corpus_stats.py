import argparse
import json
from collections import Counter


def analyze_jsonl(path):
    total_instances = 0
    total_groups = 0
    current_group = []

    unique_targets = set()
    group_sizes = []
    inconsistent_groups = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if current_group:
                    total_groups += 1
                    group_sizes.append(len(current_group))
                    targets = set(item["target"] for item in current_group)
                    if len(targets) > 1:
                        inconsistent_groups.append((total_groups, targets))
                    current_group = []
                continue

            obj = json.loads(line)
            total_instances += 1
            current_group.append(obj)
            unique_targets.add(obj["target"])

        if current_group:
            total_groups += 1
            group_sizes.append(len(current_group))
            targets = set(item["target"] for item in current_group)
            if len(targets) > 1:
                inconsistent_groups.append((total_groups, targets))

    avg_group_size = total_instances / total_groups if total_groups else 0
    size_distribution = Counter(group_sizes)

    print("===== DATASET STATS =====")
    print(f"Total instances: {total_instances}")
    print(f"Total groups (clusters): {total_groups}")
    print(f"Unique target values: {len(unique_targets)}")
    print(f"Average group size: {avg_group_size:.2f}")

    print("\n===== GROUP SIZE DISTRIBUTION =====")
    for size, count in sorted(size_distribution.items()):
        print(f"{size} instances: {count} groups")

    if inconsistent_groups:
        print("\n===== WARNING: INCONSISTENT GROUPS =====")
        for gid, targets in inconsistent_groups:
            print(f"Group {gid}: {targets}")
    else:
        print("\nAll groups are consistent (each group has one target).")


def main():
    parser = argparse.ArgumentParser(
        description="Count instances and groups, and flag groups with more than one target."
    )
    parser.add_argument("data", help="JSONL file; a blank line separates groups")
    args = parser.parse_args()
    analyze_jsonl(args.data)


if __name__ == "__main__":
    main()
