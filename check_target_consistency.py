import argparse
import json


def validate_targets(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        content = f.read()

    groups = content.strip().split("\n\n")
    mismatch_count = 0

    for group_idx, group in enumerate(groups, start=1):
        lines = [line.strip() for line in group.split("\n") if line.strip()]
        targets = []
        records = []

        for line in lines:
            try:
                obj = json.loads(line)
                targets.append(obj.get("target"))
                records.append(obj)
            except json.JSONDecodeError:
                print(f"[Group {group_idx}] Invalid JSON: {line}")

        if len(set(targets)) > 1:
            mismatch_count += 1
            print(f"\nMismatch in group {group_idx}:")
            for record in records:
                print(record)

    print(f"\nDone. {mismatch_count} mismatched groups found.")


def main():
    parser = argparse.ArgumentParser(
        description="Print groups whose target field is not identical on every line."
    )
    parser.add_argument("data", help="JSONL file; a blank line separates groups")
    args = parser.parse_args()
    validate_targets(args.data)


if __name__ == "__main__":
    main()
