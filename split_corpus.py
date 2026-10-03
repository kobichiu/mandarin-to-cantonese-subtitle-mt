import argparse
import random

SEED = 42


def read_groups(path):
    """A group is one original pair plus its paraphrases, separated by a blank line."""
    groups = []
    current = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    groups.append(current)
                    current = []
                continue
            current.append(line)

    if current:
        groups.append(current)

    return groups


def write_groups(groups, path):
    """Keep every variant, one group per blank-line-separated block."""
    with open(path, "w", encoding="utf-8") as f:
        for group in groups:
            for line in group:
                f.write(line + "\n")
            f.write("\n")


def write_originals(groups, path):
    """Keep only the first line of each group, so paraphrases never reach evaluation."""
    ditched = 0
    with open(path, "w", encoding="utf-8") as f:
        for group in groups:
            f.write(group[0] + "\n\n")
            ditched += len(group) - 1
    return ditched


def split_corpus(input_path, train_path, dev_path, test_path, seed=SEED):
    groups = read_groups(input_path)
    print(f"Total groups: {len(groups)}")

    random.seed(seed)
    random.shuffle(groups)

    n_train = int(0.8 * len(groups))
    n_dev = int(0.1 * len(groups))
    train_groups = groups[:n_train]
    dev_groups = groups[n_train : n_train + n_dev]
    test_groups = groups[n_train + n_dev :]

    write_groups(train_groups, train_path)
    dev_ditched = write_originals(dev_groups, dev_path)
    test_ditched = write_originals(test_groups, test_path)

    train_lines = sum(len(group) for group in train_groups)
    print(f"Train groups: {len(train_groups)} | instances: {train_lines}")
    print(f"Dev groups:   {len(dev_groups)} | instances: {len(dev_groups)} | dropped {dev_ditched}")
    print(f"Test groups:  {len(test_groups)} | instances: {len(test_groups)} | dropped {test_ditched}")


def main():
    parser = argparse.ArgumentParser(
        description="Split the corpus 80/10/10 by group, keeping paraphrases in train only."
    )
    parser.add_argument("data", help="JSONL file; a blank line separates groups")
    parser.add_argument("train", help="Output path for the training set")
    parser.add_argument("dev", help="Output path for the development set")
    parser.add_argument("test", help="Output path for the test set")
    args = parser.parse_args()

    split_corpus(args.data, args.train, args.dev, args.test)


if __name__ == "__main__":
    main()
