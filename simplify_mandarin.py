import argparse
import json

from opencc import OpenCC


def convert_input_to_simplified(input_path, output_path):
    cc = OpenCC("t2s")

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            stripped = line.strip()

            if not stripped:
                fout.write("\n")
                continue

            obj = json.loads(stripped)
            if "input" in obj:
                obj["input"] = cc.convert(obj["input"])

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Conversion finished.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert the Mandarin input field from Traditional to Simplified Chinese."
    )
    parser.add_argument("input", help="Input JSONL; a blank line separates groups")
    parser.add_argument("output", help="Output JSONL")
    args = parser.parse_args()
    convert_input_to_simplified(args.input, args.output)


if __name__ == "__main__":
    main()
