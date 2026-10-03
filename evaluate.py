import argparse
import csv

import sacrebleu
from opencc import OpenCC

# mT5-small writes ASCII punctuation where the references use Chinese full-width
# forms. The code points differ, so the metrics count them as mismatches.
PUNCTUATION_MAP = {
    ",": "，",
    "!": "！",
    "?": "？",
}


def normalise_punctuation(text):
    for ascii_mark, fullwidth in PUNCTUATION_MAP.items():
        text = text.replace(ascii_mark, fullwidth)
    return text


def normalise_orthography(text, converter):
    return converter.convert(text)


def read_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [row["target"] for row in rows], [row["predicted"] for row in rows]


def score(references, hypotheses):
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="zh")
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    chrf_plus = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    return bleu.score, chrf.score, chrf_plus.score


def main():
    parser = argparse.ArgumentParser(
        description="Score predictions.csv with BLEU, chrF, and chrF++, with and without normalisation."
    )
    parser.add_argument("predictions", help="CSV with target and predicted columns")
    args = parser.parse_args()

    references, hypotheses = read_predictions(args.predictions)
    print(f"Scoring {len(references)} instances\n")

    converter = OpenCC("s2hk")
    settings = {
        "no normalisation": hypotheses,
        "orthography only": [normalise_orthography(h, converter) for h in hypotheses],
        "punctuation only": [normalise_punctuation(h) for h in hypotheses],
        "both": [
            normalise_punctuation(normalise_orthography(h, converter))
            for h in hypotheses
        ],
    }

    print(f"{'Setting':<20}{'BLEU':>8}{'chrF':>8}{'chrF++':>8}")
    for name, candidates in settings.items():
        bleu, chrf, chrf_plus = score(references, candidates)
        print(f"{name:<20}{bleu:>8.2f}{chrf:>8.2f}{chrf_plus:>8.2f}")


if __name__ == "__main__":
    main()
