# Seq2Seq Mandarin-to-Cantonese Style Transfer: Colloquial Subtitle Rewriting in Hong Kong Urban Cinema

## Bachelor's Thesis by Hing Man Kobi Chiu

📄 **Full thesis (PDF):** [here](https://github.com/kobichiu/mandarin-to-cantonese-subtitle-mt/blob/main/BA_thesis_Hing-Man_Kobi_Chiu.pdf)

Hong Kong cinema has contributed many remarkable movies to the world. Even though its golden age in the 80s-90s passed, its subtitles are valuable data for Cantonese linguistics. As a fan, I was inspired to explore machine translation between Mandarin and Cantonese from the perspective of movie subtitles, under the umbrella of the Chinese language.

In this project, 4 Hong Kong movies produced between the 80s and 90s were selected to build a Mandarin–Cantonese parallel corpus in JSONL (`.jsonl`). Mandarin subtitles from OpenSubtitles were the starting point. Lines that were missing or did not match the spoken dialogue were added or revised by listening to the audio. Because Cantonese has no standardised written form, the corpus was built with heavy manual involvement, following a structured annotation guideline to keep the quality consistent. Because of copyright, the parallel corpus is not available in this repository. A short sample, `example.jsonl`, is provided instead so that the preprocessing steps can be run and inspected.

### Corpus Structure

Each line is one subtitle pair. `input` is the Mandarin subtitle in Simplified Chinese, and `target` is the spoken Cantonese line in Traditional Chinese. A group is the original pair plus Mandarin paraphrases of the same Cantonese line, so the Mandarin side varies while the Cantonese side stays identical. Lines in a group are consecutive, and a blank line separates groups.

How the parallel corpus looks like:
```
{"input":"明白了", "target":"知啦"} 
{"input":"知道了", "target":"知啦"}

{"input":"我帮帮你吧", "target":"我幫下你啦"} 
{"input":"我来帮你吧", "target":"我幫下你啦"}
```

Paraphrases were generated with DeepSeek-V3.2 and then checked by hand against the film context. After augmentation, the training set contains 9,948 instances from 5,974 groups. The dev and test sets keep only the original line of each group, 746 and 748 instances respectively, so no paraphrase reaches evaluation.

### Experiment Details & Result

`train.py` fine-tunes `google/mt5-small` for 5 epochs, with a batch size of 8, a learning rate of 2e-4, weight decay of 0.01, a maximum sequence length of 128 tokens, AdamW, beam search with 4 beams, and the best checkpoint selected on development loss.

Scores are computed with sacreBLEU using the `zh` tokenizer on the 748 test instances. The Google Translate baseline was collected separately through the web interface and is not produced by any script here.

| Systems                                    | BLEU  | chrF  | chrF++ |
|--------------------------------------------|-------|-------|--------|
| Google Translate baseline                  | 37.62 | 31.83 | 29.25  |
| mT5-small                                  | 28.66 | 24.12 | 19.13  |
| mT5-small (orthography normalisation only) | 29.20 | 24.51 | 19.43  |
| mT5-small (punctuation normalisation only) | 32.66 | 27.36 | 25.73  |
| mT5-small (both normalisations)            | 33.20 | 27.77 | 26.08  |

Google Translate still scores higher, but a large part of the gap is surface form. mT5-small often writes ASCII punctuation, such as `,` instead of `，`, and occasional Simplified characters, while the references use Chinese full-width punctuation and Traditional Chinese. Normalising both in the output raises BLEU from 28.66 to 33.20. The remaining gap is not only formatting: the model sometimes produces Traditional-character text that is still Mandarin in vocabulary and grammar, rather than colloquial Cantonese.

### Scripts

All scripts take file paths as command-line arguments and read the JSONL format described above.

| Script | What it does |
|---|---|
| `corpus_stats.py` | Counts instances, groups, unique Cantonese lines, and group sizes, and flags any group whose `target` is not identical on every line. |
| `check_target_consistency.py` | Prints the full contents of every group whose `target` differs between lines, so the mismatch can be corrected by hand. |
| `augmentation_stats.py` | Prints the groups that were never paraphrased and reports how many groups are augmented and how many variants each group has. |
| `simplify_mandarin.py` | Converts the `input` field from Traditional to Simplified Chinese with OpenCC, leaving `target` and the group separators untouched. |
| `split_corpus.py` | Splits the corpus 80/10/10 by group. Paraphrases stay in the training set; the dev and test files keep only the original line of each group. |
| `train.py` | Fine-tunes `google/mt5-small` on the training and dev files, then writes test-set predictions to `predictions.csv` and saves the model. |
| `evaluate.py` | Scores `predictions.csv` with BLEU, chrF, and chrF++, reporting each normalisation setting from the table above. |

```bash
pip install -r requirements.txt

# corpus checks
python corpus_stats.py example.jsonl
python check_target_consistency.py example.jsonl
python augmentation_stats.py example.jsonl
python simplify_mandarin.py example.jsonl example_simplified.jsonl

# training and evaluation
python split_corpus.py example.jsonl train.jsonl dev.jsonl test.jsonl
python train.py train.jsonl dev.jsonl --test test.jsonl
python evaluate.py predictions.csv
