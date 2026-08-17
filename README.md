# ChatgaiyyaAlap MT — Zero-Shot vs Few-Shot vs Dictionary-Augmented Prompting

**Standard Bangla ↔ Chittagonian (Chatgaiya) dialect translation**, evaluated across
three prompting strategies on an open-source LLM.

## TL;DR

Dictionary-augmented prompting wins clearly in both directions — roughly **9x**
zero-shot's BLEU on Bangla→Chatgaiya and **7x** on Chatgaiya→Bangla. Few-shot gives a
modest bump over zero-shot but nowhere near what explicit word-level grounding
provides. Full numbers and reasoning in [`outputs/day6_comparison.json`](outputs/day6_comparison.json).

| Direction | Technique | Corpus BLEU | Corpus chrF |
|---|---|---|---|
| B2C | zero-shot | 0.042 | 0.139 |
| B2C | few-shot | 0.065 | 0.221 |
| B2C | **dictionary** | **0.378** | **0.610** |
| C2B | zero-shot | 0.047 | 0.160 |
| C2B | few-shot | 0.097 | 0.258 |
| C2B | **dictionary** | **0.335** | **0.532** |

![comparison chart](outputs/day5_comparison_chart.png)

## Dataset

[**ChatgaiyyaAlap**](https://data.mendeley.com/datasets/wtms9xbkkw/1) — a parallel
corpus of Standard Bangla and Chittagonian (Chatgaiya) sentence pairs collected from
YouTube/Facebook posts, comments, videos, short films, and dramas, plus a ~1,500-word
Bangla↔Chatgaiya dictionary. Chatgaiya has no standardized written form, so spelling
variation in the source data is expected, not an error.

## Model

`Qwen/Qwen2.5-1.5B-Instruct` — small, open, ungated, runs on free-tier Colab (GPU or
CPU). Kept identical across all techniques and days for a fair comparison.

## Project structure

```
chatgaiyya-mt/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                              # raw dataset (gitignored — see setup below)
│   ├── sentence_pairs.csv
│   └── dictionary.csv
├── prompts/                           # versioned prompt templates
│   ├── zero_shot_b2c_v1.txt
│   ├── zero_shot_c2b_v1.txt
│   ├── few_shot_b2c_v1.txt
│   ├── few_shot_c2b_v1.txt
│   ├── dictionary_b2c_example_v1.txt  # rendered example (prompt is built per-sentence)
│   └── dictionary_c2b_example_v1.txt
├── logs/
│   └── issue_log.txt                  # hallucinations, failures, and observations by day
└── outputs/
    ├── <yourname>_week1day1_data.json # Day 1 — dataset summary
    ├── sampled_pairs.csv              # Day 1 — shared 50-80 sentence test sample
    ├── fewshot_pool.csv               # Day 1 — held-out pool for few-shot examples
    ├── day2_zeroshot_b2c.json         # Day 2 — zero-shot, Bangla → Chatgaiya
    ├── day2_zeroshot_c2b.json         # Day 2 — zero-shot, Chatgaiya → Bangla
    ├── day3_fewshot_b2c.json          # Day 3 — few-shot, Bangla → Chatgaiya
    ├── day3_fewshot_c2b.json          # Day 3 — few-shot, Chatgaiya → Bangla
    ├── day4_dictionary_variant.json   # Day 4 — dictionary-augmented, both directions
    ├── day5_eval.json                 # Day 5 — aggregated scoring + consistency check
    ├── day5_comparison_chart.png      # Day 5 — BLEU/chrF chart, all techniques
    └── day6_comparison.json           # Day 6 — final comparison + recommendation
```

## The three techniques

1. **Zero-shot** — instruction only, no examples. Baseline.
2. **Few-shot** — 4 examples (spanning short/medium/long sentences) shown in-context
   before the target sentence, drawn from a pool held out from the test set so there's
   no leakage.
3. **Dictionary-augmented** — per-sentence lookup against the ~1,500-word dictionary;
   only word pairs that actually appear in the sentence are injected into the prompt,
   not the whole dictionary.

All three ran on the **same fixed 60-sentence sample** (seed-controlled, sampled once
on Day 1) in **both directions**, so results are directly comparable across techniques
and across the team.

## Day-by-day

| Day | Task | Key output |
|---|---|---|
| 1 | Repo setup, load + inspect dataset, sample 50-80 pairs, model smoke test | `<name>_week1day1_data.json` |
| 2 | Zero-shot prompting, both directions | `day2_zeroshot_*.json` |
| 3 | Few-shot prompting, both directions | `day3_fewshot_*.json` |
| 4 | Dictionary-augmented prompting, both directions | `day4_dictionary_variant.json` |
| 5 | Aggregate BLEU/chrF across all techniques + run-to-run consistency check | `day5_eval.json` |
| 6 | Final comparison table + recommendation | `day6_comparison.json` |

## Setup

### 1. Get the dataset
Download both CSVs from Mendeley (no public API, manual download required):
https://data.mendeley.com/datasets/wtms9xbkkw/1
Save as `data/sentence_pairs.csv` and `data/dictionary.csv`. `data/` is gitignored —
don't commit the raw dataset.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
Each day's script/notebook reads the previous day's committed outputs from `outputs/`
rather than regenerating them — run in order, Day 1 through Day 6.

**Team rule:** everyone uses the same sampling `seed` and `n` (agree on these before
anyone runs Day 1) so every technique is scored against an identical sentence set.

## Methodology notes

- **Decoding:** greedy (`do_sample=False`) rather than temperature sampling, for
  reproducibility — confirmed in Day 5's consistency check (100% identical output
  across repeated runs under greedy decoding).
- **Output cleaning:** model output is stripped of stray commentary, quotes, and
  echoed labels before scoring, and normalized to NFC Unicode (Bengali/Chatgaiya
  script can otherwise encode visually identical text differently and silently fail
  to string-match).
- **Metrics:** corpus-level BLEU and chrF (via `sacrebleu`) are the primary reported
  numbers — sentence-level BLEU is too noisy on short sentences to trust as a headline
  metric, though sentence-level scores are kept per-example for spot-checking.

## Key findings

- **Dictionary augmentation is the clear winner**, by a wide margin, in both
  directions — explicit word-level grounding matters far more than in-context
  sentence examples for a dialect this under-resourced.
- **Few-shot helps, modestly** — roughly 1.5–2x zero-shot, well short of dictionary
  augmentation.
- Dictionary augmentation's advantage is **bounded by dictionary coverage** — a
  sentence with few or no matched words performs closer to zero-shot; this isn't a
  free win independent of the dictionary's completeness.
- Absolute scores remain moderate even for the best technique, consistent with this
  being a genuinely hard task for a small (1.5B) open model on a barely-resourced
  dialect — see `logs/issue_log.txt` for specific observed failure modes
  (hallucinations, defaulting to Standard Bangla, etc.).

Full detail, caveats, and the team's reasoning: [`outputs/day6_comparison.json`](outputs/day6_comparison.json).

## Team

Everyone should be able to explain the full pipeline end to end — dataset → sampling →
each technique → evaluation → consistency check → conclusion — not just the day they
personally ran.
