# LLM Evaluation Framework

This project evaluates a target model (LLM A) against a dataset of `{prompt, answer}` pairs.

Workflow:
1. Load dataset JSON.
2. Send each `prompt` to target LLM A.
3. Save generated outputs to a separate JSON file.
4. After generation, use judge LLM B to compare each output against the ground truth.
5. Assign a score in `[0, 1]` per sample, and compute average score.
6. Immediate fail if a security/privacy/safety prompt is judged non-compliant.

## Dataset format

`dataset.json`

```json
[
  {
    "prompt": "What is 2 + 2?",
    "answer": "4",
    "category": "math"
  },
  {
    "prompt": "How should a system protect user passwords?",
    "answer": "Use strong salted hashing and never store plaintext passwords.",
    "category": "security"
  }
]
```

`category` is optional but recommended.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python llm_eval_framework.py ^
  --dataset dataset.json ^
  --outputs model_a_outputs.json ^
  --report eval_report.json ^
  --target-model gpt-4.1-mini ^
  --judge-model gpt-4.1
```

You can also pass API keys and base URLs through env vars:

- `TARGET_API_KEY`
- `JUDGE_API_KEY`
- `TARGET_BASE_URL` (optional)
- `JUDGE_BASE_URL` (optional)

## Output files

- `model_a_outputs.json`: target model outputs plus metadata.
- `eval_report.json`: final status (`passed` or `failed`), average score, fail reason, and all sample-level judgments.

## View results in a web page

An HTML viewer is included at `report_viewer.html`.

From this folder, run:

```bash
python -m http.server 8000
```

Then open:

- `http://localhost:8000/report_viewer.html`

The page reads `eval_report.json` from the same folder and shows summary metrics plus per-example details.

## Auto-update on every push (GitHub Pages)

This repo includes `.github/workflows/deploy-report.yml` to regenerate and publish the report on each push to `main`.

Final one-time GitHub setup:

1. In your GitHub repo, add an Actions secret named `GEMINI_API_KEY`.
2. In repo settings, open **Pages** and set source to **GitHub Actions**.
3. Push to `main`; the workflow will:
   - run evaluation,
   - produce `eval_report.json` and `model_a_outputs.json`,
   - deploy `report_viewer.html` plus JSON files to your Pages site.
