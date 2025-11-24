# SPOC Evaluation Pipeline

Quick reference for the pseudocode → C++ code generation evaluation system.

## TL;DR

**Local (CPU):**
```bash
conda activate spoc
cd scripts
python run_eval.py --n-samples 50  # Generates code + evaluates
```

**Colab (GPU) → Local (CPU):**
1. Open `notebooks/colab_generate.ipynb` in Colab
2. Run cells to generate code on GPU
3. Download `outputs/generations_colab.json`
4. Locally: `python run_eval.py --skip-inference --results-file generations_colab.json`

## What This Does

Evaluates LLM code generation on SPOC dataset:
1. Load pseudocode from SPOC dataset
2. Generate C++ code using **Qwen2.5-Coder-1.5B-Instruct**
3. Compile with g++
4. Run against test cases
5. Compute metrics

## Metrics Explained

| Metric | Definition |
|--------|------------|
| **CompAcc** | % of generated programs that compile |
| **FCorrAcc** | % that compile AND pass all test cases |
| **Pass@1** | % of problems with ≥1 correct solution |
| **BLEU** | Lexical similarity to gold code |

Reference: Based on [RLSF paper](https://arxiv.org/abs/2405.16661)

## File Structure

```
src/
  data_loader.py    # Parses TSV → Program objects
  inference.py      # LLM code generation
  evaluator.py      # Compile + test + metrics

scripts/
  eda.py           # Quick dataset preview
  run_eval.py      # Main evaluation script

outputs/          # Results saved here
```

## Usage

### 1. Explore Dataset
```bash
cd scripts
python eda.py
```
Shows: sample programs, statistics, test case format

### 2. Run Evaluation
```bash
# Basic (50 samples from testp)
python run_eval.py --n-samples 50

# More samples
python run_eval.py --n-samples 200

# Different test split (new workers vs new problems)
python run_eval.py --split testw

# Different model
python run_eval.py --model Qwen/Qwen2.5-Coder-7B-Instruct

# Adjust generation (higher = more creative)
python run_eval.py --temperature 0.8

# Save memory
python run_eval.py --load-in-8bit
```

### 3. Check Results
```bash
cd ../outputs
cat metrics_summary_*.txt  # Quick overview
```

## Output Files

- `evaluation_{timestamp}.json` - Full results: all programs, generations, test outcomes
- `metrics_summary_{timestamp}.txt` - Just the metrics
- `generations_{timestamp}.json` - Only the generated code (no eval)

## Dataset Splits

| Split | Description | Size |
|-------|-------------|------|
| `testp` | New problems (unseen in training) | ~1,000 programs |
| `testw` | New workers (unseen annotators) | ~700 programs |

Use `--split` to choose which one.

## Common Options

```bash
--n-samples 100              # Number of programs to evaluate
--split testp                # Test split: testp or testw
--temperature 0.2            # Generation temperature (0.0-1.0)
--max-tokens 512             # Max tokens to generate
--use-public-tests           # Use public test cases (default: hidden)
--load-in-8bit               # 8-bit quantization for less memory
--skip-inference             # Skip generation, just re-evaluate
--results-file path.json     # Use with --skip-inference
```

## Quick Debugging

If evaluation fails:
- Check `g++` is installed: `g++ --version`
- Try public tests first: `--use-public-tests`
- Start with fewer samples: `--n-samples 10`
- Check output JSON for specific errors

## Dependencies

```
transformers torch pandas sacrebleu tqdm accelerate
```

Install via conda or pip in the `spoc` environment.

## Git Setup (for Colab)

```bash
# Local: push code to git
git add src/ scripts/ notebooks/ EVALUATION.md
git commit -m "Add SPOC evaluation pipeline"
git push

# Colab: clone and use
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO/spoc
# Then run notebook cells
```

Data not in git - download separately in Colab if needed.

## Notes

- **Hidden test cases** by default (more realistic evaluation)
- **Empty pseudocode lines** (e.g., `}`, `return 0;`) are structural - model generates these
- Requires **g++ compiler** for evaluation
- Temperature 0.2 = more deterministic, 0.8 = more creative
- Results timestamped to avoid overwrites
