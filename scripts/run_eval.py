"""
Main evaluation script for SPOC code generation.

This script:
1. Loads a subset of test data
2. Generates C++ code using an LLM
3. Evaluates using BLEU, CompAcc, FCorrAcc, and Pass@1 metrics
4. Saves results to JSON
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Load .env if it exists (local only, not needed in Colab)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except Exception as e:
    print("No .env file found")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import SPOCDataLoader
from inference import CodeGenerator
from evaluator import CodeEvaluator

# from azure_storage import AzureStorage  # Disabled by default - use Modal volumes or local storage


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SPOC code generation")

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-0.5B",
        help="HuggingFace model name"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="testp",
        choices=["testp", "testw"],
        help="Test split to use"
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of programs to evaluate"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save results"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )

    parser.add_argument(
        "--use-public-tests",
        action="store_true",
        help="Use public test cases instead of hidden"
    )

    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit mode"
    )

    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference and load from existing results file"
    )

    parser.add_argument(
        "--results-file",
        type=str,
        help="Path to existing results file (for --skip-inference)"
    )

    parser.add_argument(
        "--run-name",
        type=str,
        help="Run name for consistent naming across Colab/local (uses timestamp if not provided)"
    )

    return parser.parse_args()


def main():
    """Run the evaluation pipeline."""
    args = parse_args()

    print("=" * 80)
    print("SPOC Code Generation Evaluation")
    print("=" * 80)
    print()
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"N samples: {args.n_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Using {'public' if args.use_public_tests else 'hidden'} test cases")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run identifier: use run-name if provided, otherwise timestamp
    run_id = args.run_name if args.run_name else datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1]

    # Azure storage (disabled by default - use Modal volumes or local storage)
    # try:
    #     azure = AzureStorage(container="spoc")
    #     print(f"✓ Connected to Azure storage")
    # except:
    #     azure = None
    #     print("⚠ Azure not configured (local only)")
    azure = None

    if args.run_name:
        print(f"Run name: {args.run_name}")
    print()

    # %%
    # Load data
    print("-" * 80)
    print("Loading data...")
    print("-" * 80)

    loader = SPOCDataLoader(data_dir="..")
    programs = loader.load_test_split(args.split)

    print(f"✓ Loaded {len(programs)} programs from {args.split}")

    # Get subset
    subset = loader.get_subset(programs, n=args.n_samples)
    print(f"✓ Using {len(subset)} programs for evaluation")
    print()

    # %%
    # Generate code
    if args.skip_inference:
        print("-" * 80)
        print("Loading existing results...")
        print("-" * 80)

        # If no results file specified, error
        if not args.results_file:
            # if azure and args.run_name:
            #     gen_file = f"generations_{model_short}_{args.split}_n{args.n_samples}_{run_id}.json"
            #     azure_path = f"generations/{model_short}/{args.split}/{gen_file}"
            #     local_path = output_dir / gen_file
            #
            #     print(f"Downloading from Azure: {azure_path}")
            #     azure.load_file(azure_path, str(local_path))
            #     args.results_file = str(local_path)
            # else:
            print("Error: --results-file required when using --skip-inference")
            return

        with open(args.results_file, 'r') as f:
            results = json.load(f)

        print(f"✓ Loaded {len(results)} results from {args.results_file}")
        print()

    else:
        print("-" * 80)
        print("Generating code...")
        print("-" * 80)

        generator = CodeGenerator(
            model_name=args.model,
            load_in_8bit=args.load_in_8bit
        )

        results = generator.generate_batch(
            subset,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=True if args.temperature > 0 else False,
            top_p=0.95
        )

        # Save generation results locally
        gen_file = f"generations_{model_short}_{args.split}_n{args.n_samples}_{run_id}.json"
        gen_output_file = output_dir / gen_file
        generator.save_results(results, gen_output_file)

        # if azure:
        #     azure_path = f"generations/{model_short}/{args.split}/{gen_file}"
        #     azure.save_file(str(gen_output_file), azure_path)
        #     print(f"✓ Uploaded to Azure: {azure_path}")
        print()

    # %%
    # Evaluate
    print("-" * 80)
    print("Evaluating...")
    print("-" * 80)

    evaluator = CodeEvaluator(loader)

    # Compute BLEU
    print("Computing BLEU scores...")
    bleu_metrics = evaluator.compute_bleu(results)
    print(f"✓ BLEU: {bleu_metrics['bleu']:.2f}")
    print()

    # Compile and test
    print("Compiling and testing programs...")
    results = evaluator.evaluate_batch(
        results,
        use_hidden=not args.use_public_tests
    )

    # Compute metrics
    metrics = evaluator.compute_metrics(results)

    # Print summary
    evaluator.print_metrics_summary(metrics, bleu_metrics)

    # %%
    # Save full results
    eval_file = f"evaluation_{model_short}_{args.split}_n{args.n_samples}_{run_id}.json"
    eval_output_file = output_dir / eval_file

    full_results = {
        "config": {
            "model": args.model,
            "split": args.split,
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "use_hidden_tests": not args.use_public_tests,
            "run_id": run_id,
        },
        "metrics": {
            **metrics,
            **bleu_metrics
        },
        "results": results
    }

    with open(eval_output_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"✓ Saved locally: {eval_output_file}")

    # if azure:
    #     azure_path = f"evaluations/{model_short}/{args.split}/{eval_file}"
    #     azure.save_file(str(eval_output_file), azure_path)
    #     print(f"✓ Uploaded to Azure: {azure_path}")
    print()

    # Save metrics summary
    metrics_file = f"metrics_{model_short}_{args.split}_n{args.n_samples}_{run_id}.txt"
    summary_file = output_dir / metrics_file
    with open(summary_file, 'w') as f:
        f.write(f"SPOC Evaluation Summary\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"N samples: {args.n_samples}\n")
        f.write(f"Temperature: {args.temperature}\n\n")
        f.write(f"BLEU: {bleu_metrics['bleu']:.2f}\n")
        f.write(f"CompAcc: {metrics['comp_acc']:.2f}%\n")
        f.write(f"FCorrAcc: {metrics['fcorr_acc']:.2f}%\n")
        f.write(f"Pass@1: {metrics['pass_at_1']:.2f}%\n")

    print(f"✓ Saved locally: {summary_file}")

    # if azure:
    #     azure_path = f"metrics/{model_short}/{args.split}/{metrics_file}"
    #     azure.save_file(str(summary_file), azure_path)
    #     print(f"✓ Uploaded to Azure: {azure_path}")
    print()
    print("=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
