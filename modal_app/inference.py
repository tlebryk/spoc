"""
Modal-based inference for SPOC code generation.

This module provides GPU-accelerated code generation using Modal.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import modal

# Create Modal app
app = modal.App("spoc-evaluation")

# Define container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "g++",  # C++ compiler for evaluation
        "build-essential",  # Build tools
    )
    .pip_install(
        "torch==2.1.0",
        "transformers==4.45.0",  # Updated for Qwen2 support
        "accelerate==0.26.0",  # Required for device_map
        "sacrebleu==2.3.1",
        "pandas==2.1.4",
        "tqdm==4.66.1",
    )
)

# Persistent volume for datasets and artifacts
volume = modal.Volume.from_name("spoc-artifacts", create_if_missing=True)

# Path where volume will be mounted in containers
VOLUME_PATH = "/artifacts"

# Dataset paths within the volume
DATASET_PATH = f"{VOLUME_PATH}/datasets"
RESULTS_PATH = f"{VOLUME_PATH}/results"
GENERATIONS_PATH = f"{VOLUME_PATH}/generations"
EVALUATIONS_PATH = f"{VOLUME_PATH}/evaluations"
METRICS_PATH = f"{VOLUME_PATH}/metrics"

# Import helper class for data loading (we'll include minimal version here)
class Program:
    """Represents a single program from SPOC dataset."""

    def __init__(self, lines):
        self.lines = lines
        self.probid = lines[0]["probid"]
        self.subid = lines[0]["subid"]
        self.workerid = lines[0]["workerid"]

    def get_program_id(self):
        return f"{self.probid}_{self.subid}_{self.workerid}"

    def get_full_pseudocode(self):
        """Get complete pseudocode for the program."""
        pseudocode_lines = []
        for line in self.lines:
            text = line.get("text", "")
            if text and str(text) != "nan":
                pseudocode_lines.append(text)
        return "\n".join(pseudocode_lines)

    def get_full_code(self):
        """Get complete gold C++ code."""
        code_lines = []
        for line in self.lines:
            code = line.get("code", "")
            if code:
                code_lines.append(code)
        return "\n".join(code_lines)


@app.function(
    image=image,
    gpu="T4",  # Change to "A10G" for faster inference
    timeout=3600,  # 1 hour timeout
    volumes={VOLUME_PATH: volume},
    # Uncomment if you need HuggingFace authentication for private models:
    # secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_code_batch(
    programs_data: List[Dict],
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> List[Dict]:
    """
    Generate C++ code for a batch of programs.

    Args:
        programs_data: List of program dictionaries (serialized Program objects)
        model_name: HuggingFace model identifier
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        List of result dictionaries with generated code
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm

    print(f"Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if device != "cuda":
        model = model.to(device)

    model.eval()
    print(f"Model loaded successfully on {device}")

    # Helper function to create prompt
    def create_prompt(pseudocode: str) -> str:
        return f"""You are an expert C++ programmer. Convert the following pseudocode into complete, working C++ code.

Pseudocode:
{pseudocode}

Generate only the C++ code without any explanations. The code should be syntactically correct and ready to compile."""

    # Helper function to extract code
    def extract_code(generated_text: str) -> str:
        # Look for code blocks
        if "```cpp" in generated_text:
            parts = generated_text.split("```cpp")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()
        elif "```c++" in generated_text:
            parts = generated_text.split("```c++")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()
        elif "```" in generated_text:
            parts = generated_text.split("```")
            if len(parts) >= 3:
                code = parts[1]
                lines = code.split('\n')
                if lines and lines[0].strip() in ['cpp', 'c++', 'c']:
                    code = '\n'.join(lines[1:])
                return code.strip()
        return generated_text.strip()

    # Generate code for each program
    results = []
    print(f"Generating code for {len(programs_data)} programs...")

    for prog_data in tqdm(programs_data, desc="Generating"):
        # Reconstruct Program object
        program = Program(prog_data["lines"])
        pseudocode = program.get_full_pseudocode()
        gold_code = program.get_full_code()

        try:
            # Create prompt
            prompt = create_prompt(pseudocode)

            # Create messages for chat models
            messages = [
                {"role": "system", "content": "You are an expert C++ programmer."},
                {"role": "user", "content": prompt}
            ]

            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt").to(device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the generated part
            generated_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            generated_code = extract_code(generated_text)

            result = {
                "program_id": program.get_program_id(),
                "probid": program.probid,
                "subid": program.subid,
                "workerid": program.workerid,
                "pseudocode": pseudocode,
                "gold_code": gold_code,
                "generated_code": generated_code,
                "success": True,
                "error": None
            }
        except Exception as e:
            result = {
                "program_id": program.get_program_id(),
                "probid": program.probid,
                "subid": program.subid,
                "workerid": program.workerid,
                "pseudocode": pseudocode,
                "gold_code": gold_code,
                "generated_code": "",
                "success": False,
                "error": str(e)
            }

        results.append(result)

    success_count = sum(1 for r in results if r["success"])
    print(f"Successfully generated {success_count}/{len(results)} programs")

    return results


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-Coder-0.5B",
    split: str = "testp",
    n_samples: int = 100,
    temperature: float = 0.2,
    max_tokens: int = 512,
    run_name: str = None,
):
    """
    Main entrypoint for running inference on Modal.

    Usage:
        modal run modal_app/inference.py --model Qwen/Qwen2.5-Coder-0.5B --n-samples 50
    """
    import pandas as pd

    print("=" * 80)
    print("SPOC Code Generation on Modal")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Split: {split}")
    print(f"N samples: {n_samples}")
    print(f"Temperature: {temperature}")
    print()

    # Generate run ID
    run_id = run_name if run_name else datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model.split("/")[-1]

    # Load dataset from volume
    print("Loading dataset from Modal volume...")
    dataset_file = f"{DATASET_PATH}/test/spoc-{split}.tsv"

    # Read the TSV file from volume
    with volume.batch_upload() as batch:
        pass  # Just to ensure volume is mounted

    # For now, we'll need to load data locally and pass it
    # In a production setup, you'd want to ensure data is in the volume first
    print("Note: Make sure you've uploaded data using upload_data.py first!")
    print()

    # Load data locally (temporary - assumes you have data locally)
    local_data_path = Path(__file__).parent.parent / "test" / f"spoc-{split}.tsv"

    if not local_data_path.exists():
        print(f"Error: {local_data_path} not found.")
        print("Please run 'modal run modal_app/upload_data.py' first to upload datasets.")
        return

    df = pd.read_csv(local_data_path, sep='\t')

    # Group by program
    programs = []
    current_lines = []

    for _, row in df.iterrows():
        if row['line'] == 0 and current_lines:
            programs.append(current_lines)
            current_lines = []
        current_lines.append(row.to_dict())

    if current_lines:
        programs.append(current_lines)

    print(f"Loaded {len(programs)} programs from {split}")

    # Sample subset
    programs_subset = programs[:n_samples]
    print(f"Using {len(programs_subset)} programs for generation")
    print()

    # Prepare data for Modal function (serialize Program objects as dicts)
    programs_data = [{"lines": prog} for prog in programs_subset]

    # Run generation on Modal
    print("Starting generation on Modal GPU...")
    results = generate_code_batch.remote(
        programs_data=programs_data,
        model_name=model,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    print(f"\nGeneration complete! Got {len(results)} results")

    # Save results locally first
    output_filename = f"generations_{model_short}_{split}_n{n_samples}_{run_id}.json"
    local_output_dir = Path(__file__).parent.parent / "outputs"
    local_output_dir.mkdir(exist_ok=True)
    local_output_path = local_output_dir / output_filename

    with open(local_output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Upload to volume
    output_path = f"{GENERATIONS_PATH}/{model_short}/{split}/{output_filename}"
    print(f"Saving results to volume: {output_path}")

    with volume.batch_upload() as batch:
        batch.put_file(str(local_output_path), output_path)

    print(f"✓ Saved to volume: {output_path}")
    print(f"✓ Saved locally: {local_output_path}")
    print()
    print("=" * 80)
    print("Generation complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"1. Download results: modal volume get spoc-artifacts {output_path} ./outputs/")
    print(f"2. Run evaluation locally: python scripts/run_eval.py --skip-inference --results-file outputs/{output_filename}")
