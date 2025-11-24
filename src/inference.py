"""
Inference pipeline for generating C++ code from pseudocode using LLMs.

This module handles loading models, creating prompts, and generating code.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
from tqdm import tqdm
import json
from pathlib import Path

from data_loader import Program


class CodeGenerator:
    """Generate C++ code from pseudocode using an LLM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        device: Optional[str] = None,
        load_in_8bit: bool = False
    ):
        """Initialize the code generator.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on (None for auto)
            load_in_8bit: Whether to load model in 8-bit mode for memory efficiency
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        if not load_in_8bit and self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()

        print("✓ Model loaded successfully")

    def create_prompt(self, pseudocode: str) -> str:
        """Create a prompt for code generation.

        Args:
            pseudocode: Line-by-line pseudocode description

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert C++ programmer. Convert the following pseudocode into complete, working C++ code.

Pseudocode:
{pseudocode}

Generate only the C++ code without any explanations. The code should be syntactically correct and ready to compile."""

        return prompt

    def generate_code(
        self,
        pseudocode: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        do_sample: bool = True,
        top_p: float = 0.95,
    ) -> str:
        """Generate C++ code from pseudocode.

        Args:
            pseudocode: Pseudocode description
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            do_sample: Whether to use sampling
            top_p: Nucleus sampling parameter

        Returns:
            Generated C++ code
        """
        prompt = self.create_prompt(pseudocode)

        # Create messages for chat models
        messages = [
            {"role": "system", "content": "You are an expert C++ programmer."},
            {"role": "user", "content": prompt}
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return self._extract_code(generated_text)

    def _extract_code(self, generated_text: str) -> str:
        """Extract C++ code from generated text.

        Sometimes models wrap code in markdown blocks or add explanations.
        This tries to extract just the code.

        Args:
            generated_text: Raw generated text

        Returns:
            Extracted code
        """
        # Look for code blocks
        if "```cpp" in generated_text:
            # Extract from cpp code block
            parts = generated_text.split("```cpp")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()
        elif "```c++" in generated_text:
            # Extract from c++ code block
            parts = generated_text.split("```c++")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()
        elif "```" in generated_text:
            # Extract from generic code block
            parts = generated_text.split("```")
            if len(parts) >= 3:
                code = parts[1]
                # Remove language identifier if present
                lines = code.split('\n')
                if lines and lines[0].strip() in ['cpp', 'c++', 'c']:
                    code = '\n'.join(lines[1:])
                return code.strip()

        # No code block found, return as is
        return generated_text.strip()

    def generate_batch(
        self,
        programs: List[Program],
        **generation_kwargs
    ) -> List[Dict]:
        """Generate code for a batch of programs.

        Args:
            programs: List of Program objects
            **generation_kwargs: Arguments to pass to generate_code

        Returns:
            List of dictionaries with results
        """
        results = []

        print(f"Generating code for {len(programs)} programs...")

        for program in tqdm(programs, desc="Generating"):
            pseudocode = program.get_full_pseudocode()
            gold_code = program.get_full_code()

            try:
                generated_code = self.generate_code(pseudocode, **generation_kwargs)

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
        print(f"✓ Successfully generated {success_count}/{len(results)} programs")

        return results

    def save_results(self, results: List[Dict], output_path: Path):
        """Save generation results to JSON.

        Args:
            results: List of result dictionaries
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to {output_path}")
