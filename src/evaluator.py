"""
Evaluation module for SPOC code generation.

Implements metrics:
- BLEU: Lexical similarity between generated and gold code
- CompAcc: Compilation accuracy (% of programs that compile)
- FCorrAcc: Functional correctness (% that compile and pass all tests)
- Pass@1: % of problems with at least one correct solution
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
import sacrebleu
from collections import defaultdict

from data_loader import SPOCDataLoader, TestCase


class CodeEvaluator:
    """Evaluate generated C++ code."""

    def __init__(
        self,
        data_loader: SPOCDataLoader,
        compiler: str = "g++",
        timeout: int = 5
    ):
        """Initialize evaluator.

        Args:
            data_loader: SPOC data loader for accessing test cases
            compiler: C++ compiler command (default: g++)
            timeout: Timeout in seconds for compilation and execution
        """
        self.data_loader = data_loader
        self.compiler = compiler
        self.timeout = timeout

    def compute_bleu(self, results: List[Dict]) -> Dict[str, float]:
        """Compute BLEU scores.

        Args:
            results: List of generation results

        Returns:
            Dictionary with BLEU metrics
        """
        references = []
        hypotheses = []

        for result in results:
            if result["success"]:
                references.append([result["gold_code"]])
                hypotheses.append(result["generated_code"])

        if not hypotheses:
            return {"bleu": 0.0}

        # Compute corpus BLEU
        bleu = sacrebleu.corpus_bleu(hypotheses, references)

        return {
            "bleu": bleu.score,
            "bleu_precisions": bleu.precisions
        }

    def compile_code(self, code: str) -> Tuple[bool, str, Path]:
        """Compile C++ code.

        Args:
            code: C++ source code

        Returns:
            Tuple of (success, error_message, executable_path)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            source_file = tmpdir_path / "program.cpp"
            executable = tmpdir_path / "program"

            # Write source code
            with open(source_file, 'w') as f:
                f.write(code)

            # Compile
            try:
                result = subprocess.run(
                    [self.compiler, str(source_file), "-o", str(executable)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

                if result.returncode == 0:
                    # Copy executable to a persistent location
                    persistent_exec = Path(tempfile.mktemp(suffix=".out"))
                    subprocess.run(["cp", str(executable), str(persistent_exec)], check=True)
                    return True, "", persistent_exec
                else:
                    return False, result.stderr, None

            except subprocess.TimeoutExpired:
                return False, "Compilation timeout", None
            except Exception as e:
                return False, str(e), None

    def run_test_case(self, executable: Path, test_case: TestCase) -> Tuple[bool, str]:
        """Run a single test case.

        Args:
            executable: Path to compiled executable
            test_case: Test case with input and expected output

        Returns:
            Tuple of (passed, actual_output)
        """
        try:
            result = subprocess.run(
                [str(executable)],
                input=test_case.input_data,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            actual_output = result.stdout.strip()
            expected_output = test_case.expected_output.strip()

            # Check if output matches
            passed = actual_output == expected_output

            return passed, actual_output

        except subprocess.TimeoutExpired:
            return False, "Execution timeout"
        except Exception as e:
            return False, str(e)

    def evaluate_program(
        self,
        code: str,
        probid: str,
        use_hidden: bool = True
    ) -> Dict:
        """Evaluate a single program.

        Args:
            code: Generated C++ code
            probid: Problem identifier for loading test cases
            use_hidden: Whether to use hidden test cases (True) or public (False)

        Returns:
            Dictionary with evaluation results
        """
        result = {
            "compiled": False,
            "compile_error": None,
            "tests_passed": 0,
            "tests_total": 0,
            "all_tests_passed": False,
            "test_details": []
        }

        # Compile
        compiled, error, executable = self.compile_code(code)
        result["compiled"] = compiled
        result["compile_error"] = error

        if not compiled:
            return result

        # Load test cases
        try:
            public_tests, hidden_tests = self.data_loader.load_test_cases(probid)
            test_cases = hidden_tests if use_hidden else public_tests

            result["tests_total"] = len(test_cases)

            # Run test cases
            for i, test_case in enumerate(test_cases):
                passed, actual_output = self.run_test_case(executable, test_case)

                if passed:
                    result["tests_passed"] += 1

                result["test_details"].append({
                    "test_num": i,
                    "passed": passed,
                    "expected": test_case.expected_output[:100],  # Truncate for storage
                    "actual": actual_output[:100] if isinstance(actual_output, str) else str(actual_output)[:100]
                })

            result["all_tests_passed"] = result["tests_passed"] == result["tests_total"]

        except Exception as e:
            result["compile_error"] = f"Test execution error: {str(e)}"
        finally:
            # Clean up executable
            if executable and executable.exists():
                executable.unlink()

        return result

    def evaluate_batch(
        self,
        results: List[Dict],
        use_hidden: bool = True
    ) -> List[Dict]:
        """Evaluate a batch of generated programs.

        Args:
            results: List of generation results
            use_hidden: Whether to use hidden test cases

        Returns:
            Updated results with evaluation metrics
        """
        print(f"Evaluating {len(results)} programs...")

        for i, result in enumerate(results):
            if not result["success"]:
                continue

            print(f"Evaluating {i+1}/{len(results)}: {result['program_id']}", end="\r")

            eval_result = self.evaluate_program(
                result["generated_code"],
                result["probid"],
                use_hidden=use_hidden
            )

            result["evaluation"] = eval_result

        print()  # New line after progress
        return results

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute aggregate metrics from evaluation results.

        Metrics:
        - CompAcc: % of programs that compiled successfully
        - FCorrAcc: % of programs that passed all test cases
        - Pass@1: % of problems with at least one correct solution

        Args:
            results: List of evaluated results

        Returns:
            Dictionary of metrics
        """
        # Filter to successfully generated programs
        valid_results = [r for r in results if r.get("success", False)]

        if not valid_results:
            return {
                "comp_acc": 0.0,
                "fcorr_acc": 0.0,
                "pass_at_1": 0.0,
                "total_programs": 0,
                "compiled_programs": 0,
                "correct_programs": 0
            }

        # CompAcc: % that compiled
        compiled_count = sum(
            1 for r in valid_results
            if r.get("evaluation", {}).get("compiled", False)
        )
        comp_acc = (compiled_count / len(valid_results)) * 100

        # FCorrAcc: % that passed all tests
        correct_count = sum(
            1 for r in valid_results
            if r.get("evaluation", {}).get("all_tests_passed", False)
        )
        fcorr_acc = (correct_count / len(valid_results)) * 100

        # Pass@1: % of problems with at least one correct solution
        problem_correct = defaultdict(bool)
        for r in valid_results:
            probid = r["probid"]
            if r.get("evaluation", {}).get("all_tests_passed", False):
                problem_correct[probid] = True

        total_problems = len(set(r["probid"] for r in valid_results))
        correct_problems = sum(problem_correct.values())
        pass_at_1 = (correct_problems / total_problems * 100) if total_problems > 0 else 0.0

        return {
            "comp_acc": comp_acc,
            "fcorr_acc": fcorr_acc,
            "pass_at_1": pass_at_1,
            "total_programs": len(valid_results),
            "compiled_programs": compiled_count,
            "correct_programs": correct_count,
            "total_problems": total_problems,
            "correct_problems": correct_problems
        }

    def print_metrics_summary(self, metrics: Dict, bleu_metrics: Dict = None):
        """Print a formatted metrics summary.

        Args:
            metrics: Compilation/execution metrics
            bleu_metrics: Optional BLEU metrics
        """
        print()
        print("=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print()

        if bleu_metrics:
            print("Lexical Similarity:")
            print(f"  BLEU: {bleu_metrics['bleu']:.2f}")
            print()

        print("Compilation & Correctness:")
        print(f"  CompAcc (Compilation Accuracy):   {metrics['comp_acc']:.2f}%")
        print(f"  FCorrAcc (Functional Correctness): {metrics['fcorr_acc']:.2f}%")
        print(f"  Pass@1:                            {metrics['pass_at_1']:.2f}%")
        print()

        print("Details:")
        print(f"  Total programs:    {metrics['total_programs']}")
        print(f"  Compiled:          {metrics['compiled_programs']}")
        print(f"  Functionally correct: {metrics['correct_programs']}")
        print(f"  Total problems:    {metrics['total_problems']}")
        print(f"  Correct problems:  {metrics['correct_problems']}")
        print()
        print("=" * 80)
