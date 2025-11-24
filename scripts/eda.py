"""
Simple EDA script for SPOC dataset.

Tests that we can load data and previews what it looks like.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import SPOCDataLoader

# %%
def main():
    """Run basic EDA on SPOC dataset."""
    print("=" * 80)
    print("SPOC Dataset - Exploratory Data Analysis")
    print("=" * 80)
    print()

    # Initialize loader
    loader = SPOCDataLoader(data_dir="..")

    # Load test splits
    print("Loading test splits...")
    testp_programs = loader.load_test_split("testp")
    testw_programs = loader.load_test_split("testw")

    print(f"✓ Loaded {len(testp_programs)} programs from testp split")
    print(f"✓ Loaded {len(testw_programs)} programs from testw split")
    print()

    # Basic statistics
    print("-" * 80)
    print("Dataset Statistics")
    print("-" * 80)

    # Get unique counts
    all_programs = testp_programs + testw_programs
    unique_problems = loader.get_unique_problems(all_programs)
    unique_workers = set(p.workerid for p in all_programs)

    print(f"Total programs: {len(all_programs)}")
    print(f"Unique problems: {len(unique_problems)}")
    print(f"Unique workers: {len(unique_workers)}")
    print()

    # Line statistics
    total_lines = sum(len(p.code_lines) for p in all_programs)
    avg_lines = total_lines / len(all_programs) if all_programs else 0
    print(f"Total code lines: {total_lines}")
    print(f"Average lines per program: {avg_lines:.1f}")
    print()

    # %%
    # Show sample programs
    print("-" * 80)
    print("Sample Programs")
    print("-" * 80)
    print()

    for i, program in enumerate(testp_programs[:3]):
        print(f"Program {i+1}: {program.get_program_id()}")
        print(f"Problem ID: {program.probid}")
        print(f"Lines of code: {len(program.code_lines)}")
        print()

        print("Pseudocode:")
        print("-" * 40)
        for j, (pseudo, code) in enumerate(zip(program.pseudocode_lines[:10], program.code_lines[:10])):
            if pseudo:  # Only show lines with pseudocode
                print(f"  [{j}] {pseudo}")
        if len(program.pseudocode_lines) > 10:
            print(f"  ... ({len(program.pseudocode_lines) - 10} more lines)")
        print()

        print("Code:")
        print("-" * 40)
        for j, line in enumerate(program.code_lines[:10]):
            indent = "  " * program.indent_levels[j]
            print(f"  {indent}{line}")
        if len(program.code_lines) > 10:
            print(f"  ... ({len(program.code_lines) - 10} more lines)")
        print()
        print("=" * 80)
        print()

    # %%
    # Test case preview
    print("-" * 80)
    print("Test Case Preview")
    print("-" * 80)
    print()

    # Get test cases for first problem
    sample_probid = testp_programs[0].probid
    print(f"Loading test cases for problem: {sample_probid}")

    try:
        public_tests, hidden_tests = loader.load_test_cases(sample_probid)
        print(f"✓ Found {len(public_tests)} public test cases")
        print(f"✓ Found {len(hidden_tests)} hidden test cases")
        print()

        if public_tests:
            print("Sample public test case:")
            test = public_tests[0]
            print("Input:")
            print(f"  {test.input_data[:100]}{'...' if len(test.input_data) > 100 else ''}")
            print("Expected Output:")
            print(f"  {test.expected_output[:100]}{'...' if len(test.expected_output) > 100 else ''}")
    except Exception as e:
        print(f"✗ Error loading test cases: {e}")

    print()
    print("=" * 80)
    print("EDA Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
