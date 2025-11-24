"""
Data loader for SPOC (Stack Overflow Pseudocode to Code) dataset.

This module handles loading TSV files, grouping lines into complete programs,
and loading test cases for evaluation.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Program:
    """Represents a complete program with pseudocode and code."""
    probid: str
    subid: str
    workerid: str
    pseudocode_lines: List[str]
    code_lines: List[str]
    indent_levels: List[int]

    def get_full_pseudocode(self) -> str:
        """Get the full pseudocode as a single string."""
        return "\n".join(self.pseudocode_lines)

    def get_full_code(self) -> str:
        """Get the full code as a single string."""
        return "\n".join(self.code_lines)

    def get_program_id(self) -> str:
        """Get unique program identifier."""
        return f"{self.probid}_{self.subid}_{self.workerid}"


@dataclass
class TestCase:
    """Represents a test case with input and expected output."""
    input_data: str
    expected_output: str


class SPOCDataLoader:
    """Loader for SPOC dataset."""

    def __init__(self, data_dir: str = "."):
        """Initialize data loader.

        Args:
            data_dir: Root directory containing train/, test/, and testcases/ folders
        """
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        self.testcases_dir = self.data_dir / "testcases"

    def load_tsv(self, file_path: Path) -> pd.DataFrame:
        """Load a TSV file into a pandas DataFrame.

        Args:
            file_path: Path to the TSV file

        Returns:
            DataFrame with columns: text, code, workerid, probid, subid, line, indent
        """
        df = pd.read_csv(
            file_path,
            sep="\t",
            dtype={
                'text': str,
                'code': str,
                'workerid': str,
                'probid': str,
                'subid': str,
                'line': int,
                'indent': int
            }
        )
        # Replace NaN in text column with empty string
        df['text'] = df['text'].fillna('')
        return df

    def group_into_programs(self, df: pd.DataFrame) -> List[Program]:
        """Group DataFrame rows into complete programs.

        Each program is identified by (probid, subid, workerid).
        Lines are grouped by detecting when line number resets to 0.

        Args:
            df: DataFrame loaded from TSV

        Returns:
            List of Program objects
        """
        programs = []

        # Group by probid, subid, workerid
        for (probid, subid, workerid), group in df.groupby(['probid', 'subid', 'workerid']):
            # Sort by line number to ensure correct order
            group = group.sort_values('line')

            program = Program(
                probid=str(probid),
                subid=str(subid),
                workerid=str(workerid),
                pseudocode_lines=group['text'].tolist(),
                code_lines=group['code'].tolist(),
                indent_levels=group['indent'].tolist()
            )
            programs.append(program)

        return programs

    def load_test_split(self, split_name: str = "testp") -> List[Program]:
        """Load a test split.

        Args:
            split_name: Name of split ('testp' or 'testw')

        Returns:
            List of programs from that split
        """
        file_path = self.test_dir / f"spoc-{split_name}.tsv"
        df = self.load_tsv(file_path)
        return self.group_into_programs(df)

    def load_train_split(self) -> List[Program]:
        """Load the full training split.

        Returns:
            List of programs from training data
        """
        file_path = self.train_dir / "spoc-train.tsv"
        df = self.load_tsv(file_path)
        return self.group_into_programs(df)

    def get_subset(self, programs: List[Program], n: int = 100) -> List[Program]:
        """Get a small subset of programs for testing.

        Args:
            programs: List of all programs
            n: Number of programs to sample

        Returns:
            Subset of n programs
        """
        return programs[:min(n, len(programs))]

    def load_test_cases(self, probid: str) -> Tuple[List[TestCase], List[TestCase]]:
        """Load test cases for a specific problem.

        Args:
            probid: Problem identifier

        Returns:
            Tuple of (public_test_cases, hidden_test_cases)
        """
        prob_dir = self.testcases_dir / probid

        public_tests = self._parse_test_file(prob_dir / f"{probid}_testcases_public.txt")
        hidden_tests = self._parse_test_file(prob_dir / f"{probid}_testcases_hidden.txt")

        return public_tests, hidden_tests

    def _parse_test_file(self, file_path: Path) -> List[TestCase]:
        """Parse a test case file.

        Format:
        input_line1
        input_line2
        ###ENDINPUT###
        output_line1
        output_line2
        ###ENDOUTPUT###

        Args:
            file_path: Path to test case file

        Returns:
            List of TestCase objects
        """
        if not file_path.exists():
            return []

        with open(file_path, 'r') as f:
            content = f.read()

        test_cases = []

        # Split by ###ENDOUTPUT### to get individual test cases
        raw_cases = content.split('###ENDOUTPUT###')

        for raw_case in raw_cases:
            raw_case = raw_case.strip()
            if not raw_case:
                continue

            # Split by ###ENDINPUT### to separate input and output
            parts = raw_case.split('###ENDINPUT###')
            if len(parts) != 2:
                continue

            input_data = parts[0].strip()
            expected_output = parts[1].strip()

            test_cases.append(TestCase(
                input_data=input_data,
                expected_output=expected_output
            ))

        return test_cases

    def get_unique_problems(self, programs: List[Program]) -> List[str]:
        """Get list of unique problem IDs from programs.

        Args:
            programs: List of programs

        Returns:
            Sorted list of unique problem IDs
        """
        return sorted(list(set(p.probid for p in programs)))
