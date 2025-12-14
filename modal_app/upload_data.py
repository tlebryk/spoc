"""
One-time data upload to Modal volume.

This script uploads the SPOC dataset and test cases to the Modal persistent volume
so they can be accessed by inference and evaluation functions.
"""

from pathlib import Path
import modal

# Create Modal app
app = modal.App("spoc-upload")

# Persistent volume
volume = modal.Volume.from_name("spoc-artifacts", create_if_missing=True)

# Dataset paths
VOLUME_PATH = "/artifacts"
DATASET_PATH = f"{VOLUME_PATH}/datasets"


@app.local_entrypoint()
def upload():
    """Upload SPOC dataset to Modal volume."""

    print("=" * 80)
    print("Uploading SPOC Dataset to Modal Volume")
    print("=" * 80)
    print()

    # Get paths relative to this script
    base_dir = Path(__file__).parent.parent

    # Check if data exists
    test_dir = base_dir / "test"
    testcases_dir = base_dir / "testcases"
    train_dir = base_dir / "train"

    if not test_dir.exists():
        print("Error: 'test/' directory not found!")
        print("Please ensure SPOC dataset is downloaded and extracted.")
        return

    print(f"Found local data directory: {base_dir}")
    print()

    # Upload test data
    if test_dir.exists():
        print(f"Uploading test data...")
        test_files = list(test_dir.glob("*.tsv"))
        print(f"  Found {len(test_files)} test files")

        with volume.batch_upload() as batch:
            for test_file in test_files:
                remote_path = f"{DATASET_PATH}/test/{test_file.name}"
                print(f"  Uploading {test_file.name} -> {remote_path}")
                batch.put_file(str(test_file), remote_path)

        print(f"  ✓ Test data uploaded")
        print()

    # Upload train data (optional, for future training)
    if train_dir.exists():
        print(f"Uploading train data...")
        train_files = list(train_dir.glob("*.tsv"))
        print(f"  Found {len(train_files)} train files")

        with volume.batch_upload() as batch:
            for train_file in train_files:
                remote_path = f"{DATASET_PATH}/train/{train_file.name}"
                print(f"  Uploading {train_file.name} -> {remote_path}")
                batch.put_file(str(train_file), remote_path)

        print(f"  ✓ Train data uploaded")
        print()

    # Upload test cases
    if testcases_dir.exists():
        print(f"Uploading test cases...")
        problem_dirs = [d for d in testcases_dir.iterdir() if d.is_dir()]
        print(f"  Found {len(problem_dirs)} problem directories")

        with volume.batch_upload() as batch:
            for problem_dir in problem_dirs:
                problem_id = problem_dir.name
                testcase_files = list(problem_dir.glob("*.txt"))

                for testcase_file in testcase_files:
                    remote_path = f"{DATASET_PATH}/testcases/{problem_id}/{testcase_file.name}"
                    batch.put_file(str(testcase_file), remote_path)

        print(f"  ✓ Test cases uploaded ({len(problem_dirs)} problems)")
        print()

    print("=" * 80)
    print("Upload Complete!")
    print("=" * 80)
    print()
    print("Data uploaded to Modal volume 'spoc-artifacts'")
    print("You can now run inference using: modal run modal_app/inference.py")
    print()
    print("To verify upload, check volume contents:")
    print("  modal volume ls spoc-artifacts")
