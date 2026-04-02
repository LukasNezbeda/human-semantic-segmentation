"""
Script to download the Person Segmentation dataset from Kaggle using curl.

The dataset will be downloaded and extracted to: data/person_segmentation/

Prerequisites:
    - curl must be installed and available in PATH
"""

import shutil
import subprocess
import sys
from pathlib import Path


def setup_dataset_directory() -> Path:
    """Create the dataset directory structure.

    Returns:
        Path: The path to the person_segmentation directory.
    """
    dataset_dir = Path(__file__).parent / "person_segmentation"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def download_dataset(output_dir: Path) -> bool:
    """Download dataset using curl.

    Args:
        output_dir: Directory where dataset will be saved.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    try:
        url = "https://www.kaggle.com/api/v1/datasets/download/nikhilroxtomar/person-segmentation"
        zip_path = output_dir / "person_segmentation.zip"

        print(f"Downloading dataset from: {url}")
        print(f"Destination: {zip_path}")

        # Curl command with progress bar
        cmd = [
            "curl",
            "-L",  # Follow redirects
            "-o", str(zip_path),  # Output file
            "--progress-bar",  # Show progress bar
            url,
        ]

        result = subprocess.run(cmd, check=True)

        if zip_path.exists():
            print("✓ Dataset downloaded successfully!")
            # Extract the zip file
            print("Extracting dataset...")
            shutil.unpack_archive(str(zip_path), output_dir)
            zip_path.unlink()  # Remove zip file after extraction
            print("✓ Dataset extracted successfully!")
            return True

        return False

    except FileNotFoundError:
        print("ERROR: curl not found. Please install curl and add it to PATH")
        return False
    except Exception as e:
        print(f"ERROR downloading dataset with curl: {e}")
        return False


def main() -> None:
    """Main function to download the Person Segmentation dataset."""
    print("=" * 60)
    print("Person Segmentation Dataset Downloader")
    print("=" * 60)

    output_dir = setup_dataset_directory()

    # Check if dataset already exists
    expected_subdirs = {"images", "masks"}
    existing_subdirs = {
        item.name for item in output_dir.iterdir() if item.is_dir()
    }

    if expected_subdirs.issubset(existing_subdirs):
        print("\n✓ Dataset already exists at:", output_dir)
        return

    print("\nDownloading dataset with curl...")
    if download_dataset(output_dir):
        print("\n" + "=" * 60)
        print(f"Dataset ready at: {output_dir}")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print("ERROR: Failed to download dataset!")
    print("=" * 60)
    print("\nPlease ensure curl is installed and available in your PATH")
    sys.exit(1)


if __name__ == "__main__":
    main()