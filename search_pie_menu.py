"""Search for Pie Menu question in GAIA dataset."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from benchmark.gaia_loader import GAIALoader

def main():
    # Initialize loader
    data_dir = Path(__file__).parent / "data" / "gaia"
    loader = GAIALoader(data_dir)

    # Load validation tasks
    print("Loading GAIA validation tasks...")
    tasks = loader.load_tasks(split="validation", year="2023")
    print(f"Loaded {len(tasks)} tasks\n")

    # Search for "Pie Menu" in questions
    print("Searching for 'Pie Menu' in questions...")
    matches = []
    for task in tasks:
        if "Pie Menu" in task.question:
            matches.append(task)

    if matches:
        print(f"\nFound {len(matches)} matching task(s):\n")
        print("=" * 80)
        for task in matches:
            print(f"\nTask ID: {task.task_id}")
            print(f"Level: {task.level}")
            print(f"\nQuestion:\n{task.question}")
            print(f"\nFinal Answer: {task.final_answer}")
            if task.has_file():
                print(f"\nAssociated File: {task.file_name}")
                print(f"File Path: {task.file_path}")
            print("\n" + "=" * 80)
    else:
        print("\nNo tasks found containing 'Pie Menu'")

if __name__ == "__main__":
    main()
