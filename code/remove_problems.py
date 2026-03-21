"""
remove_problems.py  --  Remove specific problems from paraphrases.json by ID
=============================================================================
Edits data/paraphrases.json in-place, removing any problems whose ID
matches the REMOVE_IDS list below. A backup is saved before any changes.

Usage
-----
  1. Edit the REMOVE_IDS list below with the IDs you want to remove.
  2. Run:  python code/remove_problems.py

The script also accepts IDs as command-line arguments, which override
the REMOVE_IDS list:

  python code/remove_problems.py GSM_Charlie_has_three_times GSM_Indras_has_letters_in

"""

import json
import os
import shutil
import sys

# ---------------------------------------------------------------------------
# Edit this list to specify which problems to remove.
# IDs are matched exactly (case-sensitive).
# ---------------------------------------------------------------------------
REMOVE_IDS = [
    # "GSM_Charlie_has_three_times",
    # "GSM_Indras_has_letters_in",
    # "GSM_iPhone_is_four_times",
    # "GSM_John_drives_to_his",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")
INPUT_FILE = os.path.join(DATA_DIR, "paraphrases.json")
BACKUP_FILE = os.path.join(DATA_DIR, "paraphrases.json.bak")


def main():
    # Allow IDs to be passed as command-line arguments
    remove_ids = set(sys.argv[1:]) if len(sys.argv) > 1 else set(REMOVE_IDS)

    if not remove_ids:
        print("No IDs specified. Edit REMOVE_IDS in the script or pass IDs as arguments.")
        print("Example:  python code/remove_problems.py GSM_Charlie_has_three_times")
        sys.exit(0)

    if not os.path.exists(INPUT_FILE):
        sys.exit(f"ERROR: {INPUT_FILE} not found.")

    # Load
    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    original_count = len(data)
    print(f"Loaded {original_count} problems from {INPUT_FILE}")
    print(f"IDs to remove: {sorted(remove_ids)}\n")

    # Check which IDs actually exist
    existing_ids = {p["id"] for p in data}
    not_found = remove_ids - existing_ids
    if not_found:
        print(f"WARNING: These IDs were not found in the file and will be skipped:")
        for nf in sorted(not_found):
            print(f"  {nf}")
        print()

    # Filter
    kept    = [p for p in data if p["id"] not in remove_ids]
    removed = [p for p in data if p["id"] in remove_ids]

    if not removed:
        print("Nothing to remove. File unchanged.")
        sys.exit(0)

    # Backup
    shutil.copy2(INPUT_FILE, BACKUP_FILE)
    print(f"Backup saved: {BACKUP_FILE}")

    # Write
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    print(f"\nRemoved {len(removed)} problem(s):")
    for p in removed:
        n_paraphrases = len(p.get("paraphrases", [])) - 1  # exclude P0
        print(f"  [{p['id']}]  domain={p['domain']}  difficulty={p['difficulty']}  paraphrases={n_paraphrases}")

    print(f"\nProblems remaining: {len(kept)} (was {original_count})")
    print(f"Updated: {INPUT_FILE}")
    print(f"\nTo undo: cp {BACKUP_FILE} {INPUT_FILE}")


if __name__ == "__main__":
    main()
