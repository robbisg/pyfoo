#!/usr/bin/env python3
"""
Rename files/dirs and update content/mtimes from OLD_YEAR to NEW_YEAR.

Usage:
    python rename_year.py <root_dir> [--dry-run]
"""
import os
import re
import sys
from datetime import datetime

ROOT = sys.argv[1]
DRY_RUN = "--dry-run" in sys.argv
OLD_YEAR = "2025"
NEW_YEAR = "2023"

# Matches 2023 only in date-like contexts:
#   year-first: 2023-MM, 2023/MM, 2023Txx
#   year-last:  DD.2023, DD/2023, DD-2023
DATE_RE = re.compile(
    r"(?<!\d)2023(?=[-/T]\d{2})"
    r"|(?<=\d{2}[./-])2023(?!\d)"
)

if DRY_RUN:
    print("[DRY RUN] No changes will be made.\n")


def maybe_rename(path):
    parent = os.path.dirname(path)
    name = os.path.basename(path)
    new_name = name.replace(OLD_YEAR, NEW_YEAR)
    if new_name != name:
        new_path = os.path.join(parent, new_name)
        print(f"RENAME  {path}  ->  {new_path}")
        if not DRY_RUN:
            os.rename(path, new_path)
        return new_path
    return path


def replace_in_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    new_content = DATE_RE.sub(NEW_YEAR, content)
    if new_content != content:
        print(f"MODIFY  {filepath}")
        if not DRY_RUN:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)


def fix_mtime(path):
    stat = os.stat(path)
    mtime = datetime.fromtimestamp(stat.st_mtime)
    new_mtime = mtime.replace(year=int(NEW_YEAR))
    print(f"MTIME   {path}  {mtime:%Y-%m-%d %H:%M:%S}  ->  {new_mtime:%Y-%m-%d %H:%M:%S}")
    if not DRY_RUN:
        os.utime(path, (stat.st_atime, new_mtime.timestamp()))


# Snapshot root entries before any renaming
entries = os.listdir(ROOT)

# Step 1: rename root-level files and directories
new_names = {}
for name in entries:
    path = os.path.join(ROOT, name)
    new_path = maybe_rename(path)
    new_names[name] = os.path.basename(new_path)

# Step 2: process content of root-level txt files (e.g. ImpedenceData)
for original_name in entries:
    fpath = os.path.join(ROOT, new_names[original_name])
    if os.path.isfile(fpath) and fpath.rsplit(".", 1)[-1].lower() == "txt":
        replace_in_file(fpath)

# Step 3: process inside each dated directory
for original_name in entries:
    dir_path = os.path.join(ROOT, new_names[original_name])
    if not os.path.isdir(dir_path):
        continue

    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)

        # Replace year inside xml / txt / log files
        if os.path.isfile(fpath):
            ext = fname.rsplit(".", 1)[-1].lower()
            if ext in ("xml", "txt", "log"):
                replace_in_file(fpath)

# Step 4: fix mtime on every file and folder in the tree (bottom-up)
for dirpath, dirnames, filenames in os.walk(ROOT, topdown=False):
    for fname in filenames:
        fix_mtime(os.path.join(dirpath, fname))
    fix_mtime(dirpath)
