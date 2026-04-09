#!/usr/bin/env python3
"""
miner.py — Files everything into the palace.

Reads mempalace.yaml from the project directory to know the wing + rooms.
Routes each file to the right room based on content.
Stores verbatim chunks as drawers. No summaries. Ever.
"""

import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import chromadb

READABLE_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".css",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".sh",
    ".csv",
    ".sql",
    ".toml",
}

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "coverage",
    ".mempalace",
}

CHUNK_SIZE = 800  # chars per drawer
CHUNK_OVERLAP = 100  # overlap between chunks
MIN_CHUNK_SIZE = 50  # skip tiny chunks


# =============================================================================
# CONFIG
# =============================================================================


def load_config(project_dir: str) -> dict:
    """Load mempalace.yaml from project directory (falls back to mempal.yaml)."""
    import yaml

    config_path = Path(project_dir).expanduser().resolve() / "mempalace.yaml"
    if not config_path.exists():
        # Fallback to legacy name
        legacy_path = Path(project_dir).expanduser().resolve() / "mempal.yaml"
        if legacy_path.exists():
            config_path = legacy_path
        else:
            print(f"ERROR: No mempalace.yaml found in {project_dir}")
            print(f"Run: mempalace init {project_dir}")
            sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# FILE ROUTING — which room does this file belong to?
# =============================================================================


def detect_room(filepath: Path, content: str, rooms: list, project_path: Path) -> str:
    """
    Route a file to the right room.
    Priority:
    1. Folder path matches a room name
    2. Filename matches a room name or keyword
    3. Content keyword scoring
    4. Fallback: "general"
    """
    relative = str(filepath.relative_to(project_path)).lower()
    filename = filepath.stem.lower()
    content_lower = content[:2000].lower()

    # Priority 1: folder path contains room name
    path_parts = relative.replace("\\", "/").split("/")
    for part in path_parts[:-1]:  # skip filename itself
        for room in rooms:
            if room["name"].lower() in part or part in room["name"].lower():
                return room["name"]

    # Priority 2: filename matches room name
    for room in rooms:
        if room["name"].lower() in filename or filename in room["name"].lower():
            return room["name"]

    # Priority 3: keyword scoring from room keywords + name
    scores = defaultdict(int)
    for room in rooms:
        keywords = room.get("keywords", []) + [room["name"]]
        for kw in keywords:
            count = content_lower.count(kw.lower())
            scores[room["name"]] += count

    if scores:
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best

    return "general"


# =============================================================================
# CHUNKING
# =============================================================================


def chunk_text(content: str, source_file: str) -> list:
    """
    Split content into drawer-sized chunks.
    Tries to split on paragraph/line boundaries.
    Returns list of {"content": str, "chunk_index": int}
    """
    # Clean up
    content = content.strip()
    if not content:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(content):
        end = min(start + CHUNK_SIZE, len(content))

        # Try to break at paragraph boundary
        if end < len(content):
            newline_pos = content.rfind("\n\n", start, end)
            if newline_pos > start + CHUNK_SIZE // 2:
                end = newline_pos
            else:
                newline_pos = content.rfind("\n", start, end)
                if newline_pos > start + CHUNK_SIZE // 2:
                    end = newline_pos

        chunk = content[start:end].strip()
        if len(chunk) >= MIN_CHUNK_SIZE:
            chunks.append(
                {
                    "content": chunk,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

        start = end - CHUNK_OVERLAP if end < len(content) else end

    return chunks


# =============================================================================
# PALACE — ChromaDB operations
# =============================================================================


def get_collection(palace_path: str):
    os.makedirs(palace_path, exist_ok=True)
    client = chromadb.PersistentClient(path=palace_path)
    from .embeddings import get_collection as _emb_get_collection
    return _emb_get_collection(client, "mempalace_drawers", create=True)


def file_already_mined(collection, source_file: str) -> bool:
    """Fast check: has this file been filed before?"""
    try:
        results = collection.get(where={"source_file": source_file}, limit=1)
        return len(results.get("ids", [])) > 0
    except Exception:
        return False


def add_drawer(
    collection, wing: str, room: str, content: str, source_file: str, chunk_index: int, agent: str
):
    """Add one drawer to the palace."""
    drawer_id = f"drawer_{wing}_{room}_{hashlib.md5((source_file + str(chunk_index)).encode()).hexdigest()[:16]}"
    try:
        collection.add(
            documents=[content],
            ids=[drawer_id],
            metadatas=[
                {
                    "wing": wing,
                    "room": room,
                    "source_file": source_file,
                    "chunk_index": chunk_index,
                    "added_by": agent,
                    "filed_at": datetime.now().isoformat(),
                }
            ],
        )
        return True
    except Exception as e:
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            return False
        raise


# =============================================================================
# PROCESS ONE FILE
# =============================================================================


def process_file(
    filepath: Path,
    project_path: Path,
    collection,
    wing: str,
    rooms: list,
    agent: str,
    dry_run: bool,
) -> list:
    """Read, chunk, route, and prepare drawers for one file. Returns list of drawer dicts."""

    source_file = str(filepath)
    if not dry_run and file_already_mined(collection, source_file):
        return []

    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    content = content.strip()
    if len(content) < MIN_CHUNK_SIZE:
        return []

    content_hash = hashlib.md5(content.encode()).hexdigest()

    room = detect_room(filepath, content, rooms, project_path)
    chunks = chunk_text(content, source_file)

    if dry_run:
        print(f"    [DRY RUN] {filepath.name} → room:{room} ({len(chunks)} drawers)")
        return [{"dry_run": True} for _ in chunks]

    drawers = []
    for chunk in chunks:
        drawer_id = f"drawer_{wing}_{room}_{hashlib.md5((source_file + str(chunk['chunk_index'])).encode()).hexdigest()[:16]}"
        drawers.append({
            "id": drawer_id,
            "document": chunk["content"],
            "metadata": {
                "wing": wing,
                "room": room,
                "source_file": source_file,
                "chunk_index": chunk["chunk_index"],
                "content_hash": content_hash,
                "added_by": agent,
                "filed_at": datetime.now().isoformat(),
            }
        })
    return drawers


# =============================================================================
# SCAN PROJECT
# =============================================================================


def scan_project(project_dir: str) -> list:
    """Return list of all readable file paths."""
    project_path = Path(project_dir).expanduser().resolve()
    files = []
    for root, dirs, filenames in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            filepath = Path(root) / filename
            if filepath.suffix.lower() in READABLE_EXTENSIONS:
                # Skip config files
                if filename in (
                    "mempalace.yaml",
                    "mempalace.yml",
                    "mempal.yaml",
                    "mempal.yml",
                    ".gitignore",
                    "package-lock.json",
                ):
                    continue
                files.append(filepath)
    return files


# =============================================================================
# MAIN: MINE
# =============================================================================


def mine(
    project_dir: str,
    palace_path: str,
    wing_override: str = None,
    agent: str = "mempalace",
    limit: int = 0,
    dry_run: bool = False,
):
    """Mine a project directory into the palace."""

    project_path = Path(project_dir).expanduser().resolve()
    config = load_config(project_dir)

    wing = wing_override or config["wing"]
    rooms = config.get("rooms", [{"name": "general", "description": "All project files"}])

    files = scan_project(project_dir)
    if limit > 0:
        files = files[:limit]

    print(f"\n{'=' * 55}")
    print("  MemPalace Mine")
    print(f"{'=' * 55}")
    print(f"  Wing:    {wing}")
    print(f"  Rooms:   {', '.join(r['name'] for r in rooms)}")
    print(f"  Files:   {len(files)}")
    print(f"  Palace:  {palace_path}")
    if dry_run:
        print("  DRY RUN — nothing will be filed")
    print(f"{'─' * 55}\n")

    if not dry_run:
        collection = get_collection(palace_path)
    else:
        collection = None

    from .embeddings import flush_batch, BATCH_SIZE

    total_drawers = 0
    files_skipped = 0
    room_counts = defaultdict(int)

    pending = []
    for i, filepath in enumerate(files, 1):
        drawers = process_file(
            filepath=filepath,
            project_path=project_path,
            collection=collection,
            wing=wing,
            rooms=rooms,
            agent=agent,
            dry_run=dry_run,
        )
        if not drawers:
            files_skipped += 1
            continue

        total_drawers += len(drawers)
        room = detect_room(filepath, "", rooms, project_path)
        room_counts[room] += 1

        if not dry_run:
            pending.extend(drawers)
            if len(pending) >= BATCH_SIZE:
                flush_batch(collection, pending)
                print(f"  ✓ Batch flushed — {len(pending)} drawers (file {i}/{len(files)})")
                pending = []

    # Flush remaining
    if pending and not dry_run:
        flush_batch(collection, pending)
        print(f"  ✓ Final batch — {len(pending)} drawers")

    print(f"\n{'=' * 55}")
    print("  Done.")
    print(f"  Files processed: {len(files) - files_skipped}")
    print(f"  Files skipped (already filed): {files_skipped}")
    print(f"  Drawers filed: {total_drawers}")
    print("\n  By room:")
    for room, count in sorted(room_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {room:20} {count} files")
    print('\n  Next: mempalace search "what you\'re looking for"')
    print(f"{'=' * 55}\n")


# =============================================================================
# INCREMENTAL UPDATE
# =============================================================================


def update(
    project_dir: str,
    palace_path: str,
    wing_override: str = None,
    agent: str = "mempalace",
    dry_run: bool = False,
):
    """Incremental update: sync palace with current file state."""
    from .embeddings import flush_batch, BATCH_SIZE

    project_path = Path(project_dir).expanduser().resolve()
    config = load_config(project_dir)
    wing = wing_override or config["wing"]
    rooms = config.get("rooms", [{"name": "general", "description": "All project files"}])

    # Get current files on disk
    disk_files = scan_project(project_dir)
    disk_paths = {str(f) for f in disk_files}

    print(f"\n{'=' * 55}")
    print("  MemPalace Update")
    print(f"{'=' * 55}")
    print(f"  Wing:    {wing}")
    print(f"  Files:   {len(disk_files)} on disk")
    print(f"  Palace:  {palace_path}")
    if dry_run:
        print("  DRY RUN — nothing will change")
    print(f"{'─' * 55}\n")

    collection = get_collection(palace_path)

    # Get all drawers in this wing from the palace
    try:
        results = collection.get(
            where={"wing": wing},
            include=["metadatas"],
            limit=100000,
        )
    except Exception:
        results = {"ids": [], "metadatas": []}

    # Build map: source_file -> {ids: [...], content_hash: str}
    palace_files = defaultdict(lambda: {"ids": [], "content_hash": None})
    for drawer_id, meta in zip(results["ids"], results["metadatas"]):
        sf = meta.get("source_file", "")
        palace_files[sf]["ids"].append(drawer_id)
        if meta.get("content_hash"):
            palace_files[sf]["content_hash"] = meta["content_hash"]

    palace_paths = set(palace_files.keys())

    # Classify files
    new_files = []
    changed_files = []
    deleted_paths = palace_paths - disk_paths
    unchanged = 0

    for filepath in disk_files:
        source_file = str(filepath)
        if source_file not in palace_paths:
            new_files.append(filepath)
        else:
            # Check if content changed
            try:
                content = filepath.read_text(encoding="utf-8", errors="replace").strip()
                current_hash = hashlib.md5(content.encode()).hexdigest()
            except Exception:
                continue
            stored_hash = palace_files[source_file]["content_hash"]
            if stored_hash and stored_hash == current_hash:
                unchanged += 1
            else:
                changed_files.append(filepath)

    print(f"  New:       {len(new_files)}")
    print(f"  Changed:   {len(changed_files)}")
    print(f"  Deleted:   {len(deleted_paths)}")
    print(f"  Unchanged: {unchanged}")
    print()

    if not new_files and not changed_files and not deleted_paths:
        print("  Everything up to date.")
        print(f"{'=' * 55}\n")
        return

    # 1. Remove drawers for deleted files
    if deleted_paths:
        for sf in deleted_paths:
            ids = palace_files[sf]["ids"]
            if dry_run:
                print(f"    [DRY RUN] DELETE {Path(sf).name} ({len(ids)} drawers)")
            else:
                collection.delete(ids=ids)
                print(f"  ✗ Removed {Path(sf).name} ({len(ids)} drawers)")

    # 2. Remove old drawers for changed files
    if changed_files:
        for filepath in changed_files:
            sf = str(filepath)
            ids = palace_files[sf]["ids"]
            if not dry_run:
                collection.delete(ids=ids)

    # 3. Mine new + changed files
    files_to_mine = new_files + changed_files
    if files_to_mine:
        pending = []
        total_drawers = 0
        for i, filepath in enumerate(files_to_mine, 1):
            label = "NEW" if filepath in new_files else "UPD"
            drawers = process_file(
                filepath=filepath,
                project_path=project_path,
                collection=collection,
                wing=wing,
                rooms=rooms,
                agent=agent,
                dry_run=dry_run,
            )
            if not drawers:
                continue
            total_drawers += len(drawers)
            if dry_run:
                print(f"    [DRY RUN] {label} {filepath.name} ({len(drawers)} drawers)")
            else:
                pending.extend(drawers)
                if len(pending) >= BATCH_SIZE:
                    flush_batch(collection, pending)
                    print(f"  ✓ Batch flushed — {len(pending)} drawers")
                    pending = []

        if pending and not dry_run:
            flush_batch(collection, pending)
            print(f"  ✓ Final batch — {len(pending)} drawers")

        print(f"\n  Filed {total_drawers} drawers from {len(files_to_mine)} files")

    deleted_count = sum(len(palace_files[sf]["ids"]) for sf in deleted_paths)
    print(f"\n{'=' * 55}")
    print("  Update complete.")
    if deleted_paths:
        print(f"  Removed: {deleted_count} drawers ({len(deleted_paths)} files)")
    print(f"{'=' * 55}\n")


# =============================================================================
# STATUS
# =============================================================================


def status(palace_path: str):
    """Show what's been filed in the palace."""
    try:
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
    except Exception:
        print(f"\n  No palace found at {palace_path}")
        print("  Run: mempalace init <dir> then mempalace mine <dir>")
        return

    # Count by wing and room
    r = col.get(limit=10000, include=["metadatas"])
    metas = r["metadatas"]

    wing_rooms = defaultdict(lambda: defaultdict(int))
    for m in metas:
        wing_rooms[m.get("wing", "?")][m.get("room", "?")] += 1

    print(f"\n{'=' * 55}")
    print(f"  MemPalace Status — {len(metas)} drawers")
    print(f"{'=' * 55}\n")
    for wing, rooms in sorted(wing_rooms.items()):
        print(f"  WING: {wing}")
        for room, count in sorted(rooms.items(), key=lambda x: x[1], reverse=True):
            print(f"    ROOM: {room:20} {count:5} drawers")
        print()
    print(f"{'=' * 55}\n")
