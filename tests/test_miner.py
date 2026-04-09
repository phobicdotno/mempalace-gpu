import os
import tempfile
import shutil
import yaml
from mempalace.miner import mine, update, get_collection


def test_project_mining():
    tmpdir = tempfile.mkdtemp()
    # Create a mini project
    os.makedirs(os.path.join(tmpdir, "backend"))
    with open(os.path.join(tmpdir, "backend", "app.py"), "w") as f:
        f.write("def main():\n    print('hello world')\n" * 20)
    # Create config
    with open(os.path.join(tmpdir, "mempalace.yaml"), "w") as f:
        yaml.dump(
            {
                "wing": "test_project",
                "rooms": [
                    {"name": "backend", "description": "Backend code"},
                    {"name": "general", "description": "General"},
                ],
            },
            f,
        )

    palace_path = os.path.join(tmpdir, "palace")
    mine(tmpdir, palace_path)

    # Verify
    col = get_collection(palace_path)
    assert col.count() > 0

    shutil.rmtree(tmpdir)


def test_update_detects_changes():
    """Test that update finds new, changed, and deleted files."""
    tmpdir = os.path.realpath(tempfile.mkdtemp())
    # Create initial files
    with open(os.path.join(tmpdir, "file1.py"), "w") as f:
        f.write("# File 1\n" + "x = 1\n" * 20)
    with open(os.path.join(tmpdir, "file2.py"), "w") as f:
        f.write("# File 2\n" + "y = 2\n" * 20)
    with open(os.path.join(tmpdir, "mempalace.yaml"), "w") as f:
        yaml.dump({"wing": "test", "rooms": [{"name": "general", "description": "all"}]}, f)

    palace = os.path.realpath(tempfile.mkdtemp())
    mine(tmpdir, palace)

    col = get_collection(palace)
    initial_count = col.count()
    assert initial_count > 0

    # Modify file1, delete file2, add file3
    with open(os.path.join(tmpdir, "file1.py"), "w") as f:
        f.write("# File 1 MODIFIED\n" + "x = 999\n" * 20)
    os.remove(os.path.join(tmpdir, "file2.py"))
    with open(os.path.join(tmpdir, "file3.py"), "w") as f:
        f.write("# File 3 NEW\n" + "z = 3\n" * 20)

    update(tmpdir, palace)

    # Verify: file2 drawers gone, file1 updated, file3 added
    results = col.get(include=["metadatas"], limit=10000)
    source_files = {m["source_file"] for m in results["metadatas"]}
    assert os.path.join(tmpdir, "file2.py") not in source_files, "Deleted file should be removed"
    assert os.path.join(tmpdir, "file3.py") in source_files, "New file should be added"
    assert os.path.join(tmpdir, "file1.py") in source_files, "Modified file should be re-mined"

    shutil.rmtree(tmpdir)
    shutil.rmtree(palace)
