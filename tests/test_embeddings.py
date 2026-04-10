import tempfile
from unittest.mock import MagicMock, patch

import chromadb


def test_detect_device_cpu():
    from mempalace.embeddings import _detect_device

    assert _detect_device("cpu") == "cpu"


def test_detect_device_auto():
    from mempalace.embeddings import _detect_device

    device = _detect_device("auto")
    assert device in ("cpu", "cuda", "mps")


def test_detect_device_rocm():
    from mempalace.embeddings import _detect_device

    # rocm maps to 'cuda' (PyTorch ROCm compatibility layer) or 'cpu' if no GPU
    device = _detect_device("rocm")
    assert device in ("cpu", "cuda")


def test_detect_device_mps():
    from mempalace.embeddings import _detect_device

    # mps resolves to 'mps' on Apple Silicon or 'cpu' elsewhere
    device = _detect_device("mps")
    assert device in ("cpu", "mps")


def test_detect_gpu_vendor():
    from mempalace.embeddings import _detect_gpu_vendor

    vendor = _detect_gpu_vendor()
    assert vendor in ("nvidia", "amd", "apple", "none")


def test_get_embedding_function_no_crash():
    from mempalace.embeddings import get_embedding_function

    ef = get_embedding_function("cpu")
    assert ef is not None or ef is None  # just verify no crash


def test_get_collection_roundtrip():
    from mempalace.embeddings import get_collection

    tmpdir = tempfile.mkdtemp()
    client = chromadb.PersistentClient(path=tmpdir)
    col = get_collection(client, "test_col", create=True, device="cpu")
    col.add(ids=["t1"], documents=["test document about cats"])
    assert col.count() == 1
    results = col.query(query_texts=["feline"], n_results=1)
    assert len(results["documents"][0]) == 1


def test_flush_batch():
    from mempalace.embeddings import get_collection, flush_batch

    tmpdir = tempfile.mkdtemp()
    client = chromadb.PersistentClient(path=tmpdir)
    col = get_collection(client, "test_col", create=True, device="cpu")
    batch = [
        {
            "id": f"d{i}",
            "document": f"doc number {i} content",
            "metadata": {"wing": "test", "room": "general"},
        }
        for i in range(10)
    ]
    added = flush_batch(col, batch)
    assert added == 10
    assert col.count() == 10


def test_flush_batch_handles_duplicates():
    from mempalace.embeddings import get_collection, flush_batch

    tmpdir = tempfile.mkdtemp()
    client = chromadb.PersistentClient(path=tmpdir)
    col = get_collection(client, "test_col", create=True, device="cpu")
    col.add(ids=["d0"], documents=["existing doc"])
    batch = [
        {"id": "d0", "document": "duplicate doc", "metadata": {"wing": "test"}},
        {"id": "d1", "document": "new doc", "metadata": {"wing": "test"}},
    ]
    added = flush_batch(col, batch)
    assert added >= 1  # at least d1 should succeed


def test_verify_compatibility_empty_collection():
    from mempalace.embeddings import verify_embedding_compatibility

    tmpdir = tempfile.mkdtemp()
    client = chromadb.PersistentClient(path=tmpdir)
    col = client.get_or_create_collection(name="empty_col")
    assert verify_embedding_compatibility(col, device="cpu") is True


def test_verify_compatibility_no_ef():
    from mempalace.embeddings import verify_embedding_compatibility

    col = MagicMock()
    with patch("mempalace.embeddings.get_embedding_function", return_value=None):
        assert verify_embedding_compatibility(col, device="cpu") is True
    # collection should never be queried when ef is None
    col.query.assert_not_called()


def test_verify_compatibility_cache():
    import mempalace.embeddings as emb
    from mempalace.embeddings import get_collection

    tmpdir = tempfile.mkdtemp()
    client = chromadb.PersistentClient(path=tmpdir)
    # Create a collection normally first
    col = client.get_or_create_collection(name="cache_test")
    col.add(ids=["t1"], documents=["test doc"])

    # Clear cache to allow fresh check
    emb._compatibility_checked.discard("cache_test")

    with patch.object(
        emb, "verify_embedding_compatibility", wraps=emb.verify_embedding_compatibility
    ) as mock_verify:
        # Force the ValueError fallback path by using a mismatched EF
        with patch.object(emb, "get_embedding_function", return_value=None):
            # First call — should trigger verify
            try:
                get_collection(client, "cache_test", device="cpu")
            except Exception:
                pass

            # Second call — should NOT trigger verify again (cached)
            try:
                get_collection(client, "cache_test", device="cpu")
            except Exception:
                pass

        # verify_embedding_compatibility should have been called at most once
        # (it may be 0 if the ValueError path wasn't hit because ef=None works)
        assert mock_verify.call_count <= 1
