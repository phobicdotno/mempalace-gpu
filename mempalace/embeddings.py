"""
Shared embedding function factory for MemPalace.

Creates a SentenceTransformer embedding function with CUDA support when available.
Falls back to ChromaDB's default ONNX embedder when sentence-transformers is not installed.
"""

import logging

logger = logging.getLogger("mempalace.embeddings")

DEFAULT_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 100

_cached_ef = None
_cached_device = None


def _detect_device(preference: str = "auto") -> str:
    if preference == "cpu":
        return "cpu"
    if preference in ("cuda", "auto"):
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
    return "cpu"


def get_embedding_function(device: str = "auto"):
    """Get or create a cached embedding function for ChromaDB.

    Returns SentenceTransformerEmbeddingFunction when available, None otherwise.
    """
    global _cached_ef, _cached_device
    resolved = _detect_device(device)
    if _cached_ef is not None and _cached_device == resolved:
        return _cached_ef

    try:
        from chromadb.utils import embedding_functions

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=DEFAULT_MODEL,
            device=resolved,
        )
        logger.info(f"Embeddings: SentenceTransformer on {resolved}")
        _cached_ef = ef
        _cached_device = resolved
        return ef
    except Exception:
        logger.info("Embeddings: ChromaDB default (ONNX/CPU)")
        _cached_ef = None
        _cached_device = resolved
        return None


def init(device: str = "auto"):
    """Pre-warm the embedding function cache. Call once at startup."""
    get_embedding_function(device)


def get_collection(client, name: str, create: bool = False, device: str = "auto"):
    """Get or create a ChromaDB collection with the shared embedding function."""
    ef = get_embedding_function(device)
    kwargs = {"name": name}
    if ef is not None:
        kwargs["embedding_function"] = ef
    if create:
        return client.get_or_create_collection(**kwargs)
    return client.get_collection(**kwargs)


def flush_batch(collection, batch: list) -> int:
    """Add a batch of drawers to ChromaDB in one call.

    Falls back to one-at-a-time on duplicate errors. Returns count added.
    """
    if not batch:
        return 0
    try:
        collection.add(
            ids=[d["id"] for d in batch],
            documents=[d["document"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        return len(batch)
    except Exception as e:
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            added = 0
            for d in batch:
                try:
                    collection.add(
                        ids=[d["id"]], documents=[d["document"]], metadatas=[d["metadata"]]
                    )
                    added += 1
                except Exception:
                    pass
            return added
        raise
