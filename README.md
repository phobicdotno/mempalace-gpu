# mempalace-gpu

> GPU-accelerated fork of [milla-jovovich/mempalace](https://github.com/milla-jovovich/mempalace)

This fork adds CUDA/GPU-accelerated embeddings and batch processing to MemPalace. For documentation on MemPalace itself (palace structure, AAAK dialect, MCP tools, benchmarks), see the [upstream README](https://github.com/milla-jovovich/mempalace#readme).

---

## What this fork adds

### GPU-accelerated embeddings

Embeddings are computed via `sentence-transformers` on CUDA when available, falling back to ChromaDB's default CPU/ONNX model when not.

```bash
pip install mempalace[gpu]          # installs sentence-transformers + torch
mempalace mine ~/myproject --device cuda
```

`--device` options: `auto` (default, detect GPU), `cuda`, `cpu`

Also configurable via `MEMPALACE_DEVICE` env var or `"device"` in `~/.mempalace/config.json`.

### Batch processing

`collection.add()` calls are batched (100 documents per call instead of 1), reducing ChromaDB overhead regardless of CPU or GPU mode.

### Self-update MCP tool

The MCP server includes a `mempalace_self_update` tool that pulls the latest version from PyPI, callable directly from your AI assistant.

---

## Performance

Tested on two real-world codebases. Same files, same drawers — only the device changes.

| Test | Files | Drawers | Size | CPU | GPU | Speedup |
|------|-------|---------|------|-----|-----|---------|
| Large mixed codebase (JS/TS/Dart/Python/HTML) | 118 | 13,673 | ~1.7 GB | 156.7s | 26.3s | **6.0x** |
| Medium Flutter app (Dart/YAML/JSON) | 145 | 2,906 | ~85 MB | 37.3s | 10.7s | **3.5x** |

Speedup scales with drawer count. More chunks = more embedding work = bigger GPU advantage.

---

## Installation

```bash
# Clone this fork
git clone https://github.com/phobicdotno/mempalace-gpu.git
cd mempalace-gpu

# Install with GPU support
pip install -e ".[gpu]"

# Or without GPU (still gets batch processing)
pip install -e .
```

### Staying in sync with upstream

```bash
git remote add upstream https://github.com/milla-jovovich/mempalace.git
git fetch upstream
git merge upstream/main
```

---

## Changes from upstream

| File | Change |
|------|--------|
| `mempalace/embeddings.py` | **New** -- shared embedding function factory, device detection, batch flush |
| `mempalace/miner.py` | Batched `collection.add()`, content hashing, `update()` command |
| `mempalace/convo_miner.py` | Batched `collection.add()` |
| `mempalace/config.py` | `device` property (auto/cuda/cpu) |
| `mempalace/cli.py` | `--device` flag, `update` subcommand |
| `mempalace/mcp_server.py` | `mempalace_self_update` tool, shared embeddings |
| `mempalace/searcher.py` | Shared embedding function for vector compatibility |
| `mempalace/layers.py` | Shared embedding function |
| `mempalace/palace_graph.py` | Shared embedding function |
| `pyproject.toml` | `gpu` optional dependency group |

All other files are unmodified from upstream. Existing palaces remain compatible.

---

## License

MIT -- same as upstream.
