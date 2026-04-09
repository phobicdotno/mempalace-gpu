# mempalace-gpu

> GPU-accelerated fork of [milla-jovovich/mempalace](https://github.com/milla-jovovich/mempalace)

This fork adds GPU-accelerated embeddings and batch processing to MemPalace. Supports **NVIDIA (CUDA)**, **AMD (ROCm)**, and **Apple Silicon (MPS)**. For documentation on MemPalace itself (palace structure, AAAK dialect, MCP tools, benchmarks), see the [upstream README](https://github.com/milla-jovovich/mempalace#readme).

---

## What this fork adds

### GPU-accelerated embeddings

Embeddings are computed via `sentence-transformers` on GPU when available, falling back to ChromaDB's default CPU/ONNX model when not.

```bash
mempalace mine ~/myproject --device auto    # auto-detect best GPU
mempalace mine ~/myproject --device cuda    # NVIDIA
mempalace mine ~/myproject --device rocm    # AMD
mempalace mine ~/myproject --device mps     # Apple Silicon (M1-M5)
mempalace mine ~/myproject --device cpu     # force CPU
```

Also configurable via `MEMPALACE_DEVICE` env var or `"device"` in `~/.mempalace/config.json`.

### Batch processing

`collection.add()` calls are batched (100 documents per call instead of 1), reducing ChromaDB overhead regardless of CPU or GPU mode.

### Self-update MCP tool

The MCP server includes a `mempalace_self_update` tool that pulls the latest version from PyPI, callable directly from your AI assistant.

---

## Performance

Tested on two real-world codebases. GPU: **NVIDIA GeForce RTX 4080 SUPER**. Same files, same drawers — only the device changes.

| Test | Files | Drawers | Size | CPU | RTX 4080 SUPER | Speedup |
|------|-------|---------|------|-----|----------------|---------|
| Large mixed codebase (JS/TS/Dart/Python/HTML) | 118 | 13,673 | ~1.7 GB | 156.7s | 26.3s | **6.0x** |
| Medium Flutter app (Dart/YAML/JSON) | 145 | 2,906 | ~85 MB | 37.3s | 10.7s | **3.5x** |

Speedup scales with drawer count. More chunks = more embedding work = bigger GPU advantage. Results will vary by GPU — expect similar gains on any modern NVIDIA/AMD/Apple Silicon GPU.

---

## Installation

```bash
pip install mempalace-gpu
claude mcp add mempalace-gpu -- python -m mempalace.mcp_server
```

Restart Claude Code — `mempalace-gpu` appears in `/plugin` with all tools. Works on NVIDIA, AMD, and Apple Silicon — GPU is auto-detected.

### AMD (ROCm) note

AMD GPUs need the ROCm version of PyTorch installed first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install mempalace-gpu
```

### Your data is safe

Installing or upgrading `mempalace-gpu` only replaces the Python code. Your mined data lives in `~/.mempalace/palace/` (ChromaDB files) and is never touched. Existing palaces remain fully compatible.

### Development install

```bash
git clone https://github.com/phobicdotno/mempalace-gpu.git
cd mempalace-gpu
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
| `mempalace/embeddings.py` | **New** -- GPU detection (NVIDIA/AMD/Apple), embedding factory, batch flush |
| `mempalace/miner.py` | Batched `collection.add()`, content hashing, `update()` command |
| `mempalace/convo_miner.py` | Batched `collection.add()` |
| `mempalace/config.py` | `device` property (auto/cuda/rocm/mps/cpu) |
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
