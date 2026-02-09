# ClawMem

Hybrid agent memory system built on [QMD](https://github.com/tobi/qmd)'s retrieval substrate (BM25 + vectors + RRF + query expansion + cross-encoder reranking), layered with [SAME](https://github.com/sgx-labs/statelessagent)-derived composite scoring (recency decay, confidence, content-type half-lives), [MAGMA](https://arxiv.org/abs/2501.13956)-inspired intent classification and multi-graph traversal (semantic, temporal, causal beam search), and [A-MEM](https://arxiv.org/abs/2510.02178) self-evolving memory notes with automatic keyword/tag/context enrichment and inter-document link generation. Designed for [OpenClaw](https://github.com/openclaw/openclaw) and Claude Code.

TypeScript on Bun. ~12,700 lines across 30 source files. 103 tests.

## What It Does

ClawMem turns your markdown notes, project docs, and research dumps into an intelligent memory layer for Claude Code. It automatically:

- **Surfaces relevant context** on every prompt (context-surfacing hook)
- **Bootstraps sessions** with your profile, latest handoff, recent decisions, and stale notes
- **Captures decisions** from session transcripts using a local GGUF observer model
- **Generates handoffs** at session end so the next session can pick up where you left off
- **Learns what matters** via a feedback loop that boosts referenced notes and decays unused ones
- **Guards against prompt injection** in surfaced content
- **Classifies query intent** (WHY / WHEN / ENTITY / WHAT) to weight search strategies
- **Traverses multi-graphs** (semantic, temporal, causal) via adaptive beam search
- **Evolves memory metadata** as new documents create or refine connections
- **Infers causal relationships** between facts extracted from session observations
- **Syncs project issues** from Beads issue trackers into searchable memory

All context injection runs through Claude Code's hook system — no API keys needed, no cloud services, fully local.

## Architecture

```
Claude Code Session
    │
    ├─ UserPromptSubmit ──→ context-surfacing hook
    │                       search vault → composite score → sanitize → inject
    │
    ├─ SessionStart ──────→ session-bootstrap hook
    │                       profile + handoff + decisions + stale → inject
    │
    └─ Stop ──────────────→ decision-extractor + handoff-generator + feedback-loop
    │                       + causal inference from Observer facts
    ▼
┌─────────────────────────────────────────────────────────┐
│  Intent-Aware Search Layer                               │
│  Query → Intent Classification (WHY/WHEN/ENTITY/WHAT)    │
│  → Intent-Weighted RRF → Graph Expansion → Reranking     │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│  Multi-Graph Memory Layer                                │
│  Semantic Graph (vector similarity > 0.7 threshold)      │
│  Temporal Backbone (chronological document ordering)     │
│  Causal Graph (LLM-inferred cause → effect chains)       │
│  A-MEM Notes (keywords, tags, contextual descriptions)   │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│  SAME Composite Scoring Layer                            │
│  compositeScore = w.search × searchScore                 │
│                 + w.recency × recencyDecay               │
│                 + w.confidence × confidence               │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│  QMD Search Backend (forked)                             │
│  BM25 (FTS5) + Vector (sqlite-vec 768d) + Query         │
│  Expansion + RRF (k=60) + Cross-Encoder Reranking        │
└─────────────────────────────────────────────────────────┘
```

## Install

### Prerequisites

- [Bun](https://bun.sh) v1.0+
- SQLite with FTS5 support (included with Bun)

### Setup

```bash
git clone https://github.com/yoloshii/clawmem.git ~/clawmem
cd ~/clawmem
bun install

# Add to PATH
ln -sf ~/clawmem/bin/clawmem ~/.bun/bin/clawmem
```

### Quick Start (Bootstrap)

One command to set up a vault:

```bash
# Initialize, index, embed, install hooks, register MCP
./bin/clawmem bootstrap ~/notes --name notes

# Or step by step:
./bin/clawmem init
./bin/clawmem collection add ~/notes --name notes
./bin/clawmem update --embed
./bin/clawmem setup hooks
./bin/clawmem setup mcp
```

### GPU Services

ClawMem uses three lightweight `llama-server` (llama.cpp) instances for neural inference. Run them on your local GPU — total VRAM is ~4.5GB, fitting comfortably alongside other workloads on any modern NVIDIA card.

| Service | Port | Model | VRAM | Purpose |
|---|---|---|---|---|
| Embedding | 8088 | granite-embedding-278m-multilingual-Q6_K | ~400MB | Vector search, indexing, context-surfacing |
| LLM | 8089 | qmd-query-expansion-1.7B-q4_k_m | ~2.2GB | Intent classification, query expansion, A-MEM |
| Reranker | 8090 | qwen3-reranker-0.6B-Q8_0 | ~1.3GB | Cross-encoder reranking (query, intent_search) |

The `bin/clawmem` wrapper defaults to `localhost:8088/8089/8090`. Start the three servers, and ClawMem connects automatically.

#### Remote GPU (optional)

If your GPU lives on a separate machine, point the env vars at it:

```bash
export CLAWMEM_EMBED_URL=http://gpu-host:8088
export CLAWMEM_LLM_URL=http://gpu-host:8089
export CLAWMEM_RERANK_URL=http://gpu-host:8090
```

For remote setups, set `CLAWMEM_NO_LOCAL_MODELS=true` to prevent `node-llama-cpp` from auto-downloading multi-GB model files if a server is unreachable. Operations fail fast instead of silently falling back.

#### CPU-Only Mode (no GPU)

Without a GPU, unset the endpoint vars:

```bash
unset CLAWMEM_EMBED_URL CLAWMEM_LLM_URL CLAWMEM_RERANK_URL
```

`node-llama-cpp` will auto-download GGUF models on first use (~1.1GB LLM + ~600MB reranker). CPU inference works but is much slower — GPU is strongly recommended for responsive query expansion and reranking.

**Note:** Embedding requires a running `llama-server --embeddings` instance (local or remote) — there is no in-process fallback for embedding.

### Embedding Server

Embeddings use [granite-embedding-278m-multilingual-Q6_K](https://huggingface.co/bartowski/granite-embedding-278m-multilingual-GGUF) via `llama-server --embeddings` on port 8088. ClawMem calls the OpenAI-compatible `/v1/embeddings` endpoint.

**Model specs:**
- Size: 226MB, Dimensions: 768
- Performance: **~5ms per fragment**, ~200 fragments/sec on RTX 3090
- Benchmark: 180 docs, 6,335 fragments in **67 seconds**

**Known issues:**
- Native context is only 512 tokens (~1100 chars after formatting)
- Client-side truncation at 1100 chars prevents HTTP 500 errors
- Code fragments have higher token density (occasionally fail even at 1100 chars)

**Server setup:**

```bash
# Download model
wget https://huggingface.co/bartowski/granite-embedding-278m-multilingual-GGUF/resolve/main/granite-embedding-278m-multilingual-Q6_K.gguf

# Start llama-server in embedding mode
llama-server -m granite-embedding-278m-multilingual-Q6_K.gguf \
  --embeddings --port 8088 --host 0.0.0.0 \
  --no-mmap -ngl 99 -c 2048 --batch-size 2048
```

**Verify:** `curl http://localhost:8088/v1/embeddings -d '{"input":"test","model":"embedding"}' -H 'Content-Type: application/json'` should return a 768-dimensional vector.

To embed your vault:

```bash
./bin/clawmem embed  # Embeds all documents via the embedding server
```

### LLM Server

Intent classification, query expansion, and A-MEM extraction use [qmd-query-expansion-1.7B](https://huggingface.co/tobil/qmd-query-expansion-1.7B-gguf) — a Qwen3-1.7B finetuned by QMD specifically for generating search expansion terms (hyde, lexical, and vector variants). ~1.1GB at q4_k_m quantization, served via `llama-server` on port 8089.

**Without a server:** If `CLAWMEM_LLM_URL` is unset, `node-llama-cpp` auto-downloads the model on first use.

**Performance (RTX 3090):**
- Intent classification: **27ms**
- Query expansion: **333 tok/s**
- VRAM: ~2.2-2.8GB depending on quantization

**Qwen3 /no_think flag:** Qwen3 uses thinking tokens by default. ClawMem appends `/no_think` to all prompts automatically to get structured output in the `content` field.

**Intent classification:** Uses a dual-path approach:
1. **Heuristic regex classifier** (instant) — handles strong signals (why/when/who keywords) with 0.8+ confidence
2. **LLM refinement** (27ms on GPU) — only for ambiguous queries below 0.8 confidence

**Server setup:**

```bash
# Download the finetuned model
wget https://huggingface.co/tobil/qmd-query-expansion-1.7B-gguf/resolve/main/qmd-query-expansion-1.7B-q4_k_m.gguf

# Start llama-server for LLM inference
llama-server -m qmd-query-expansion-1.7B-q4_k_m.gguf \
  --port 8089 --host 0.0.0.0 \
  -ngl 99 -c 4096 --batch-size 512
```

### Reranker Server

Cross-encoder reranking for `query` and `intent_search` pipelines using [qwen3-reranker-0.6B-Q8_0](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF) on port 8090. ClawMem calls the `/v1/rerank` endpoint (or falls back to scoring via `/v1/completions` for compatible servers).

**Model specs:**
- Size: ~600MB (Q8_0), VRAM: ~1.3GB on GPU
- Scores each candidate against the original query (cross-encoder architecture)
- `query` pipeline: 4000 char context per doc (deep reranking); `intent_search`: 200 char context per doc (fast reranking)

**Without a server:** If `CLAWMEM_RERANK_URL` is unset, `node-llama-cpp` auto-downloads the model (~600MB) on first use.

**Server setup:**

```bash
# Download model
wget https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/Qwen3-Reranker-0.6B-Q8_0.gguf

# Start llama-server for reranking
llama-server -m Qwen3-Reranker-0.6B-Q8_0.gguf \
  --port 8090 --host 0.0.0.0 \
  -ngl 99 -c 2048 --batch-size 512 --reranking
```

### MCP Server

ClawMem exposes 18 tools via the [Model Context Protocol](https://modelcontextprotocol.io). Any MCP-compatible client can use it.

**Claude Code (automatic):**

```bash
./bin/clawmem setup mcp   # Registers in ~/.claude.json
```

**Manual (any MCP client):**

Add to your MCP config (e.g. `~/.claude.json`, `claude_desktop_config.json`, or your client's equivalent):

```json
{
  "mcpServers": {
    "clawmem": {
      "command": "/absolute/path/to/clawmem/bin/clawmem",
      "args": ["mcp"]
    }
  }
}
```

The server runs via stdio — no network port needed. The `bin/clawmem` wrapper sets the GPU endpoint env vars automatically.

**Verify:** After registering, your client should see 18 tools including `search`, `vsearch`, `query`, `intent_search`, etc.

### Verify Installation

```bash
./bin/clawmem doctor   # Full health check
./bin/clawmem status   # Quick index status
bun test               # Run test suite (103 tests)
```

## CLI Reference

```
clawmem init                                    Create DB + config
clawmem bootstrap <vault> [--name N] [--skip-embed]  One-command setup
clawmem collection add <path> --name <name>     Add a collection
clawmem collection list                         List collections
clawmem collection remove <name>                Remove a collection

clawmem update [--pull] [--embed]               Incremental re-scan
clawmem embed [-f]                              Generate fragment embeddings
clawmem reindex [--force]                       Full re-index
clawmem watch                                   File watcher daemon

clawmem search <query> [-n N] [--json]          BM25 keyword search
clawmem vsearch <query> [-n N] [--json]         Vector semantic search
clawmem query <query> [-n N] [--json]           Full hybrid pipeline

clawmem profile                                 Show user profile
clawmem profile rebuild                         Force profile rebuild
clawmem update-context                          Regenerate per-folder CLAUDE.md

clawmem budget [--session ID]                   Token utilization
clawmem log [--last N]                          Session history
clawmem hook <name>                             Manual hook trigger

clawmem install-service [--enable] [--remove]   Systemd watcher service
clawmem setup hooks [--remove]                  Install/remove Claude Code hooks
clawmem setup mcp [--remove]                    Register/remove MCP server
clawmem mcp                                     Start stdio MCP server
clawmem doctor                                  Full health check
clawmem status                                  Quick index status
```

## MCP Tools (18)

Registered by `clawmem setup mcp`. Available to any MCP-compatible client.

| Tool | Description |
|---|---|
| `__IMPORTANT` | Workflow guide: search compact → review → multi_get |

### Core Search & Retrieval

| Tool | Description |
|---|---|
| `search` | BM25 keyword search with composite scoring + compact mode |
| `vsearch` | Vector semantic search with composite scoring + compact mode |
| `query` | Full hybrid pipeline with composite scoring + compact mode |
| `get` | Retrieve single document by path or docid |
| `multi_get` | Retrieve multiple docs by glob or comma-separated list |
| `find_similar` | Find notes similar to a reference document |

### Intent-Aware Search

| Tool | Description |
|---|---|
| `intent_search` | Intent-classified search with graph expansion and reranking |

**Pipeline:** Query → Intent Classification → BM25 + Vector → Intent-Weighted RRF → Graph Expansion (WHY/ENTITY intents) → Cross-Encoder Reranking → Composite Scoring

### Multi-Graph & Causal

| Tool | Description |
|---|---|
| `build_graphs` | Build temporal and/or semantic graphs from document corpus |
| `find_causal_links` | Traverse causal chains (causes / caused_by / both) up to N hops |
| `memory_evolution_status` | Show how a document's A-MEM metadata evolved over time |

### Beads Integration

| Tool | Description |
|---|---|
| `beads_sync` | Sync `.beads/beads.jsonl` into memory: creates docs, bridges deps to `memory_relations`, runs A-MEM enrichment |

### Memory Management

| Tool | Description |
|---|---|
| `memory_forget` | Search → deactivate closest match (with audit trail) |
| `status` | Index health with content type distribution |
| `reindex` | Trigger vault re-scan |
| `index_stats` | Detailed stats: types, staleness, access counts, sessions |
| `session_log` | Recent sessions with handoff info |
| `profile` | Current static + dynamic user profile |

### Compact Mode

`search`, `vsearch`, and `query` accept `compact: true` to return `{ id, path, title, score, snippet, content_type, fragment }` instead of full content. Saves ~5x tokens for initial filtering.

## Hooks (Claude Code Integration)

Six hooks auto-installed by `clawmem setup hooks`:

| Hook | Event | What It Does |
|---|---|---|
| `context-surfacing` | UserPromptSubmit | Searches vault, scores, sanitizes, injects relevant notes (800 token budget) |
| `session-bootstrap` | SessionStart | Injects profile + handoff + decisions + stale notes (2000 token budget) |
| `staleness-check` | SessionStart | Flags documents needing review |
| `decision-extractor` | Stop | GGUF observer extracts structured decisions + infers causal links between facts |
| `handoff-generator` | Stop | GGUF observer generates rich handoff, regex fallback |
| `feedback-loop` | Stop | Silently boosts referenced notes, decays unused ones |

## Search Pipeline

```
User Query → Intent Classification (WHY/WHEN/ENTITY/WHAT)
  → BM25 + Vector Search (parallel)
  → Intent-Weighted RRF (boost BM25 for WHEN, boost vector for WHY)
  → Graph Expansion (WHY/ENTITY: adaptive beam search over multi-graph)
  → Cross-Encoder Reranking (0.6B GGUF)
  → SAME Composite Scoring (search × 0.5 + recency × 0.25 + confidence × 0.25)
  → Ranked Results
```

### Multi-Graph Traversal

For WHY and ENTITY queries, the search pipeline expands results through the memory graph:

1. Start from top-10 baseline results as anchor nodes
2. For each frontier node: get neighbors via any relation type
3. Score transitions: `λ1·structure + λ2·semantic_affinity`
4. Apply decay: `new_score = parent_score * γ + transition_score`
5. Keep top-k (beam search), repeat until max depth or budget

**Graph types:**
- **Semantic** — vector similarity edges (threshold > 0.7)
- **Temporal** — chronological document ordering
- **Causal** — LLM-inferred cause→effect from Observer facts + Beads `blocks`/`waits-for` deps
- **Supporting** — LLM-analyzed document relationships + Beads `discovered-from` deps
- **Contradicts** — LLM-analyzed document relationships

### Content Type Scoring

| Type | Half-life | Baseline | Notes |
|---|---|---|---|
| `decision` | ∞ | 0.85 | Never decays |
| `hub` | ∞ | 0.80 | Never decays |
| `research` | 90 days | 0.70 | |
| `project` | 120 days | 0.65 | |
| `handoff` | 30 days | 0.60 | Fast decay — most recent matters |
| `progress` | 45 days | 0.50 | |
| `note` | 60 days | 0.50 | Default |

Content types are inferred from frontmatter or file path patterns.

## Features

### A-MEM (Adaptive Memory Evolution)

Documents are automatically enriched with structured metadata when indexed:
- **Keywords** (3-7 specific terms)
- **Tags** (3-5 broad categories)
- **Context** (1-2 sentence description)

When new documents create links, neighboring documents' metadata evolves — keywords merge, context updates, and the evolution history is tracked with version numbers and reasoning.

### Causal Inference

The decision-extractor hook analyzes Observer facts for causal relationships. When multiple facts exist in an observation, an LLM identifies cause→effect pairs (confidence ≥ 0.6). Causal chains can be queried via `find_causal_links` with multi-hop traversal using recursive CTEs.

### Beads Integration

Projects using [Beads](https://github.com/steveyegge/beads) (v0.49+) issue tracking are fully integrated into the MAGMA memory graph:

- **Auto-sync**: Watcher detects `.beads/beads.jsonl` changes → `syncBeadsIssues()` creates markdown docs in `beads` collection
- **Dependency bridging**: Beads deps map to `memory_relations` edges — `blocks`→causal, `discovered-from`→supporting, `relates-to`→semantic, `waits-for`→causal. Tagged `{origin: "beads"}` for traceability.
- **A-MEM enrichment**: New beads docs get full `postIndexEnrich()` — memory note construction, semantic/entity link generation, memory evolution
- **Graph traversal**: `intent_search` and `find_causal_links` traverse beads dependency edges alongside observation-inferred causal chains

`beads_sync` MCP tool for manual sync; watcher handles routine operations automatically.

### Fragment-Level Embedding

Documents are split into semantic fragments (sections, lists, code blocks, frontmatter, facts) and each fragment gets its own vector embedding. Full-doc embedding is preserved for broad-match queries.

### Local Observer Agent

Uses the LLM server (shared with query expansion and intent classification) to extract structured observations from session transcripts: type, title, facts, narrative, concepts, files read/modified. Falls back to regex patterns if the model is unavailable.

### User Profile

Two-tier auto-curated profile extracted from your decisions and hub documents:
- **Static**: persistent facts (Levenshtein-deduplicated)
- **Dynamic**: recent session context

Injected at session start for instant personalization.

### Prompt Injection Filtering

Five detection layers protect injected content: legacy string patterns, role injection regex, instruction override patterns, delimiter injection, and unicode obfuscation detection. Filtered results are skipped entirely (no placeholder tokens wasted).

### Consolidation Worker

Optional background process that enriches documents missing A-MEM metadata. Runs on a configurable interval, processing 3 documents per tick. Non-blocking (Timer.unref).

### Per-Folder CLAUDE.md Generation

Automatically generates context sections in per-folder CLAUDE.md files from recent decisions and session activity related to that directory.

### Feedback Loop

Notes referenced by the agent during a session get boosted (`access_count++`). Unreferenced notes decay via recency. Over time, useful notes rise and noise fades.

## Feature Flags

| Variable | Default | Effect |
|---|---|---|
| `CLAWMEM_ENABLE_AMEM` | enabled | A-MEM note construction + link generation during indexing |
| `CLAWMEM_ENABLE_CONSOLIDATION` | disabled | Background worker for backlog A-MEM enrichment |
| `CLAWMEM_CONSOLIDATION_INTERVAL` | 300000 | Worker interval in ms (min 15000) |
| `CLAWMEM_EMBED_URL` | `http://localhost:8088` | Embedding server URL. No in-process fallback — a `llama-server --embeddings` instance is required. |
| `CLAWMEM_LLM_URL` | `http://localhost:8089` | LLM server URL for intent/query/A-MEM. Without it, falls to `node-llama-cpp` (if allowed). |
| `CLAWMEM_RERANK_URL` | `http://localhost:8090` | Reranker server URL. Without it, falls to `node-llama-cpp` (if allowed). |
| `CLAWMEM_NO_LOCAL_MODELS` | `false` | Block `node-llama-cpp` from auto-downloading GGUF models. Set `true` for remote-only setups where you want fail-fast on unreachable endpoints. |

## Configuration

### Collection Config

`~/.config/clawmem/config.yaml`:

```yaml
collections:
  notes:
    path: /home/user/notes
    pattern: "**/*.md"
    autoEmbed: true
  docs:
    path: /home/user/docs
    pattern: "**/*.md"
    update: "git pull"
directoryContext: false  # opt-in per-folder CLAUDE.md generation
```

### Database

`~/.cache/clawmem/index.sqlite` — single SQLite file with FTS5 + sqlite-vec extensions.

### Frontmatter

Parsed via `gray-matter`. Supported fields:

```yaml
---
title: "Document Title"
tags: [tag1, tag2]
domain: "infrastructure"
workstream: "project-name"
content_type: "decision"   # decision|hub|research|project|handoff|progress|note
review_by: "2026-03-01"
---
```

## Suggested Memory Filesystem (OpenClaw)

For agent systems using ClawMem as their memory backend, this 5-layer structure maps cleanly to ClawMem collections:

```
~/.openclaw/workspace/              ← Collection: "workspace"
├── MEMORY.md                        # Layer 1: Workspace long-term (human-curated)
├── memory/                          # Layer 2: Workspace daily (session logs)
│   ├── 2026-02-05.md
│   └── 2026-02-06.md
├── _clawmem/                        # Auto-generated — DO NOT EDIT
│   ├── decisions/                   #   Structured decisions from session transcripts
│   ├── handoffs/                    #   Session handoffs (summary, next steps, files)
│   └── profile.md                   #   Auto-curated user profile (static + dynamic)
└── ...                              # Behavioral docs, configs, etc.

~/Projects/<project>/               ← Collection: "<project>"
├── .beads/                          # Beads issue tracker (auto-synced to memory graph)
│   └── beads.jsonl
├── MEMORY.md                        # Layer 3: Project long-term (human-curated)
├── memory/                          # Layer 4: Project daily (session logs)
│   └── 2026-02-06.md
├── _clawmem/                        # Auto-generated per-project
│   ├── beads/                       #   Beads issues as searchable markdown
│   ├── decisions/
│   ├── handoffs/
│   └── profile.md
├── research/                        # Layer 5: Research dumps (fragment-embedded)
│   └── 2026-02-06-topic-slug.md
├── CLAUDE.md                        # May include auto-generated ClawMem context section
├── src/
└── README.md
```

### Layer Mapping

| Layer | Path | Manual/Auto | ClawMem Role |
|---|---|---|---|
| 1. Workspace Long-Term | `MEMORY.md` | Manual | Indexed, searched, profile supplements |
| 2. Workspace Daily | `memory/*.md` | Manual | Indexed, searched, handoff auto-generated |
| 3. Project Long-Term | `Projects/X/MEMORY.md` | Manual | Indexed per-collection, decisions auto-captured |
| 4. Project Daily | `Projects/X/memory/*.md` | Manual | Indexed, searched |
| 5. Research Dumps | `Projects/X/research/*.md` | Manual | Fragment-embedded for granular retrieval |
| Auto: Decisions | `_clawmem/decisions/*.md` | Auto | Observer-extracted from transcripts |
| Auto: Handoffs | `_clawmem/handoffs/*.md` | Auto | Session summaries with next steps |
| Auto: Profile | `_clawmem/profile.md` | Auto | Static facts + dynamic context |

Manual layers benefit from periodic re-indexing — a cron job running `clawmem update --embed` keeps the index fresh for content edited outside of watched directories.

### Setup for OpenClaw

```bash
# Bootstrap workspace collection
./bin/clawmem bootstrap ~/.openclaw/workspace --name workspace

# Bootstrap each project
./bin/clawmem bootstrap ~/Projects/my-project --name my-project

# Enable auto-embed for real-time indexing
# Edit ~/.config/clawmem/config.yaml → autoEmbed: true

# Install watcher as systemd service
./bin/clawmem install-service --enable
```

## Dependencies

| Package | Purpose |
|---|---|
| `@modelcontextprotocol/sdk` | MCP server |
| `gray-matter` | YAML frontmatter parsing |
| `node-llama-cpp` | GGUF model inference (reranking, query expansion, A-MEM) |
| `sqlite-vec` | Vector similarity extension |
| `yaml` | Config parsing |
| `zod` | MCP schema validation |

## Deployment

Three-tier retrieval architecture: infrastructure (watcher + embed timer) → hooks (~90%) → agent MCP (~10%). Best with three `llama-server` instances (embedding, LLM, reranker) on a local or remote GPU. See GPU Services section above for setup.

Key services: `clawmem-watcher` (auto-index on file change + beads sync), `clawmem-embed` timer (daily embedding sweep), Claude Code hooks (6 hooks for context injection + extraction).

## Acknowledgments

Built on the shoulders of:

- [QMD](https://github.com/tobi/qmd) — search backend (BM25 + vectors + RRF + reranking)
- [SAME](https://github.com/sgx-labs/statelessagent) — agent memory concepts (recency decay, confidence scoring, session tracking)
- [supermemory](https://github.com/supermemoryai/clawdbot-supermemory) — hook patterns and context surfacing ideas
- [claude-mem](https://github.com/thedotmack/claude-mem) — Claude Code memory integration reference
- [A-MEM](https://arxiv.org/abs/2510.02178) — self-evolving memory architecture
- [MAGMA](https://arxiv.org/abs/2501.13956) — multi-graph memory agent
- [Beads](https://github.com/steveyegge/beads) — git-backed issue tracker for AI agents

## License

MIT
