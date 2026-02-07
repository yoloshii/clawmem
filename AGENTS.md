# ClawMem — Agent Quick Reference

## Inference Services

ClawMem uses three `llama-server` instances for neural inference. By default, the `bin/clawmem` wrapper points at `localhost:8088/8089/8090` — run them on your local GPU.

| Service | Port | Model | VRAM | Protocol |
|---|---|---|---|---|
| Embedding | 8088 | granite-embedding-278m-multilingual-Q6_K | ~400MB | `/v1/embeddings` |
| LLM | 8089 | qmd-query-expansion-1.7B-q4_k_m | ~2.2GB | `/v1/chat/completions` |
| Reranker | 8090 | qwen3-reranker-0.6B-Q8_0 | ~1.3GB | `/v1/rerank` |

**Total VRAM:** ~4.5GB. Fits alongside other workloads on any modern GPU.

**Remote option:** To offload to a separate GPU machine, set `CLAWMEM_EMBED_URL`, `CLAWMEM_LLM_URL`, `CLAWMEM_RERANK_URL` to the remote host. Set `CLAWMEM_NO_LOCAL_MODELS=true` to prevent surprise fallback downloads.

**No GPU:** LLM and reranker fall back to in-process `node-llama-cpp` automatically (auto-downloads models on first use). CPU inference works but is significantly slower — GPU is strongly recommended. Embedding has no in-process fallback — a `llama-server --embeddings` instance is always required.

### Model Recommendations

| Role | Recommended Model | Source | Size | Notes |
|---|---|---|---|---|
| Embedding | granite-embedding-278m-multilingual-Q6_K | [bartowski/granite-embedding-278m-multilingual-GGUF](https://huggingface.co/bartowski/granite-embedding-278m-multilingual-GGUF) | 226MB | 768 dimensions. 512-token context (~1100 chars). Client-side truncation prevents 500 errors. |
| LLM | qmd-query-expansion-1.7B-q4_k_m | [tobil/qmd-query-expansion-1.7B-gguf](https://huggingface.co/tobil/qmd-query-expansion-1.7B-gguf) | ~1.1GB | QMD's Qwen3-1.7B finetune — trained specifically for query expansion (hyde/lex/vec). |
| Reranker | qwen3-reranker-0.6B-Q8_0 | [ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF) | ~600MB | Cross-encoder architecture. Scores candidates against original query. |

**Qwen3 /no_think flag:** Qwen3 uses thinking tokens by default. ClawMem appends `/no_think` to all prompts automatically for structured output.

### Server Setup (all three use llama-server)

```bash
# Embedding (--embeddings flag required)
llama-server -m granite-embedding-278m-multilingual-Q6_K.gguf \
  --embeddings --port 8088 --host 0.0.0.0 --no-mmap -ngl 99 -c 2048 --batch-size 2048

# LLM (QMD finetuned model recommended)
llama-server -m qmd-query-expansion-1.7B-q4_k_m.gguf \
  --port 8089 --host 0.0.0.0 -ngl 99 -c 4096 --batch-size 512

# Reranker (--reranking flag required)
llama-server -m Qwen3-Reranker-0.6B-Q8_0.gguf \
  --port 8090 --host 0.0.0.0 -ngl 99 -c 2048 --batch-size 512 --reranking
```

### Verify Endpoints

```bash
# Embedding
curl http://host:8088/v1/embeddings -d '{"input":"test","model":"embedding"}' -H 'Content-Type: application/json'

# LLM
curl http://host:8089/v1/models

# Reranker
curl http://host:8090/v1/models
```

## Environment Variable Reference

| Variable | Default (via wrapper) | Effect |
|---|---|---|
| `CLAWMEM_EMBED_URL` | `http://localhost:8088` | Embedding server. No in-process fallback — `llama-server --embeddings` required. |
| `CLAWMEM_LLM_URL` | `http://localhost:8089` | LLM server for intent, expansion, A-MEM. Falls to `node-llama-cpp` if unset + `NO_LOCAL_MODELS=false`. |
| `CLAWMEM_RERANK_URL` | `http://localhost:8090` | Reranker server. Falls to `node-llama-cpp` if unset + `NO_LOCAL_MODELS=false`. |
| `CLAWMEM_NO_LOCAL_MODELS` | `false` | Blocks `node-llama-cpp` from auto-downloading GGUF models. Set `true` for remote-only setups. |
| `CLAWMEM_ENABLE_AMEM` | enabled | A-MEM note construction + link generation during indexing. |
| `CLAWMEM_ENABLE_CONSOLIDATION` | disabled | Background worker backfills unenriched docs. Needs long-lived MCP process. |
| `CLAWMEM_CONSOLIDATION_INTERVAL` | 300000 | Worker interval in ms (min 15000). |

**Note:** The `bin/clawmem` wrapper sets all endpoint defaults and `CLAWMEM_NO_LOCAL_MODELS`. Always use the wrapper — never `bun run src/clawmem.ts` directly.

## Quick Setup

```bash
git clone https://github.com/yoloshii/clawmem.git ~/clawmem
cd ~/clawmem && bun install
ln -sf ~/clawmem/bin/clawmem ~/.bun/bin/clawmem

# Bootstrap a vault (init + index + embed + hooks + MCP)
./bin/clawmem bootstrap ~/notes --name notes

# Or step by step:
./bin/clawmem init
./bin/clawmem collection add ~/notes --name notes
./bin/clawmem update --embed
./bin/clawmem setup hooks
./bin/clawmem setup mcp

# Verify
./bin/clawmem doctor    # Full health check
./bin/clawmem status    # Quick index status
```

---

## Memory Retrieval (90/10 Rule)

ClawMem hooks handle ~90% of retrieval automatically. Agent-initiated MCP calls cover the remaining ~10%.

### Tier 2 — Automatic (hooks, zero agent effort)

| Hook | Trigger | Budget | Content |
|------|---------|--------|---------|
| `session-bootstrap` | SessionStart | 2000 tokens | profile + latest handoff + recent decisions + stale notes |
| `context-surfacing` | UserPromptSubmit | 800 tokens | hybrid search → `<vault-context>` injection |
| `staleness-check` | SessionStart | 250 tokens | flags notes not modified in 30+ days |
| `decision-extractor` | Stop / IO3 post-response | — | LLM extracts observations → `_clawmem/observations/`, infers causal links |
| `handoff-generator` | Stop / IO3 post-response | — | LLM summarizes session → `_clawmem/handoffs/` |
| `feedback-loop` | Stop / IO3 post-response | — | tracks referenced notes → boosts confidence |

**Default behavior:** Read injected `<vault-context>` first. If sufficient, answer immediately.

**Hook blind spots (by design):** Hooks filter out `_clawmem/` system artifacts, enforce score thresholds, and cap token budget. Absence in `<vault-context>` does NOT mean absence in memory. If you expect a memory to exist but it wasn't surfaced, escalate to Tier 3.

### Tier 3 — Agent-Initiated (one targeted MCP call)

**Escalate ONLY when one of these three rules fires:**
1. **Low-specificity injection** — `<vault-context>` is empty or lacks the specific fact/chain the task requires. Hooks surface top-k by relevance; if the needed memory wasn't in top-k, escalate.
2. **Cross-session question** — the task explicitly references prior sessions or decisions: "why did we decide X", "what changed since last time", "when did we start doing Y".
3. **Pre-irreversible check** — about to make a destructive or hard-to-reverse change (deletion, config change, architecture decision). Check vault for prior decisions before proceeding.

All other retrieval is handled by Tier 2 hooks. Do NOT call MCP tools speculatively or "just to be thorough."

**Once escalated, route by query type:**

```
1a. General recall → query(query, compact=true, limit=20)
    Full hybrid: BM25 + vector + query expansion + deep reranking (4000 char).
    Supports compact and collection filter. Default for most Tier 3 needs.

1b. Causal/why/when/entity → intent_search(query, enable_graph_traversal=true)
    MAGMA intent classification + intent-weighted RRF + multi-hop graph traversal.
    Use DIRECTLY (not as fallback) when the question is "why", "when", "how did X lead to Y",
    or needs entity-relationship traversal.
    Override auto-detection: force_intent="WHY"|"WHEN"|"ENTITY"|"WHAT"
    When to override:
      WHY — "why", "what led to", "rationale", "tradeoff", "decision behind"
      ENTITY — named component/person/service needing cross-doc linkage, not just keyword hits
      WHEN — timelines, first/last occurrence, "when did this change/regress"
    WHEN note: start with enable_graph_traversal=false (BM25-biased); fall back to query() if recall drifts.

    Choose 1a or 1b based on query type. They are parallel options, not sequential.

2. Progressive disclosure → multi_get("path1,path2") for full content of top hits

3. Spot checks → search(query) (BM25, 0 GPU) or vsearch(query) (vector, 1 GPU)

4. Chain tracing → find_causal_links(docid, direction="both", depth=5)
   Traverses causal edges between _clawmem/observations/ docs (from decision-extractor).

5. Memory debugging → memory_evolution_status(docid)
```

**Other tools:**
- `find_similar(docid)` — related documents from a known anchor.
- `session_log` — recent sessions with handoff summaries.
- `profile` — user profile (static facts + dynamic context).
- `memory_forget(query)` — deactivate a memory by closest match.
- `beads_sync(project_path?)` — import `.beads/beads.jsonl` into memory. Bridges deps into `memory_relations`, runs A-MEM enrichment on new docs. Usually automatic via watcher.

### Anti-Patterns

- Do NOT call `query` or `intent_search` every turn — three rules above are the only gates.
- Do NOT re-search what's already in `<vault-context>`.
- Do NOT run `status` routinely. Only when retrieval feels broken or after large ingestion.

## Tool Selection (one-liner)

```
ClawMem escalation: query(compact=true) | intent_search(why/when/entity) → multi_get → search/vsearch (spot checks)
```

## Composite Scoring (automatic, applied to all search tools)

```
compositeScore = 0.50 × searchScore + 0.25 × recencyScore + 0.25 × confidenceScore
```

Recency intent detected ("latest", "recent", "last session"):
```
compositeScore = 0.10 × searchScore + 0.70 × recencyScore + 0.20 × confidenceScore
```

| Content Type | Half-Life | Effect |
|--------------|-----------|--------|
| decision, hub | ∞ | Never decay |
| project | 120 days | Slow decay |
| research | 90 days | Moderate decay |
| note | 60 days | Default |
| progress | 45 days | Faster decay |
| handoff | 30 days | Fast — recent matters most |

## Indexing & Graph Building

### What Gets Indexed (per collection in config.yaml)

- `**/MEMORY.md` — any depth
- `**/memory/**/*.md`, `**/memory/**/*.txt` — session logs
- `**/docs/**/*.md`, `**/docs/**/*.txt` — documentation
- `**/research/**/*.md`, `**/research/**/*.txt` — research dumps
- `**/YYYY-MM-DD*.md`, `**/YYYY-MM-DD*.txt` — date-format records

### Excluded (even if pattern matches)

- `gits/`, `scraped/`, `.git/`, `node_modules/`, `dist/`, `build/`, `vendor/`

### Indexing vs Embedding (important distinction)

**Infrastructure (Tier 1, no agent action needed):**
- **`clawmem-watcher`** — keeps index + A-MEM fresh (continuous, on `.md` change). Also watches `.jsonl` — routes `.beads/beads.jsonl` changes to `syncBeadsIssues()` (auto-bridges deps into `memory_relations`). Does NOT embed.
- **`clawmem-embed` timer** — keeps embeddings fresh (daily 04:00 UTC). Idempotent, skips already-embedded fragments.

**Impact of missing embeddings:** `vsearch`, `query` (vector component), `context-surfacing` (vector component), and `generateMemoryLinks()` (neighbor discovery) all depend on embeddings. If embeddings are missing, these degrade silently — BM25 still works, but vector recall and inter-doc link quality suffer.

**Agent escape hatches (rare):**
- `clawmem embed` via CLI if you just wrote a doc and need immediate vector recall in the next turn.
- Manual `reindex` only when immediate index freshness is required and watcher hasn't caught up.

### Graph Population (memory_relations)

The `memory_relations` table is populated by multiple independent sources:

| Source | Edge Types | Trigger | Notes |
|--------|-----------|---------|-------|
| A-MEM `generateMemoryLinks()` | semantic, supporting, contradicts | Indexing (new docs only) | LLM-assessed confidence + reasoning. Requires embeddings for neighbor discovery. |
| A-MEM `inferCausalLinks()` | causal | Post-response (IO3 decision-extractor) | Links between `_clawmem/observations/` docs, not arbitrary workspace docs. |
| Beads `syncBeadsIssues()` | causal, supporting, semantic | `beads_sync` MCP tool or watcher (.jsonl change) | Maps beads deps: blocks→causal, discovered-from→supporting, relates-to→semantic. Metadata: `{origin: "beads"}`. |
| `buildTemporalBackbone()` | temporal | `build_graphs` MCP tool (manual) | Creation-order edges between all active docs. |
| `buildSemanticGraph()` | semantic | `build_graphs` MCP tool (manual) | Pure cosine similarity. PK collision: `INSERT OR IGNORE` means A-MEM semantic edges take precedence if they exist first. |

**Edge collision:** Both `generateMemoryLinks()` and `buildSemanticGraph()` insert `relation_type='semantic'`. PK is `(source_id, target_id, relation_type)` — first writer wins.

**Graph traversal asymmetry:** `adaptiveTraversal()` traverses all edge types outbound (source→target) but only `semantic` and `entity` edges inbound (target→source). Temporal and causal edges are directional only.

### When to Run `build_graphs`

- After **bulk ingestion** (many new docs at once) — adds temporal backbone and fills semantic gaps where A-MEM links are sparse.
- When `intent_search` for WHY/ENTITY returns **weak or obviously incomplete results** and you suspect graph sparsity.
- Do NOT run after every reindex. Routine indexing creates A-MEM links automatically for new docs.

### When to Run `index_stats`

- After bulk ingestion to verify doc counts and embedding coverage.
- When retrieval quality seems degraded — check for unembedded docs or content type distribution issues.
- Do NOT run routinely.

## Pipeline Details

### `query` (default Tier 3 workhorse)

```
User Query → Intent Classification (heuristic, LLM fallback if confidence < 0.8)
  → BM25 Strong Signal Check (skip expansion if top hit ≥ 0.85 with gap ≥ 0.15)
  → Query Expansion (LLM generates hyde/lex/vec variants)
  → Parallel: BM25(original, 2×) + Vector(original, 2×) + BM25(expanded, 1×) + Vector(expanded, 1×)
  → Reciprocal Rank Fusion (k=60, top 30)
  → Cross-Encoder Reranking (4000 char context per doc)
  → Position-Aware Blending (α=0.75 top3, 0.60 mid, 0.40 tail)
  → SAME Composite Scoring
```

### `intent_search` (specialist for causal chains)

```
User Query → Intent Classification (WHY/WHEN/ENTITY/WHAT)
  → BM25 + Vector (intent-weighted RRF: boost BM25 for WHEN, vector for WHY)
  → Graph Traversal (multi-hop beam search over memory_relations)
      Outbound: all edge types (semantic, supporting, contradicts, causal, temporal)
      Inbound: semantic and entity only
  → Cross-Encoder Reranking (200 char context per doc)
  → SAME Composite Scoring
```

### Key Differences

| Aspect | `query` | `intent_search` |
|--------|---------|-----------------|
| Query expansion | Yes | No |
| Rerank context | 4000 chars/doc | 200 chars/doc |
| Graph traversal | No | Yes (WHY/ENTITY, multi-hop) |
| `compact` param | Yes | No |
| `collection` filter | Yes | No |
| Best for | Most queries, progressive disclosure | Causal chains spanning multiple docs |

## Operational Issue Tracking

When encountering tool failures, instruction contradictions, retrieval gaps, or workflow friction that would benefit from a fix:

Write to `docs/issues/YYYY-MM-DD-<slug>.md` with: category, severity, what happened, what was expected, context, suggested fix.

**File structure:**
```
# <title>
- Category: tool-failure | instruction-gap | workflow-friction | retrieval-gap | inconsistency
- Severity: critical | high | medium
- Status: open | resolved

## Observed
## Expected
## Context
## Suggested Fix
```

**Triggers:** repeated tool error, instruction that contradicts observed behavior, retrieval consistently missing known content, workflow requiring unnecessary steps.

**Do NOT log:** one-off transient errors, user-caused issues, issues already recorded.

## Troubleshooting

```
Symptom: "Local model download blocked" error
  → llama-server endpoint unreachable while CLAWMEM_NO_LOCAL_MODELS=true.
  → Fix: Start the llama-server instance. Or set CLAWMEM_NO_LOCAL_MODELS=false for in-process fallback.

Symptom: Query expansion always fails / returns garbage
  → In-process CPU inference is significantly slower and less reliable than GPU.
  → Fix: Run llama-server on a GPU. Even a low-end NVIDIA card handles 1.7B models.

Symptom: Vector search returns no results but BM25 works
  → Missing embeddings. Watcher indexes but does NOT embed.
  → Fix: Run `clawmem embed` or wait for the daily embed timer.

Symptom: context-surfacing hook returns empty
  → Prompt too short (<20 chars), starts with `/`, or no docs score above threshold.
  → Fix: Check `clawmem status` for doc counts. Check `clawmem embed` for embedding coverage.

Symptom: intent_search returns weak results for WHY/ENTITY
  → Graph may be sparse (few A-MEM edges).
  → Fix: Run `build_graphs` to add temporal backbone + semantic edges.
```

## CLI Reference

Run `clawmem --help` for full command listing. Use this before guessing at commands or parameters.

## Integration Notes

- QMD retrieval (BM25, vector, RRF, rerank, query expansion) is forked into ClawMem. Do not call standalone QMD tools.
- SAME (composite scoring), MAGMA (intent + graph), A-MEM (self-evolving notes) layer on top of QMD substrate.
- Three `llama-server` instances (embedding, LLM, reranker) on local or remote GPU. Wrapper defaults to `localhost:8088/8089/8090`.
- `CLAWMEM_NO_LOCAL_MODELS=false` (default) allows in-process LLM/reranker fallback via `node-llama-cpp`. Set `true` for remote-only setups to fail fast on unreachable endpoints.
- Consolidation worker (`CLAWMEM_ENABLE_CONSOLIDATION=true`) backfills unenriched docs with A-MEM notes + links. Only runs if the MCP process stays alive long enough to tick (every 5min). Not reliable in stateless `--print` per-request mode.
- Stop hooks (`decision-extractor`, `handoff-generator`, `feedback-loop`) are unreliable under `--print` mode. IO3 (`postrun.go`) fills the gap by invoking these hooks post-response with synthetic transcripts.
- Beads integration: `syncBeadsIssues()` creates markdown docs in `beads` collection, maps dependency edges (`blocks`→causal, `discovered-from`→supporting, `relates-to`→semantic) into `memory_relations`, and triggers A-MEM enrichment for new docs. Watcher auto-triggers on `.beads/beads.jsonl` changes; `beads_sync` MCP tool for manual sync.
