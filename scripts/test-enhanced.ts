#!/usr/bin/env bun
/**
 * Test script for enhanced ClawMem features (MAGMA + A-MEM).
 * Builds graphs, runs intent_search pipeline, checks A-MEM status.
 *
 * Must set CLAWMEM_EMBED_URL for vector search:
 *   CLAWMEM_EMBED_URL=http://localhost:8088 bun run scripts/test-enhanced.ts
 */
import { createStore, DEFAULT_EMBED_MODEL, enableProductionMode, type SearchResult } from "../src/store.ts";
enableProductionMode();
import { classifyIntent, getIntentWeights, type IntentType } from "../src/intent.ts";
import { adaptiveTraversal, mergeTraversalResults } from "../src/graph-traversal.ts";
import { reciprocalRankFusion, toRanked } from "../src/search-utils.ts";
import { getDefaultLlamaCpp } from "../src/llm.ts";

const store = createStore();
const database = store.db;
const llm = getDefaultLlamaCpp();

// ─── Step 1: Build Graphs ───────────────────────────────────────────────────
console.log("\n=== STEP 1: Build Graphs ===\n");

console.log("Building temporal backbone...");
const temporalEdges = store.buildTemporalBackbone();
console.log(`  → ${temporalEdges} temporal edges created`);

console.log("Building semantic graph (threshold=0.7)...");
const semanticEdges = await store.buildSemanticGraph(0.7);
console.log(`  → ${semanticEdges} semantic edges created`);

// Check total edges
const edgeCount = database.prepare(
  `SELECT relation_type, COUNT(*) as c FROM memory_relations GROUP BY relation_type`
).all() as { relation_type: string; c: number }[];
console.log("\nGraph edges by type:");
for (const row of edgeCount) {
  console.log(`  ${row.relation_type}: ${row.c}`);
}

// ─── Step 2: A-MEM Status ───────────────────────────────────────────────────
console.log("\n=== STEP 2: A-MEM Status ===\n");

const totalDocs = (database.prepare(`SELECT COUNT(*) as c FROM documents WHERE active=1`).get() as { c: number }).c;
const enriched = (database.prepare(`SELECT COUNT(*) as c FROM documents WHERE active=1 AND amem_keywords IS NOT NULL`).get() as { c: number }).c;
const unenriched = totalDocs - enriched;
console.log(`Total active docs: ${totalDocs}`);
console.log(`A-MEM enriched:    ${enriched}`);
console.log(`Needs enrichment:  ${unenriched}`);

// ─── Step 3: Intent Classification ──────────────────────────────────────────
console.log("\n=== STEP 3: Intent Classification Tests ===\n");

const testQueries = [
  "why did we choose browser-use over playwright?",
  "what happened with the X automation project last week?",
  "SEO crawler architecture",
  "compare alpha extraction approaches",
];

for (const q of testQueries) {
  const intent = await classifyIntent(q, llm, database);
  console.log(`  "${q}"`);
  console.log(`    → intent=${intent.intent} confidence=${intent.confidence.toFixed(2)} temporal=${intent.temporal_start || 'none'}`);
}

// ─── Step 4: Intent Search (full pipeline matching mcp.ts intent_search) ────
console.log("\n=== STEP 4: Intent Search Tests ===\n");

const searchQueries = [
  "X twitter alpha extraction strategy",
  "how does the SEO crawler work",
  "browser automation agent frameworks",
];

for (const query of searchQueries) {
  console.log(`\nQuery: "${query}"`);

  // Step 4a: Classify intent
  const intent = await classifyIntent(query, llm, database);
  console.log(`  Intent: ${intent.intent} (${intent.confidence.toFixed(2)})`);

  // Step 4b: Baseline search (BM25 + Vector)
  const bm25Results = store.searchFTS(query, 30);
  const vecResults = await store.searchVec(query, DEFAULT_EMBED_MODEL, 30);
  console.log(`  BM25: ${bm25Results.length} results | Vector: ${vecResults.length} results`);

  // Step 4c: Intent-weighted RRF
  const rrfWeights = intent.intent === 'WHEN'
    ? [1.5, 1.0]
    : intent.intent === 'WHY'
    ? [1.0, 1.5]
    : [1.0, 1.0];

  const fusedRanked = reciprocalRankFusion([bm25Results.map(toRanked), vecResults.map(toRanked)], rrfWeights);

  // Map RRF back to SearchResult
  const allSearchResults = [...bm25Results, ...vecResults];
  const fused: SearchResult[] = fusedRanked.map(fr => {
    const original = allSearchResults.find(r => r.filepath === fr.file);
    return original ? { ...original, score: fr.score } : null;
  }).filter((r): r is SearchResult => r !== null);

  console.log(`  RRF fused: ${fused.length} results`);

  // Step 4d: Graph expansion (for WHY and ENTITY intents)
  let expanded = fused;
  if (intent.intent === 'WHY' || intent.intent === 'ENTITY') {
    const anchorEmbeddingResult = await llm.embed(query);
    if (anchorEmbeddingResult) {
      const traversed = adaptiveTraversal(
        database,
        fused.slice(0, 10).map(r => ({ hash: r.hash, score: r.score })),
        {
          maxDepth: 2,
          beamWidth: 5,
          budget: 30,
          intent: intent.intent,
          queryEmbedding: anchorEmbeddingResult.embedding,
        }
      );

      const merged = mergeTraversalResults(
        database,
        fused.map(r => ({ hash: r.hash, score: r.score })),
        traversed
      );

      expanded = merged.map(m => {
        const original = fused.find(f => f.hash === m.hash);
        return original
          ? { ...original, score: m.score }
          : { ...fused[0]!, hash: m.hash, score: m.score };
      }).filter((r): r is SearchResult => r !== null);

      console.log(`  Graph expansion: ${traversed.length} nodes traversed → ${expanded.length} merged`);
    } else {
      console.log(`  Graph expansion: skipped (embedding not available)`);
    }
  } else {
    console.log(`  Graph expansion: skipped (intent=${intent.intent}, only WHY/ENTITY use graph)`);
  }

  // Show top 5
  console.log(`  Top 5 results:`);
  for (const r of expanded.slice(0, 5)) {
    console.log(`    ${r.score.toFixed(3)} ${r.title || r.filepath}`);
  }
}

console.log("\n=== DONE ===\n");
await llm.dispose();
store.close();
