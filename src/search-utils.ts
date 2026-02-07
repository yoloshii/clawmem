/**
 * ClawMem Search Utilities — Shared enrichment and fusion functions
 *
 * Consolidates duplicated code from clawmem.ts, mcp.ts, and context-surfacing.ts.
 */

import type { Store, SearchResult } from "./store.ts";
import type { EnrichedResult } from "./memory.ts";

// =============================================================================
// Result Enrichment
// =============================================================================

/**
 * Join search results with SAME metadata from the documents table.
 * Adds content_type, modified_at, access_count, confidence to each result.
 */
export function enrichResults(
  store: Store,
  results: SearchResult[],
  _query: string
): EnrichedResult[] {
  return results.map(r => {
    const row = store.db.prepare(`
      SELECT content_type, modified_at, access_count, confidence, domain, workstream, tags
      FROM documents
      WHERE active = 1 AND (collection || '/' || path) = ?
      LIMIT 1
    `).get(r.displayPath) as any | null;

    return {
      ...r,
      contentType: row?.content_type ?? "note",
      modifiedAt: row?.modified_at ?? r.modifiedAt,
      accessCount: row?.access_count ?? 0,
      confidence: row?.confidence ?? 0.5,
    } as EnrichedResult;
  });
}

// =============================================================================
// Ranked Result Type (for RRF)
// =============================================================================

export type RankedResult = {
  file: string;
  displayPath: string;
  title: string;
  body: string;
  score: number;
};

// =============================================================================
// Reciprocal Rank Fusion
// =============================================================================

/**
 * Merge multiple ranked result lists using Reciprocal Rank Fusion.
 * k=60 is the standard RRF constant. Top-rank bonuses reward results
 * that appear at rank 0 (+0.05) or rank 1-2 (+0.02).
 */
export function reciprocalRankFusion(
  resultLists: RankedResult[][],
  weights: number[],
  k: number = 60
): RankedResult[] {
  // Validate weights match result lists when explicitly provided
  if (weights.length > 0 && weights.length !== resultLists.length) {
    throw new Error(
      `weights length (${weights.length}) must match resultLists length (${resultLists.length})`
    );
  }

  // Validate k is finite and positive
  if (!Number.isFinite(k) || k <= 0) k = 60;

  // Validate all weights are finite and non-negative
  for (let w = 0; w < weights.length; w++) {
    if (!Number.isFinite(weights[w]) || weights[w]! < 0) {
      weights[w] = 1;
    }
  }

  const scores = new Map<string, { score: number; result: RankedResult }>();

  for (let i = 0; i < resultLists.length; i++) {
    const list = resultLists[i]!;
    const weight = weights[i] ?? 1;
    if (weight === 0) continue; // Skip zero-weight lists entirely
    for (let rank = 0; rank < list.length; rank++) {
      const r = list[rank]!;
      const existing = scores.get(r.file);
      const rrfScore = weight / (k + rank + 1);
      const bonus = rank === 0 ? 0.05 : rank <= 2 ? 0.02 : 0;
      const total = rrfScore + bonus;

      if (existing) {
        existing.score += total;
      } else {
        scores.set(r.file, { score: total, result: r });
      }
    }
  }

  return [...scores.values()]
    .sort((a, b) => b.score - a.score)
    .map(v => ({ ...v.result, score: v.score }));
}

/**
 * Convert a SearchResult to a RankedResult for use in RRF.
 */
export function toRanked(r: SearchResult): RankedResult {
  return {
    file: r.filepath,
    displayPath: r.displayPath,
    title: r.title,
    body: r.body || "",
    score: r.score,
  };
}
