/**
 * MAGMA Adaptive Graph Traversal
 *
 * Beam search over multi-graph memory structure with intent-aware routing.
 * Reference: MAGMA paper (arXiv:2501.XXXXX, Jan 2026)
 */

import type { Database } from "bun:sqlite";
import type { IntentType } from "./intent.ts";
import { getIntentWeights } from "./intent.ts";

// =============================================================================
// Types
// =============================================================================

export interface TraversalOptions {
  maxDepth: number;           // 2-3 hops
  beamWidth: number;          // 5-10 nodes per level
  budget: number;             // Max total nodes (20-50)
  intent: IntentType;
  queryEmbedding: number[];
}

export interface TraversalNode {
  docId: number;
  path: string;
  score: number;
  hops: number;
  viaRelation?: string;
}

// =============================================================================
// Helpers
// =============================================================================

/**
 * Calculate cosine similarity between two vectors.
 */
function cosineSimilarity(a: number[] | Float32Array, b: number[] | Float32Array): number {
  if (a.length !== b.length) return 0;

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  return magnitude === 0 ? 0 : dotProduct / magnitude;
}

/**
 * Get document embedding from vectors_vec table.
 */
function getDocEmbedding(db: Database, docId: number): Float32Array {
  // Get the hash for this document
  const doc = db.prepare(`
    SELECT hash FROM documents WHERE id = ?
  `).get(docId) as { hash: string } | undefined;

  if (!doc) return new Float32Array(0);

  // Get embedding for seq=0 (whole document)
  const row = db.prepare(`
    SELECT embedding FROM vectors_vec WHERE hash_seq = ?
  `).get(`${doc.hash}_0`) as { embedding: Float32Array } | undefined;

  return row?.embedding || new Float32Array(0);
}

/**
 * Get document path from ID.
 */
function getDocPath(db: Database, docId: number): string {
  const row = db.prepare(`
    SELECT collection, path FROM documents WHERE id = ?
  `).get(docId) as { collection: string; path: string } | undefined;

  return row ? `${row.collection}/${row.path}` : '';
}

// =============================================================================
// Adaptive Traversal
// =============================================================================

/**
 * Get document ID from hash.
 */
function getDocIdFromHash(db: Database, hash: string): number | null {
  const row = db.prepare(`
    SELECT id FROM documents WHERE hash = ? AND active = 1 LIMIT 1
  `).get(hash) as { id: number } | undefined;
  return row?.id || null;
}

/**
 * Perform intent-aware beam search over memory graph.
 *
 * Algorithm:
 * 1. Start from anchor documents (top BM25/vector results)
 * 2. Expand frontier by following edges weighted by intent
 * 3. Score each new node: parent_score * decay + transition_score
 * 4. Keep top-k nodes per level (beam search)
 * 5. Stop at maxDepth or budget
 */
export function adaptiveTraversal(
  db: Database,
  anchors: { hash: string; score: number }[],
  options: TraversalOptions
): TraversalNode[] {
  // Convert hashes to IDs
  const anchorNodes: { docId: number; score: number }[] = [];
  for (const anchor of anchors) {
    const docId = getDocIdFromHash(db, anchor.hash);
    if (docId !== null) {
      anchorNodes.push({ docId, score: anchor.score });
    }
  }
  const { maxDepth, beamWidth, budget, intent, queryEmbedding } = options;

  // Intent-specific weights for structural alignment
  const weights = getIntentWeights(intent);

  const visited = new Map<number, TraversalNode>();
  let currentFrontier: TraversalNode[] = anchorNodes.map(a => ({
    docId: a.docId,
    path: getDocPath(db, a.docId),
    score: a.score,
    hops: 0,
  }));

  // Add anchors to visited set
  for (const node of currentFrontier) {
    visited.set(node.docId, node);
  }

  // Beam search expansion
  for (let depth = 1; depth <= maxDepth; depth++) {
    const candidates: TraversalNode[] = [];

    for (const u of currentFrontier) {
      // Get all neighbors via any relation type
      const neighbors = db.prepare(`
        SELECT target_id as docId, relation_type, weight
        FROM memory_relations
        WHERE source_id = ?

        UNION

        SELECT source_id as docId, relation_type, weight
        FROM memory_relations
        WHERE target_id = ? AND relation_type IN ('semantic', 'entity')
      `).all(u.docId, u.docId) as { docId: number; relation_type: string; weight: number }[];

      for (const neighbor of neighbors) {
        if (visited.has(neighbor.docId)) continue;

        // Get neighbor embedding for semantic affinity
        const neighborVec = getDocEmbedding(db, neighbor.docId);
        const semanticAffinity = neighborVec.length > 0
          ? cosineSimilarity(queryEmbedding, neighborVec)
          : 0;

        // Calculate transition score: λ1·structure + λ2·semantic
        const λ1 = 0.6;
        const λ2 = 0.4;
        const structureScore = weights[neighbor.relation_type as keyof typeof weights] || 1.0;
        const transitionScore = Math.exp(λ1 * structureScore + λ2 * semanticAffinity);

        // Apply decay and accumulate
        const γ = 0.9;
        const newScore = u.score * γ + transitionScore * neighbor.weight;

        candidates.push({
          docId: neighbor.docId,
          path: getDocPath(db, neighbor.docId),
          score: newScore,
          hops: depth,
          viaRelation: neighbor.relation_type,
        });
      }
    }

    // Take top-k by score (beam search)
    candidates.sort((a, b) => b.score - a.score);
    currentFrontier = candidates.slice(0, beamWidth);

    for (const node of currentFrontier) {
      visited.set(node.docId, node);
    }

    // Budget check
    if (visited.size >= budget) break;
  }

  // Convert to sorted array
  return Array.from(visited.values()).sort((a, b) => b.score - a.score);
}

/**
 * Merge graph traversal results with original search results.
 * Returns results with both hash and score for re-integration.
 */
export function mergeTraversalResults(
  db: Database,
  originalResults: { hash: string; score: number }[],
  traversedNodes: TraversalNode[]
): { hash: string; score: number }[] {
  const merged = new Map<string, number>();

  // Add original results
  for (const r of originalResults) {
    merged.set(r.hash, r.score);
  }

  // Merge traversed nodes (boost scores slightly for multi-hop discoveries)
  for (const node of traversedNodes) {
    // Get hash from doc ID
    const doc = db.prepare(`SELECT hash FROM documents WHERE id = ?`).get(node.docId) as { hash: string } | undefined;
    if (!doc) continue;

    const existing = merged.get(doc.hash);
    if (existing !== undefined) {
      // Document found via both direct search and traversal - boost it
      merged.set(doc.hash, Math.max(existing, node.score * 1.1));
    } else {
      // New document discovered via traversal
      merged.set(doc.hash, node.score * 0.8); // Slight penalty for indirect hits
    }
  }

  // Convert back to array and sort
  return Array.from(merged.entries())
    .map(([hash, score]) => ({ hash, score }))
    .sort((a, b) => b.score - a.score);
}
