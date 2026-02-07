/**
 * ClawMem Consolidation Worker
 *
 * Background worker that enriches documents missing A-MEM metadata.
 * Runs periodically to backfill memory notes for documents indexed before A-MEM.
 */

import type { Store } from "./store.ts";
import type { LlamaCpp } from "./llm.ts";

// =============================================================================
// Types
// =============================================================================

interface DocumentToEnrich {
  id: number;
  hash: string;
  title: string;
}

// =============================================================================
// Worker State
// =============================================================================

let consolidationTimer: Timer | null = null;
let isRunning = false;

// =============================================================================
// Worker Functions
// =============================================================================

/**
 * Starts the consolidation worker that enriches documents missing A-MEM metadata.
 *
 * @param store - Store instance with A-MEM methods
 * @param llm - LLM instance for memory note construction
 * @param intervalMs - Tick interval in milliseconds (default: 300000 = 5 min)
 */
export function startConsolidationWorker(
  store: Store,
  llm: LlamaCpp,
  intervalMs: number = 300000
): void {
  // Clamp interval to minimum 15 seconds
  const interval = Math.max(15000, intervalMs);

  console.log(`[consolidation] Starting worker with ${interval}ms interval`);

  // Set up periodic tick
  consolidationTimer = setInterval(async () => {
    await tick(store, llm);
  }, interval);

  // Use unref() to avoid blocking process exit
  consolidationTimer.unref();

  console.log("[consolidation] Worker started");
}

/**
 * Stops the consolidation worker.
 */
export function stopConsolidationWorker(): void {
  if (consolidationTimer) {
    clearInterval(consolidationTimer);
    consolidationTimer = null;
    console.log("[consolidation] Worker stopped");
  }
}

/**
 * Single worker tick: find and enrich up to 3 documents missing A-MEM metadata.
 */
async function tick(store: Store, llm: LlamaCpp): Promise<void> {
  // Reentrancy guard
  if (isRunning) {
    console.log("[consolidation] Skipping tick (already running)");
    return;
  }

  isRunning = true;

  try {
    // Find documents missing A-MEM keywords (primary indicator of unenriched docs)
    const docs = store.db
      .prepare<DocumentToEnrich[], []>(
        `SELECT id, hash, title
         FROM documents
         WHERE amem_keywords IS NULL AND active = 1
         ORDER BY created_at ASC
         LIMIT 3`
      )
      .all();

    if (docs.length === 0) {
      // No work to do
      return;
    }

    console.log(`[consolidation] Enriching ${docs.length} documents`);

    // Enrich each document (note + links, skip evolution to avoid cascades)
    for (const doc of docs) {
      try {
        // Construct and store memory note
        const note = await store.constructMemoryNote(llm, doc.id);
        await store.storeMemoryNote(doc.id, note);

        // Generate memory links (skip evolution for backlog)
        await store.generateMemoryLinks(llm, doc.id);

        console.log(`[consolidation] Enriched doc ${doc.id} (${doc.title})`);
      } catch (err) {
        console.error(`[consolidation] Failed to enrich doc ${doc.id}:`, err);
        // Continue with remaining docs (don't let one failure block the queue)
      }
    }
  } catch (err) {
    console.error("[consolidation] Tick failed:", err);
  } finally {
    isRunning = false;
  }
}
