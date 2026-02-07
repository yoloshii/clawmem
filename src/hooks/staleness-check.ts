/**
 * Staleness Check Hook - SessionStart
 *
 * Fires at session start. Finds documents that have a review_by date
 * in the past and haven't been updated recently. Injects a gentle
 * reminder to review them.
 */

import type { Store } from "../store.ts";
import type { HookInput, HookOutput } from "../hooks.ts";
import {
  makeContextOutput,
  makeEmptyOutput,
  estimateTokens,
  logInjection,
} from "../hooks.ts";

// =============================================================================
// Config
// =============================================================================

const MAX_STALE_NOTES = 5;
const MAX_TOKEN_BUDGET = 250;
const STALE_DAYS = 30;

// =============================================================================
// Handler
// =============================================================================

export async function stalenessCheck(
  store: Store,
  input: HookInput
): Promise<HookOutput> {
  const now = new Date();

  // Find documents with review_by in the past
  const reviewDue = findReviewDue(store, now);

  // Find documents not modified in STALE_DAYS
  const stale = findStaleByAge(store, now);

  // Merge and deduplicate
  const allStale = new Map<string, { title: string; path: string; reason: string }>();

  for (const d of reviewDue) {
    allStale.set(d.path, d);
  }
  for (const d of stale) {
    if (!allStale.has(d.path)) {
      allStale.set(d.path, d);
    }
  }

  if (allStale.size === 0) return makeEmptyOutput("staleness-check");

  // Build context within budget
  const entries = [...allStale.values()].slice(0, MAX_STALE_NOTES);
  const lines = ["**Notes needing review:**"];
  const paths: string[] = [];
  let tokens = estimateTokens(lines[0]!);

  for (const entry of entries) {
    const line = `- ${entry.title} (${entry.path}) — ${entry.reason}`;
    const lineTokens = estimateTokens(line);
    if (tokens + lineTokens > MAX_TOKEN_BUDGET && lines.length > 1) break;
    lines.push(line);
    paths.push(entry.path);
    tokens += lineTokens;
  }

  if (lines.length <= 1) return makeEmptyOutput("staleness-check");

  // Log injection
  if (input.sessionId) {
    logInjection(store, input.sessionId, "staleness-check", paths, tokens);
  }

  return makeContextOutput(
    "staleness-check",
    `<vault-staleness>\n${lines.join("\n")}\n</vault-staleness>`
  );
}

// =============================================================================
// Finders
// =============================================================================

function findReviewDue(
  store: Store,
  now: Date
): { title: string; path: string; reason: string }[] {
  const nowStr = now.toISOString();

  try {
    const rows = store.db.prepare(`
      SELECT collection, path, title, review_by
      FROM documents
      WHERE active = 1
        AND review_by IS NOT NULL
        AND review_by != ''
        AND review_by <= ?
      ORDER BY review_by ASC
      LIMIT ?
    `).all(nowStr, MAX_STALE_NOTES) as { collection: string; path: string; title: string; review_by: string }[];

    return rows.map(r => ({
      title: r.title,
      path: `${r.collection}/${r.path}`,
      reason: `review due ${r.review_by.slice(0, 10)}`,
    }));
  } catch {
    return [];
  }
}

function findStaleByAge(
  store: Store,
  now: Date
): { title: string; path: string; reason: string }[] {
  const cutoff = new Date(now);
  cutoff.setDate(cutoff.getDate() - STALE_DAYS);

  const stale = store.getStaleDocuments(cutoff.toISOString());

  return stale.slice(0, MAX_STALE_NOTES).map(d => ({
    title: d.title,
    path: `${d.collection}/${d.path}`,
    reason: `not modified since ${d.modifiedAt.slice(0, 10)}`,
  }));
}
