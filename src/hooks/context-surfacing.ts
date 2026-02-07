/**
 * Context Surfacing Hook - UserPromptSubmit
 *
 * Fires on every user message. Searches the vault for relevant context,
 * applies SAME composite scoring, enforces a token budget, and injects
 * the most relevant notes as additional context for Claude.
 */

import type { Store, SearchResult } from "../store.ts";
import { DEFAULT_EMBED_MODEL, extractSnippet } from "../store.ts";
import type { HookInput, HookOutput } from "../hooks.ts";
import {
  makeContextOutput,
  makeEmptyOutput,
  smartTruncate,
  estimateTokens,
  logInjection,
  isHeartbeatPrompt,
  wasPromptSeenRecently,
} from "../hooks.ts";
import {
  applyCompositeScoring,
  hasRecencyIntent,
  type EnrichedResult,
  type ScoredResult,
} from "../memory.ts";
import { enrichResults } from "../search-utils.ts";
import { sanitizeSnippet } from "../promptguard.ts";
import { MAX_QUERY_LENGTH } from "../limits.ts";

// =============================================================================
// Config
// =============================================================================

const MAX_TOKEN_BUDGET = 800;
const MAX_RESULTS = 10;
const MIN_COMPOSITE_SCORE = 0.45;
const MIN_COMPOSITE_SCORE_RECENCY = 0.35;
const SNIPPET_MAX_CHARS = 300;
const MIN_PROMPT_LENGTH = 20;

// Directories to never surface
const FILTERED_PATHS = ["_PRIVATE/", "experiments/", "_clawmem/"];

// =============================================================================
// Handler
// =============================================================================

export async function contextSurfacing(
  store: Store,
  input: HookInput
): Promise<HookOutput> {
  let prompt = input.prompt?.trim();
  if (!prompt || prompt.length < MIN_PROMPT_LENGTH) return makeEmptyOutput("context-surfacing");

  // Bound query length to prevent DoS on search indices
  if (prompt.length > MAX_QUERY_LENGTH) prompt = prompt.slice(0, MAX_QUERY_LENGTH);

  // Skip slash commands
  if (prompt.startsWith("/")) return makeEmptyOutput("context-surfacing");

  // Heartbeat / duplicate suppression (IO4)
  if (isHeartbeatPrompt(prompt)) return makeEmptyOutput("context-surfacing");
  if (wasPromptSeenRecently(store, "context-surfacing", prompt)) {
    return makeEmptyOutput("context-surfacing");
  }

  const isRecency = hasRecencyIntent(prompt);
  const minScore = isRecency ? MIN_COMPOSITE_SCORE_RECENCY : MIN_COMPOSITE_SCORE;

  // Search: try vector first, fall back to BM25
  let results: SearchResult[] = [];
  try {
    results = await store.searchVec(prompt, DEFAULT_EMBED_MODEL, MAX_RESULTS);
  } catch {
    // Vector search unavailable (no embeddings), fall back to BM25
  }

  if (results.length === 0) {
    results = store.searchFTS(prompt, MAX_RESULTS);
  }

  if (results.length === 0) return makeEmptyOutput("context-surfacing");

  // Filter out private/excluded paths
  results = results.filter(r =>
    !FILTERED_PATHS.some(p => r.displayPath.includes(p))
  );

  if (results.length === 0) return makeEmptyOutput("context-surfacing");

  // Deduplicate by filepath (keep best score per path)
  const deduped = new Map<string, SearchResult>();
  for (const r of results) {
    const existing = deduped.get(r.filepath);
    if (!existing || r.score > existing.score) {
      deduped.set(r.filepath, r);
    }
  }
  results = [...deduped.values()];

  // Enrich with SAME metadata
  const enriched = enrichResults(store, results, prompt);

  // Apply composite scoring
  const scored = applyCompositeScoring(enriched, prompt)
    .filter(r => r.compositeScore >= minScore);

  if (scored.length === 0) return makeEmptyOutput("context-surfacing");

  // Build context within token budget
  const { context, paths, tokens } = buildContext(scored, prompt);

  if (!context) return makeEmptyOutput("context-surfacing");

  // Log the injection
  if (input.sessionId) {
    logInjection(store, input.sessionId, "context-surfacing", paths, tokens);
  }

  return makeContextOutput(
    "context-surfacing",
    `<vault-context>\n${context}\n</vault-context>`
  );
}

// =============================================================================
// Helpers
// =============================================================================

function buildContext(
  scored: ScoredResult[],
  query: string
): { context: string; paths: string[]; tokens: number } {
  const lines: string[] = [];
  const paths: string[] = [];
  let totalTokens = 0;

  for (const r of scored) {
    if (totalTokens >= MAX_TOKEN_BUDGET) break;

    const bodyStr = r.body || "";

    // Prompt injection guard: sanitize snippet before injection
    const sanitized = sanitizeSnippet(bodyStr);
    if (sanitized === "[content filtered for security]") continue;

    const snippet = smartTruncate(
      extractSnippet(sanitized, query, SNIPPET_MAX_CHARS, r.chunkPos).snippet,
      SNIPPET_MAX_CHARS
    );

    // Sanitize title and displayPath to prevent injection via metadata fields
    const safeTitle = sanitizeSnippet(r.title);
    const safePath = sanitizeSnippet(r.displayPath);
    if (safeTitle === "[content filtered for security]" || safePath === "[content filtered for security]") continue;

    const typeTag = r.contentType !== "note" ? ` (${r.contentType})` : "";
    const entry = `**${safeTitle}**${typeTag}\n${safePath}\n${snippet}`;

    const entryTokens = estimateTokens(entry);
    if (totalTokens + entryTokens > MAX_TOKEN_BUDGET && lines.length > 0) break;

    lines.push(entry);
    paths.push(r.displayPath);
    totalTokens += entryTokens;
  }

  return {
    context: lines.join("\n\n---\n\n"),
    paths,
    tokens: totalTokens,
  };
}
