#!/usr/bin/env bun
/**
 * ClawMem CLI - Hybrid agent memory (QMD search + SAME memory layer)
 */

import { parseArgs } from "util";
import { existsSync, mkdirSync, readFileSync } from "fs";
import { resolve as pathResolve, basename } from "path";
import {
  createStore,
  enableProductionMode,
  getDefaultDbPath,
  type Store,
  type SearchResult,
  DEFAULT_EMBED_MODEL,
  DEFAULT_QUERY_MODEL,
  DEFAULT_RERANK_MODEL,
  DEFAULT_GLOB,
  extractSnippet,
} from "./store.ts";
import {
  getDefaultLlamaCpp,
  setDefaultLlamaCpp,
  disposeDefaultLlamaCpp,
  formatDocForEmbedding,
  formatQueryForEmbedding,
  LlamaCpp,
  type Queryable,
} from "./llm.ts";
import {
  loadConfig,
  addCollection as collectionsAdd,
  removeCollection as collectionsRemove,
  listCollections as collectionsList,
  getCollection,
  isValidCollectionName,
  getConfigPath,
} from "./collections.ts";
import { formatSearchResults, type OutputFormat } from "./formatter.ts";
import { indexCollection, parseDocument } from "./indexer.ts";
import { detectBeadsProject } from "./beads.ts";
import { applyCompositeScoring, hasRecencyIntent, type EnrichedResult } from "./memory.ts";
import { enrichResults, reciprocalRankFusion, toRanked, type RankedResult } from "./search-utils.ts";
import { splitDocument } from "./splitter.ts";
import { getProfile, updateProfile, isProfileStale } from "./profile.ts";
import { regenerateAllDirectoryContexts } from "./directory-context.ts";
import { readHookInput, writeHookOutput, makeEmptyOutput, type HookOutput } from "./hooks.ts";
import { contextSurfacing } from "./hooks/context-surfacing.ts";
import { sessionBootstrap } from "./hooks/session-bootstrap.ts";
import { decisionExtractor } from "./hooks/decision-extractor.ts";
import { handoffGenerator } from "./hooks/handoff-generator.ts";
import { feedbackLoop } from "./hooks/feedback-loop.ts";
import { stalenessCheck } from "./hooks/staleness-check.ts";

enableProductionMode();

// =============================================================================
// Store lifecycle
// =============================================================================

let store: Store | null = null;

function getStore(): Store {
  if (!store) {
    store = createStore();
  }
  return store;
}

function closeStore(): void {
  if (store) {
    store.close();
    store = null;
  }
}

// =============================================================================
// Terminal colors
// =============================================================================

const useColor = !process.env.NO_COLOR && process.stdout.isTTY;
const c = {
  reset: useColor ? "\x1b[0m" : "",
  dim: useColor ? "\x1b[2m" : "",
  bold: useColor ? "\x1b[1m" : "",
  cyan: useColor ? "\x1b[36m" : "",
  yellow: useColor ? "\x1b[33m" : "",
  green: useColor ? "\x1b[32m" : "",
  red: useColor ? "\x1b[31m" : "",
  magenta: useColor ? "\x1b[35m" : "",
  blue: useColor ? "\x1b[34m" : "",
};

// =============================================================================
// Helpers
// =============================================================================

function die(msg: string): never {
  console.error(`${c.red}Error:${c.reset} ${msg}`);
  process.exit(1);
}

// =============================================================================
// Commands
// =============================================================================

async function cmdInit() {
  const cacheDir = getDefaultDbPath().replace(/\/[^/]+$/, "");
  if (!existsSync(cacheDir)) {
    mkdirSync(cacheDir, { recursive: true });
  }

  // Create store (initializes DB)
  const s = getStore();
  const configPath = getConfigPath();

  console.log(`${c.green}ClawMem initialized${c.reset}`);
  console.log(`  Database: ${s.dbPath}`);
  console.log(`  Config:   ${configPath}`);
  console.log();
  console.log("Next steps:");
  console.log(`  clawmem collection add ~/notes --name notes`);
  console.log(`  clawmem update`);
  console.log(`  clawmem embed`);
}

async function cmdCollectionAdd(args: string[]) {
  const { values, positionals } = parseArgs({
    args,
    options: {
      name: { type: "string" },
      pattern: { type: "string", default: DEFAULT_GLOB },
    },
    allowPositionals: true,
  });

  const dirPath = positionals[0];
  if (!dirPath) die("Usage: clawmem collection add <path> --name <name>");

  const absPath = pathResolve(dirPath);
  if (!existsSync(absPath)) die(`Directory not found: ${absPath}`);

  const name = values.name || basename(absPath).toLowerCase().replace(/[^a-z0-9_-]/g, "-");
  if (!isValidCollectionName(name)) die(`Invalid collection name: ${name}`);

  collectionsAdd(name, absPath, values.pattern);
  console.log(`${c.green}Added collection '${name}'${c.reset} → ${absPath}`);
  console.log(`  Pattern: ${values.pattern}`);
  console.log();
  console.log(`Run ${c.cyan}clawmem update${c.reset} to index files`);
}

async function cmdCollectionList() {
  const collections = collectionsList();
  if (collections.length === 0) {
    console.log("No collections configured.");
    console.log(`Add one with: ${c.cyan}clawmem collection add <path> --name <name>${c.reset}`);
    return;
  }

  for (const col of collections) {
    const s = getStore();
    const count = (s.db.prepare(
      "SELECT COUNT(*) as c FROM documents WHERE collection = ? AND active = 1"
    ).get(col.name) as { c: number }).c;

    console.log(`${c.bold}${col.name}${c.reset}`);
    console.log(`  Path:     ${col.path}`);
    console.log(`  Pattern:  ${col.pattern}`);
    console.log(`  Files:    ${count}`);
    if (col.update) console.log(`  Update:   ${col.update}`);
    console.log();
  }
}

async function cmdCollectionRemove(args: string[]) {
  const name = args[0];
  if (!name) die("Usage: clawmem collection remove <name>");

  if (collectionsRemove(name)) {
    console.log(`${c.green}Removed collection '${name}'${c.reset}`);
  } else {
    die(`Collection '${name}' not found`);
  }
}

async function cmdUpdate(args: string[]) {
  const { values } = parseArgs({
    args,
    options: {
      pull: { type: "boolean", default: false },
      embed: { type: "boolean", default: false },
    },
    allowPositionals: false,
  });

  const collections = collectionsList();
  if (collections.length === 0) die("No collections configured. Add one first.");

  const s = getStore();

  for (const col of collections) {
    // Run pre-update command if configured
    if (values.pull && col.update) {
      console.log(`${c.dim}Running: ${col.update}${c.reset}`);
      const result = Bun.spawnSync(["bash", "-c", col.update], { cwd: col.path });
      if (result.exitCode !== 0) {
        console.error(`${c.yellow}Warning: update command failed for ${col.name}${c.reset}`);
      }
    }

    console.log(`${c.cyan}Indexing ${col.name}${c.reset} (${col.path})`);
    const stats = await indexCollection(s, col.name, col.path, col.pattern);
    console.log(`  ${c.green}+${stats.added}${c.reset} added, ${c.yellow}~${stats.updated}${c.reset} updated, ${c.dim}=${stats.unchanged}${c.reset} unchanged, ${c.red}-${stats.removed}${c.reset} removed`);
  }

  // Auto-embed if --embed flag is set
  if (values.embed) {
    console.log();
    await cmdEmbed([]);
  } else {
    console.log();
    console.log(`Run ${c.cyan}clawmem embed${c.reset} to generate embeddings for new content`);
  }

  // Auto-rebuild profile if stale
  if (isProfileStale(s)) {
    updateProfile(s);
    console.log(`${c.dim}Profile auto-rebuilt (stale)${c.reset}`);
  }
}

async function cmdEmbed(args: string[]) {
  const { values } = parseArgs({
    args,
    options: { force: { type: "boolean", short: "f", default: false } },
    allowPositionals: false,
  });

  const s = getStore();

  if (values.force) {
    console.log(`${c.yellow}Force mode: clearing all embeddings${c.reset}`);
    s.clearAllEmbeddings();
  }

  // Use fragment-based pipeline: split documents into semantic fragments and embed each
  const hashes = s.getHashesNeedingFragments();
  if (hashes.length === 0) {
    console.log(`${c.green}All documents already embedded${c.reset}`);
    return;
  }

  // Count total fragments first for ETA
  let totalFragEstimate = 0;
  const docFragCounts: number[] = [];
  for (const { body, path } of hashes) {
    let frontmatter: Record<string, any> | undefined;
    try {
      const parsed = parseDocument(body, path);
      frontmatter = parsed.meta as any;
    } catch { /* skip */ }
    const frags = splitDocument(body, frontmatter);
    docFragCounts.push(frags.length);
    totalFragEstimate += frags.length;
  }
  console.log(`Embedding ${hashes.length} documents (${totalFragEstimate} fragments total)...`);

  const embedUrl = process.env.CLAWMEM_EMBED_URL;
  if (embedUrl) {
    console.log(`Using remote GPU embedding: ${embedUrl}`);
  } else {
    // Local CPU mode: disable inactivity timeout to prevent context disposal mid-batch
    setDefaultLlamaCpp(new LlamaCpp({ inactivityTimeoutMs: 0 }));
  }
  const llm = getDefaultLlamaCpp();
  let embedded = 0;
  let totalFragments = 0;
  let failedFragments = 0;
  const batchStart = Date.now();

  for (let docIdx = 0; docIdx < hashes.length; docIdx++) {
    const { hash, body, path, title: docTitle } = hashes[docIdx]!;
    const title = docTitle || basename(path).replace(/\.(md|txt)$/i, "");

    // Parse frontmatter for fragment splitting
    let frontmatter: Record<string, any> | undefined;
    try {
      const parsed = parseDocument(body, path);
      frontmatter = parsed.meta as any;
    } catch {
      // No frontmatter or parsing error — fine, skip it
    }

    const fragments = splitDocument(body, frontmatter);
    const docStart = Date.now();
    console.error(`  [${docIdx + 1}/${hashes.length}] ${basename(path)} (${fragments.length} frags, ${body.length} chars)`);

    for (let seq = 0; seq < fragments.length; seq++) {
      const frag = fragments[seq]!;
      const label = frag.label || title;
      const text = formatDocForEmbedding(frag.content, label);

      try {
        const fragStart = Date.now();
        const result = await llm.embed(text);
        const fragMs = Date.now() - fragStart;
        if (result) {
          s.ensureVecTable(result.embedding.length);
          s.insertEmbedding(
            hash, seq, frag.startLine, new Float32Array(result.embedding),
            result.model, new Date().toISOString(), frag.type, frag.label ?? undefined
          );
          totalFragments++;
          if (seq === 0 || (seq + 1) % 5 === 0 || seq === fragments.length - 1) {
            console.error(`    frag ${seq + 1}/${fragments.length} (${frag.type}) ${fragMs}ms [${text.length} chars]`);
          }
        } else {
          failedFragments++;
          console.error(`    frag ${seq + 1}/${fragments.length} (${frag.type}) → null result [${text.length} chars]`);
        }
      } catch (err) {
        failedFragments++;
        console.error(`${c.yellow}Warning: failed to embed fragment ${seq} (${frag.type}) of ${path}: ${err}${c.reset}`);
      }
    }

    embedded++;
    const docMs = Date.now() - docStart;
    const elapsed = ((Date.now() - batchStart) / 1000).toFixed(0);
    console.error(`  → doc done in ${(docMs / 1000).toFixed(1)}s | ${embedded}/${hashes.length} docs, ${totalFragments} frags, ${failedFragments} fails [${elapsed}s elapsed]`);
  }

  const totalSec = ((Date.now() - batchStart) / 1000).toFixed(1);
  console.log();
  console.log(`${c.green}Embedded ${embedded} documents (${totalFragments} fragments, ${failedFragments} failed) in ${totalSec}s${c.reset}`);

  await disposeDefaultLlamaCpp();
}

async function cmdStatus() {
  const s = getStore();
  const status = s.getStatus();

  console.log(`${c.bold}ClawMem Status${c.reset}`);
  console.log(`  Database:   ${s.dbPath}`);
  console.log(`  Documents:  ${status.totalDocuments}`);
  console.log(`  Unembedded: ${status.needsEmbedding}`);
  console.log(`  Vectors:    ${status.hasVectorIndex ? "yes" : "no"}`);
  console.log();

  if (status.collections.length > 0) {
    console.log(`${c.bold}Collections:${c.reset}`);
    for (const col of status.collections) {
      console.log(`  ${col.name}: ${col.documents} docs (${col.path})`);
    }
  }

  // SAME metadata stats
  const types = s.db.prepare(`
    SELECT content_type, COUNT(*) as cnt FROM documents WHERE active = 1 GROUP BY content_type ORDER BY cnt DESC
  `).all() as { content_type: string; cnt: number }[];

  if (types.length > 0) {
    console.log();
    console.log(`${c.bold}Content Types:${c.reset}`);
    for (const t of types) {
      console.log(`  ${t.content_type}: ${t.cnt}`);
    }
  }

  const sessions = s.db.prepare("SELECT COUNT(*) as cnt FROM session_log").get() as { cnt: number };
  if (sessions.cnt > 0) {
    console.log();
    console.log(`${c.bold}Sessions:${c.reset} ${sessions.cnt} tracked`);
  }
}

async function cmdSearch(args: string[]) {
  const { values, positionals } = parseArgs({
    args,
    options: {
      num: { type: "string", short: "n", default: "10" },
      collection: { type: "string", short: "c" },
      json: { type: "boolean", default: false },
      "min-score": { type: "string", default: "0" },
    },
    allowPositionals: true,
  });

  const query = positionals.join(" ");
  if (!query) die("Usage: clawmem search <query>");

  const s = getStore();
  const limit = parseInt(values.num!, 10);
  const minScore = parseFloat(values["min-score"]!);

  const results = s.searchFTS(query, limit * 2);
  const enriched = enrichResults(s, results, query);
  const scored = applyCompositeScoring(enriched, query)
    .filter(r => r.compositeScore >= minScore)
    .slice(0, limit);

  if (values.json) {
    console.log(JSON.stringify(scored.map(r => ({
      file: r.displayPath,
      title: r.title,
      score: r.compositeScore,
      searchScore: r.score,
      recencyScore: r.recencyScore,
      contentType: r.contentType,
    })), null, 2));
  } else {
    printResults(scored, query);
  }
}

async function cmdVsearch(args: string[]) {
  const { values, positionals } = parseArgs({
    args,
    options: {
      num: { type: "string", short: "n", default: "10" },
      collection: { type: "string", short: "c" },
      json: { type: "boolean", default: false },
      "min-score": { type: "string", default: "0.3" },
    },
    allowPositionals: true,
  });

  const query = positionals.join(" ");
  if (!query) die("Usage: clawmem vsearch <query>");

  const s = getStore();
  const limit = parseInt(values.num!, 10);
  const minScore = parseFloat(values["min-score"]!);

  const results = await s.searchVec(query, DEFAULT_EMBED_MODEL, limit * 2);
  const enriched = enrichResults(s, results, query);
  const scored = applyCompositeScoring(enriched, query)
    .filter(r => r.compositeScore >= minScore)
    .slice(0, limit);

  if (values.json) {
    console.log(JSON.stringify(scored.map(r => ({
      file: r.displayPath,
      title: r.title,
      score: r.compositeScore,
      searchScore: r.score,
      recencyScore: r.recencyScore,
      contentType: r.contentType,
    })), null, 2));
  } else {
    printResults(scored, query);
  }

  await disposeDefaultLlamaCpp();
}

async function cmdQuery(args: string[]) {
  const { values, positionals } = parseArgs({
    args,
    options: {
      num: { type: "string", short: "n", default: "10" },
      collection: { type: "string", short: "c" },
      json: { type: "boolean", default: false },
      "min-score": { type: "string", default: "0" },
    },
    allowPositionals: true,
  });

  const query = positionals.join(" ");
  if (!query) die("Usage: clawmem query <query>");

  const s = getStore();
  const limit = parseInt(values.num!, 10);
  const minScore = parseFloat(values["min-score"]!);

  // Step 1: BM25 for strong signal check
  const ftsResults = s.searchFTS(query, 20);
  const topScore = ftsResults[0]?.score ?? 0;
  const secondScore = ftsResults[1]?.score ?? 0;
  const strongSignal = topScore >= 0.85 && (topScore - secondScore) >= 0.15;

  // Step 2: Query expansion (skip if strong BM25 signal)
  let expandedQueries: { type: string; text: string }[] = [];
  if (!strongSignal) {
    try {
      const expanded = await s.expandQuery(query, DEFAULT_QUERY_MODEL);
      expandedQueries = expanded.map(text => {
        // Parse "type: text" format from expansion
        const colonIdx = text.indexOf(": ");
        if (colonIdx > 0 && colonIdx < 5) {
          return { type: text.slice(0, colonIdx), text: text.slice(colonIdx + 2) };
        }
        return { type: "vec", text };
      });
    } catch {
      // Fallback: no expansion
    }
  }

  // Step 3: Parallel searches
  const allRanked: { results: RankedResult[]; weight: number }[] = [];

  // Original query BM25 + vec (weight 2x)
  allRanked.push({ results: ftsResults.map(toRanked), weight: 2 });
  const vecResults = await s.searchVec(query, DEFAULT_EMBED_MODEL, 20);
  allRanked.push({ results: vecResults.map(toRanked), weight: 2 });

  // Expanded queries (weight 1x)
  for (const eq of expandedQueries) {
    if (eq.type === "lex") {
      const r = s.searchFTS(eq.text, 20);
      allRanked.push({ results: r.map(toRanked), weight: 1 });
    } else {
      const r = await s.searchVec(eq.text, DEFAULT_EMBED_MODEL, 20);
      allRanked.push({ results: r.map(toRanked), weight: 1 });
    }
  }

  // Step 4: RRF fusion
  const rrfResults = reciprocalRankFusion(
    allRanked.map(a => a.results),
    allRanked.map(a => a.weight),
    60
  );

  // Step 5: Take top 30 for reranking
  const candidates = rrfResults.slice(0, 30);

  // Step 6: Rerank
  let reranked: { file: string; score: number }[] = [];
  try {
    const docs = candidates.map(r => ({ file: r.file, text: r.body.slice(0, 4000) }));
    reranked = await s.rerank(query, docs, DEFAULT_RERANK_MODEL);
  } catch {
    reranked = candidates.map(r => ({ file: r.file, score: r.score }));
  }

  // Step 7: Position-aware blending
  const rrfRankMap = new Map(candidates.map((r, i) => [r.file, i + 1]));
  const blended = reranked.map(r => {
    const rrfRank = rrfRankMap.get(r.file) || candidates.length;
    let rrfWeight: number;
    if (rrfRank <= 3) rrfWeight = 0.75;
    else if (rrfRank <= 10) rrfWeight = 0.60;
    else rrfWeight = 0.40;

    const blendedScore = rrfWeight * (1 / rrfRank) + (1 - rrfWeight) * r.score;
    return { file: r.file, score: blendedScore };
  });
  blended.sort((a, b) => b.score - a.score);

  // Step 8: Map back to full results and apply composite scoring
  const resultMap = new Map(
    [...ftsResults, ...vecResults].map(r => [r.filepath, r])
  );
  const fullResults = blended
    .map(b => resultMap.get(b.file))
    .filter((r): r is SearchResult => r !== undefined)
    .map(r => ({ ...r, score: blended.find(b => b.file === r.filepath)?.score ?? r.score }));

  const enriched = enrichResults(s, fullResults, query);
  const scored = applyCompositeScoring(enriched, query)
    .filter(r => r.compositeScore >= minScore)
    .slice(0, limit);

  if (values.json) {
    console.log(JSON.stringify(scored.map(r => ({
      file: r.displayPath,
      title: r.title,
      score: r.compositeScore,
      searchScore: r.score,
      recencyScore: r.recencyScore,
      contentType: r.contentType,
    })), null, 2));
  } else {
    printResults(scored, query);
  }

  await disposeDefaultLlamaCpp();
}

function printResults(results: Array<{ displayPath: string; title: string; compositeScore: number; score: number; contentType: string; body?: string }>, query: string) {
  if (results.length === 0) {
    console.log(`${c.dim}No results found${c.reset}`);
    return;
  }

  for (const r of results) {
    const scoreBar = "█".repeat(Math.round(r.compositeScore * 10));
    const scoreStr = r.compositeScore.toFixed(2);
    const typeTag = r.contentType !== "note" ? ` ${c.magenta}[${r.contentType}]${c.reset}` : "";
    console.log(`${c.cyan}${scoreStr}${c.reset} ${c.dim}${scoreBar}${c.reset} ${c.bold}${r.title}${c.reset}${typeTag}`);
    console.log(`  ${c.dim}${r.displayPath}${c.reset}`);

    if (r.body) {
      const snippet = extractSnippet(r.body, query, 200);
      const lines = snippet.snippet.split("\n").slice(1, 4); // Skip header line
      for (const line of lines) {
        console.log(`  ${c.dim}${line.trim()}${c.reset}`);
      }
    }
    console.log();
  }
}

// =============================================================================
// Hook dispatch
// =============================================================================

async function cmdHook(args: string[]) {
  const hookName = args[0];
  if (!hookName) die("Usage: clawmem hook <name>");

  const input = await readHookInput();
  const s = getStore();
  let output: HookOutput;

  try {
    switch (hookName) {
      case "context-surfacing":
        output = await contextSurfacing(s, input);
        break;
      case "session-bootstrap":
        output = await sessionBootstrap(s, input);
        break;
      case "decision-extractor":
        output = await decisionExtractor(s, input);
        break;
      case "handoff-generator":
        output = await handoffGenerator(s, input);
        break;
      case "feedback-loop":
        output = await feedbackLoop(s, input);
        break;
      case "staleness-check":
        output = await stalenessCheck(s, input);
        break;
      default:
        die(`Unknown hook: ${hookName}. Available: context-surfacing, session-bootstrap, decision-extractor, handoff-generator, feedback-loop, staleness-check`);
        output = makeEmptyOutput(); // unreachable, satisfies TS
    }
  } catch (err) {
    // Hooks must never crash — silent fallback
    console.error(`Hook ${hookName} error: ${err}`);
    output = makeEmptyOutput(hookName);
  }

  writeHookOutput(output);
}

async function cmdBudget(args: string[]) {
  const { values } = parseArgs({
    args,
    options: { session: { type: "string" }, last: { type: "string", default: "5" } },
    allowPositionals: false,
  });

  const s = getStore();

  if (values.session) {
    const usages = s.getUsageForSession(values.session);
    if (usages.length === 0) {
      console.log(`No usage records for session ${values.session}`);
      return;
    }
    for (const u of usages) {
      const paths = JSON.parse(u.injectedPaths) as string[];
      console.log(`${c.dim}${u.timestamp}${c.reset} ${c.cyan}${u.hookName}${c.reset} ${u.estimatedTokens} tokens, ${paths.length} notes ${u.wasReferenced ? c.green + "referenced" + c.reset : c.dim + "not referenced" + c.reset}`);
    }
  } else {
    const sessions = s.getRecentSessions(parseInt(values.last!, 10));
    if (sessions.length === 0) {
      console.log("No sessions tracked yet.");
      return;
    }
    for (const sess of sessions) {
      const usages = s.getUsageForSession(sess.sessionId);
      const totalTokens = usages.reduce((sum, u) => sum + u.estimatedTokens, 0);
      const refCount = usages.filter(u => u.wasReferenced).length;
      console.log(`${c.bold}${sess.sessionId.slice(0, 8)}${c.reset} ${c.dim}${sess.startedAt}${c.reset} ${totalTokens} tokens, ${refCount}/${usages.length} referenced`);
      if (sess.summary) console.log(`  ${c.dim}${sess.summary.slice(0, 80)}${c.reset}`);
    }
  }
}

async function cmdLog(args: string[]) {
  const { values } = parseArgs({
    args,
    options: { last: { type: "string", default: "10" } },
    allowPositionals: false,
  });

  const s = getStore();
  const sessions = s.getRecentSessions(parseInt(values.last!, 10));

  if (sessions.length === 0) {
    console.log("No sessions tracked.");
    return;
  }

  for (const sess of sessions) {
    const duration = sess.endedAt
      ? `${Math.round((new Date(sess.endedAt).getTime() - new Date(sess.startedAt).getTime()) / 60000)}min`
      : "active";
    console.log(`${c.bold}${sess.sessionId.slice(0, 8)}${c.reset} ${c.dim}${sess.startedAt}${c.reset} (${duration})`);
    if (sess.handoffPath) console.log(`  Handoff: ${sess.handoffPath}`);
    if (sess.summary) console.log(`  ${sess.summary.slice(0, 100)}`);
    if (sess.filesChanged.length > 0) console.log(`  Files: ${sess.filesChanged.slice(0, 5).join(", ")}`);
    console.log();
  }
}

// =============================================================================
// MCP Server
// =============================================================================

async function cmdMcp() {
  enableMcpStdioMode();
  const { startMcpServer } = await import("./mcp.ts");
  await startMcpServer();
}

// In MCP stdio mode, stdout is reserved exclusively for JSON-RPC messages.
// Any accidental console.log/info/debug/warn output will corrupt the protocol stream.
function enableMcpStdioMode(): void {
  process.env.CLAWMEM_STDIO_MODE = "true";
  if (!process.env.NO_COLOR) process.env.NO_COLOR = "1";

  const err = console.error.bind(console);
  // Bun's console properties are writable; still guard in case of future changes.
  try { (console as any).log = err; } catch {}
  try { (console as any).info = err; } catch {}
  try { (console as any).debug = err; } catch {}
  try { (console as any).warn = err; } catch {}
}

// =============================================================================
// Setup Commands
// =============================================================================

async function cmdSetup(args: string[]) {
  const subCmd = args[0];
  switch (subCmd) {
    case "hooks": await cmdSetupHooks(args.slice(1)); break;
    case "mcp": await cmdSetupMcp(args.slice(1)); break;
    default: die("Usage: clawmem setup <hooks|mcp> [--remove]");
  }
}

async function cmdSetupHooks(args: string[]) {
  const remove = args.includes("--remove");
  const settingsPath = pathResolve(process.env.HOME || "~", ".claude", "settings.json");

  // Find clawmem binary
  const binPath = findClawmemBinary();

  let settings: any = {};
  if (existsSync(settingsPath)) {
    settings = JSON.parse(readFileSync(settingsPath, "utf-8"));
  }

  if (!settings.hooks) settings.hooks = {};

  if (remove) {
    // Remove clawmem hooks
    for (const event of ["UserPromptSubmit", "Stop", "SessionStart"]) {
      if (settings.hooks[event]) {
        settings.hooks[event] = settings.hooks[event].filter((entry: any) =>
          !entry.hooks?.some((h: any) => h.command?.includes("clawmem"))
        );
        if (settings.hooks[event].length === 0) delete settings.hooks[event];
      }
    }
    console.log(`${c.green}Removed ClawMem hooks from ${settingsPath}${c.reset}`);
  } else {
    // Install clawmem hooks
    const hookConfig: Record<string, string[]> = {
      UserPromptSubmit: ["context-surfacing"],
      SessionStart: ["session-bootstrap", "staleness-check"],
      Stop: ["decision-extractor", "handoff-generator", "feedback-loop"],
    };

    for (const [event, hooks] of Object.entries(hookConfig)) {
      if (!settings.hooks[event]) settings.hooks[event] = [];

      // Remove existing clawmem entries first
      settings.hooks[event] = settings.hooks[event].filter((entry: any) =>
        !entry.hooks?.some((h: any) => h.command?.includes("clawmem"))
      );

      // Add new entries
      settings.hooks[event].push({
        matcher: "",
        hooks: hooks.map(name => ({
          type: "command",
          command: `${binPath} hook ${name}`,
        })),
      });
    }

    console.log(`${c.green}Installed ClawMem hooks to ${settingsPath}${c.reset}`);
    for (const [event, hooks] of Object.entries(hookConfig)) {
      console.log(`  ${event}: ${hooks.join(", ")}`);
    }
  }

  const { writeFileSync: wfs } = await import("fs");
  const dir = pathResolve(process.env.HOME || "~", ".claude");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  wfs(settingsPath, JSON.stringify(settings, null, 2) + "\n");
}

async function cmdSetupMcp(args: string[]) {
  const remove = args.includes("--remove");
  const claudeJsonPath = pathResolve(process.env.HOME || "~", ".claude.json");

  let config: any = {};
  if (existsSync(claudeJsonPath)) {
    config = JSON.parse(readFileSync(claudeJsonPath, "utf-8"));
  }

  if (!config.mcpServers) config.mcpServers = {};

  if (remove) {
    delete config.mcpServers.clawmem;
    console.log(`${c.green}Removed ClawMem MCP from ${claudeJsonPath}${c.reset}`);
  } else {
    const binPath = findClawmemBinary();
    config.mcpServers.clawmem = {
      command: binPath,
      args: ["mcp"],
    };
    console.log(`${c.green}Registered ClawMem MCP in ${claudeJsonPath}${c.reset}`);
    console.log(`  Command: ${binPath} mcp`);
  }

  const { writeFileSync: wfs } = await import("fs");
  wfs(claudeJsonPath, JSON.stringify(config, null, 2) + "\n");
}

function findClawmemBinary(): string {
  // Check common locations
  const candidates = [
    pathResolve(import.meta.dir, "..", "bin", "clawmem"),
    pathResolve(process.env.HOME || "~", ".local", "bin", "clawmem"),
    "/usr/local/bin/clawmem",
  ];
  for (const p of candidates) {
    if (existsSync(p)) return p;
  }
  return "clawmem"; // Assume in PATH
}

// =============================================================================
// Watch (File Watcher Daemon)
// =============================================================================

async function cmdWatch() {
  const { startWatcher } = await import("./watcher.ts");
  const collections = collectionsList()
    .sort((a, b) => b.path.length - a.path.length); // Most specific path first for prefix matching

  if (collections.length === 0) {
    die("No collections configured. Add one first: clawmem collection add <path> --name <name>");
  }

  const dirs = collections.map(col => col.path);
  const s = getStore();

  console.log(`${c.bold}Watching ${dirs.length} collection(s) for changes...${c.reset}`);
  for (const col of collections) {
    console.log(`  ${c.dim}${col.name}: ${col.path}${c.reset}`);
  }
  console.log(`${c.dim}Press Ctrl+C to stop.${c.reset}`);

  const watcher = startWatcher(dirs, {
    debounceMs: 2000,
    onChanged: async (fullPath, event) => {
      // Find which collection this belongs to
      const col = collections.find(c => fullPath.startsWith(c.path));
      if (!col) return;

      const relativePath = fullPath.slice(col.path.length + 1);
      console.log(`${c.dim}[${event}]${c.reset} ${col.name}/${relativePath}`);

      // Beads JSONL: trigger sync instead of regular index
      if (fullPath.endsWith(".jsonl") && fullPath.includes(".beads/")) {
        const beadsPath = detectBeadsProject(fullPath.replace(/\/\.beads\/.*$/, ""));
        if (beadsPath) {
          const result = await s.syncBeadsIssues(beadsPath);
          console.log(`  beads: +${result.created} ~${result.synced}`);
        }
        return;
      }

      // Re-index just this collection
      const stats = await indexCollection(s, col.name, col.path, col.pattern);
      if (stats.added > 0 || stats.updated > 0 || stats.removed > 0) {
        console.log(`  +${stats.added} ~${stats.updated} -${stats.removed}`);
      }
    },
    onError: (err) => {
      console.error(`${c.red}Watch error: ${err.message}${c.reset}`);
    },
  });

  // Keep running until Ctrl+C
  process.on("SIGINT", () => {
    watcher.close();
    closeStore();
    process.exit(0);
  });

  // Block forever
  await new Promise(() => {});
}

// =============================================================================
// Reindex
// =============================================================================

async function cmdReindex(args: string[]) {
  const force = args.includes("--force") || args.includes("-f");
  const collections = collectionsList();

  if (collections.length === 0) {
    die("No collections configured.");
  }

  const s = getStore();

  if (force) {
    // Delete all documents and re-scan
    s.db.exec("UPDATE documents SET active = 0");
    console.log(`${c.yellow}Force reindex: deactivated all documents${c.reset}`);
  }

  for (const col of collections) {
    console.log(`Indexing ${c.bold}${col.name}${c.reset} (${col.path})...`);
    const stats = await indexCollection(s, col.name, col.path, col.pattern);
    console.log(`  +${stats.added} added, ~${stats.updated} updated, =${stats.unchanged} unchanged, -${stats.removed} removed`);
  }
}

// =============================================================================
// Doctor (Health Check)
// =============================================================================

async function cmdDoctor() {
  console.log(`${c.bold}ClawMem Doctor${c.reset}\n`);
  let issues = 0;

  // 1. Database
  try {
    const s = getStore();
    const docCount = (s.db.prepare("SELECT COUNT(*) as n FROM documents WHERE active = 1").get() as any).n;
    console.log(`${c.green}✓${c.reset} Database: ${s.dbPath} (${docCount} documents)`);
  } catch (err) {
    console.log(`${c.red}✗${c.reset} Database: ${err}`);
    issues++;
  }

  // 2. Collections
  try {
    const collections = collectionsList();
    if (collections.length === 0) {
      console.log(`${c.yellow}!${c.reset} No collections configured`);
      issues++;
    } else {
      for (const col of collections) {
        if (existsSync(col.path)) {
          console.log(`${c.green}✓${c.reset} Collection "${col.name}": ${col.path}`);
        } else {
          console.log(`${c.red}✗${c.reset} Collection "${col.name}": ${col.path} (directory not found)`);
          issues++;
        }
      }
    }
  } catch (err) {
    console.log(`${c.red}✗${c.reset} Collections config: ${err}`);
    issues++;
  }

  // 3. Embeddings
  try {
    const s = getStore();
    const needsEmbed = s.getHashesNeedingEmbedding();
    const hasVectors = !!s.db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'").get();
    if (hasVectors) {
      console.log(`${c.green}✓${c.reset} Vector index: exists (${needsEmbed} need embedding)`);
    } else {
      console.log(`${c.yellow}!${c.reset} Vector index: not created yet (run 'clawmem embed')`);
    }
  } catch {
    console.log(`${c.yellow}!${c.reset} Vector index: could not check`);
  }

  // 4. Content types
  try {
    const s = getStore();
    const types = s.db.prepare(
      "SELECT content_type, COUNT(*) as n FROM documents WHERE active = 1 GROUP BY content_type"
    ).all() as { content_type: string; n: number }[];
    const typeStr = types.map(t => `${t.content_type}:${t.n}`).join(", ");
    console.log(`${c.green}✓${c.reset} Content types: ${typeStr || "none"}`);
  } catch {
    // Skip
  }

  // 5. Hooks installed
  try {
    const settingsPath = pathResolve(process.env.HOME || "~", ".claude", "settings.json");
    if (existsSync(settingsPath)) {
      const settings = JSON.parse(readFileSync(settingsPath, "utf-8"));
      const hasHooks = Object.values(settings.hooks || {}).some((arr: any) =>
        Array.isArray(arr) && arr.some((entry: any) =>
          entry.hooks?.some((h: any) => h.command?.includes("clawmem"))
        )
      );
      if (hasHooks) {
        console.log(`${c.green}✓${c.reset} Claude Code hooks: installed`);
      } else {
        console.log(`${c.yellow}!${c.reset} Claude Code hooks: not installed (run 'clawmem setup hooks')`);
      }
    } else {
      console.log(`${c.yellow}!${c.reset} Claude Code hooks: settings.json not found`);
    }
  } catch {
    console.log(`${c.yellow}!${c.reset} Claude Code hooks: could not check`);
  }

  // 6. MCP registered
  try {
    const claudeJsonPath = pathResolve(process.env.HOME || "~", ".claude.json");
    if (existsSync(claudeJsonPath)) {
      const config = JSON.parse(readFileSync(claudeJsonPath, "utf-8"));
      if (config.mcpServers?.clawmem) {
        console.log(`${c.green}✓${c.reset} MCP server: registered in ~/.claude.json`);
      } else {
        console.log(`${c.yellow}!${c.reset} MCP server: not registered (run 'clawmem setup mcp')`);
      }
    }
  } catch {
    console.log(`${c.yellow}!${c.reset} MCP server: could not check`);
  }

  // 7. Sessions
  try {
    const s = getStore();
    const sessions = s.getRecentSessions(1);
    if (sessions.length > 0) {
      console.log(`${c.green}✓${c.reset} Sessions: last session ${sessions[0]!.startedAt}`);
    } else {
      console.log(`${c.dim}-${c.reset} Sessions: none tracked yet`);
    }
  } catch {
    // Skip
  }

  console.log();
  if (issues > 0) {
    console.log(`${c.yellow}${issues} issue(s) found.${c.reset}`);
  } else {
    console.log(`${c.green}All checks passed.${c.reset}`);
  }
}

// =============================================================================
// Bootstrap
// =============================================================================

async function cmdBootstrap(args: string[]) {
  const { values, positionals } = parseArgs({
    args,
    options: {
      name: { type: "string" },
      "skip-embed": { type: "boolean", default: false },
    },
    allowPositionals: true,
  });

  const vaultPath = positionals[0];
  if (!vaultPath) die("Usage: clawmem bootstrap <vault-path> [--name <name>] [--skip-embed]");

  const absPath = pathResolve(vaultPath);
  if (!existsSync(absPath)) die(`Directory not found: ${absPath}`);

  const name = values.name || basename(absPath).toLowerCase().replace(/[^a-z0-9_-]/g, "-");

  // 1. Init (skip if already initialized)
  const dbPath = getDefaultDbPath();
  if (!existsSync(dbPath)) {
    console.log(`${c.cyan}Step 1: Initializing ClawMem${c.reset}`);
    await cmdInit();
  } else {
    console.log(`${c.dim}Step 1: Already initialized${c.reset}`);
  }

  // 2. Collection add (skip if already exists)
  const existing = collectionsList().find(col => col.path === absPath);
  if (!existing) {
    console.log(`${c.cyan}Step 2: Adding collection '${name}'${c.reset}`);
    collectionsAdd(name, absPath, DEFAULT_GLOB);
    console.log(`  ${c.green}Added${c.reset} ${absPath}`);
  } else {
    console.log(`${c.dim}Step 2: Collection already exists (${existing.name})${c.reset}`);
  }

  // 3. Update
  console.log(`${c.cyan}Step 3: Indexing files${c.reset}`);
  await cmdUpdate([]);

  // 4. Embed (unless --skip-embed)
  if (!values["skip-embed"]) {
    console.log(`${c.cyan}Step 4: Embedding documents${c.reset}`);
    await cmdEmbed([]);
  } else {
    console.log(`${c.dim}Step 4: Skipping embeddings (--skip-embed)${c.reset}`);
  }

  // 5. Setup hooks
  console.log(`${c.cyan}Step 5: Installing hooks${c.reset}`);
  await cmdSetupHooks([]);

  // 6. Setup MCP
  console.log(`${c.cyan}Step 6: Registering MCP${c.reset}`);
  await cmdSetupMcp([]);

  console.log();
  console.log(`${c.green}ClawMem bootstrapped for ${absPath}${c.reset}`);
}

// =============================================================================
// Install Service
// =============================================================================

async function cmdInstallService(args: string[]) {
  const remove = args.includes("--remove");
  const enable = args.includes("--enable");

  const { join: pathJoin } = await import("path");
  const os = await import("os");
  const { execSync } = await import("child_process");

  const servicePath = pathJoin(os.homedir(), ".config", "systemd", "user", "clawmem-watcher.service");

  if (remove) {
    try { execSync("systemctl --user stop clawmem-watcher.service 2>/dev/null"); } catch { /* may not be running */ }
    try { execSync("systemctl --user disable clawmem-watcher.service 2>/dev/null"); } catch { /* may not be enabled */ }
    if (existsSync(servicePath)) {
      const { unlinkSync } = await import("fs");
      unlinkSync(servicePath);
    }
    execSync("systemctl --user daemon-reload");
    console.log(`${c.green}Removed clawmem-watcher service${c.reset}`);
    return;
  }

  const binPath = findClawmemBinary();
  const unit = `[Unit]
Description=ClawMem File Watcher
After=default.target

[Service]
Type=simple
ExecStart=${process.argv[0]} ${pathResolve(import.meta.dir, "clawmem.ts")} watch
Restart=on-failure
RestartSec=5
Environment=HOME=${os.homedir()}

[Install]
WantedBy=default.target
`;

  const serviceDir = pathJoin(os.homedir(), ".config", "systemd", "user");
  if (!existsSync(serviceDir)) mkdirSync(serviceDir, { recursive: true });

  const { writeFileSync: wfs } = await import("fs");
  wfs(servicePath, unit);
  execSync("systemctl --user daemon-reload");

  console.log(`${c.green}Installed clawmem-watcher service${c.reset}`);
  console.log(`  ${servicePath}`);

  if (enable) {
    execSync("systemctl --user enable --now clawmem-watcher.service");
    console.log(`  ${c.green}Enabled and started${c.reset}`);
  } else {
    console.log(`  Run: ${c.cyan}systemctl --user enable --now clawmem-watcher.service${c.reset}`);
  }
}

// =============================================================================
// Directory Context
// =============================================================================

async function cmdUpdateContext() {
  const s = getStore();
  const count = regenerateAllDirectoryContexts(s);
  console.log(`${c.green}Updated CLAUDE.md in ${count} directories${c.reset}`);
}

// =============================================================================
// Profile
// =============================================================================

async function cmdProfile(args: string[]) {
  const s = getStore();

  if (args[0] === "rebuild") {
    updateProfile(s);
    console.log(`${c.green}Profile rebuilt${c.reset}`);
    return;
  }

  const profile = getProfile(s);
  if (!profile) {
    console.log("No profile found. Run: clawmem profile rebuild");
    return;
  }

  console.log(`${c.bold}User Profile${c.reset}`);
  if (profile.static.length > 0) {
    console.log(`\n${c.cyan}Known Context:${c.reset}`);
    for (const fact of profile.static) {
      console.log(`  - ${fact}`);
    }
  }
  if (profile.dynamic.length > 0) {
    console.log(`\n${c.cyan}Current Focus:${c.reset}`);
    for (const item of profile.dynamic) {
      console.log(`  - ${item}`);
    }
  }
}

// =============================================================================
// Main dispatch
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  const command = args[0];
  const subArgs = args.slice(1);

  try {
    switch (command) {
      case "init":
        await cmdInit();
        break;
      case "collection": {
        const subCmd = subArgs[0];
        const subSubArgs = subArgs.slice(1);
        switch (subCmd) {
          case "add": await cmdCollectionAdd(subSubArgs); break;
          case "list": await cmdCollectionList(); break;
          case "remove": await cmdCollectionRemove(subSubArgs); break;
          default: die("Usage: clawmem collection <add|list|remove>");
        }
        break;
      }
      case "update":
        await cmdUpdate(subArgs);
        break;
      case "embed":
        await cmdEmbed(subArgs);
        break;
      case "status":
        await cmdStatus();
        break;
      case "search":
        await cmdSearch(subArgs);
        break;
      case "vsearch":
        await cmdVsearch(subArgs);
        break;
      case "query":
        await cmdQuery(subArgs);
        break;
      case "hook":
        await cmdHook(subArgs);
        break;
      case "budget":
        await cmdBudget(subArgs);
        break;
      case "log":
        await cmdLog(subArgs);
        break;
      case "mcp":
        await cmdMcp();
        break;
      case "setup":
        await cmdSetup(subArgs);
        break;
      case "watch":
        await cmdWatch();
        break;
      case "reindex":
        await cmdReindex(subArgs);
        break;
      case "doctor":
        await cmdDoctor();
        break;
      case "bootstrap":
        await cmdBootstrap(subArgs);
        break;
      case "install-service":
        await cmdInstallService(subArgs);
        break;
      case "profile":
        await cmdProfile(subArgs);
        break;
      case "update-context":
        await cmdUpdateContext();
        break;
      case "help":
      case "--help":
      case "-h":
      case undefined:
        printHelp();
        break;
      default:
        die(`Unknown command: ${command}. Run 'clawmem help' for usage.`);
    }
  } finally {
    closeStore();
  }
}

function printHelp() {
  console.log(`
${c.bold}ClawMem${c.reset} - Hybrid Agent Memory

${c.bold}Setup:${c.reset}
  clawmem init                         Initialize ClawMem
  clawmem bootstrap <path> [--name N]  One-command setup (init+add+update+embed+hooks+mcp)
  clawmem collection add <path> --name <name>
  clawmem collection list
  clawmem collection remove <name>
  clawmem setup hooks [--remove]       Install/remove Claude Code hooks
  clawmem setup mcp [--remove]         Register/remove MCP in ~/.claude.json
  clawmem install-service [--enable]   Install systemd watcher service

${c.bold}Indexing:${c.reset}
  clawmem update [--pull] [--embed]    Re-scan collections (--embed auto-embeds)
  clawmem embed [-f]                   Generate fragment embeddings
  clawmem reindex [--force]            Full re-index
  clawmem watch                        File watcher daemon
  clawmem status                       Show index status

${c.bold}Search:${c.reset}
  clawmem search <query> [-n N]        BM25 keyword search
  clawmem vsearch <query> [-n N]       Vector similarity
  clawmem query <query> [-n N]         Hybrid + rerank (best)

${c.bold}Memory:${c.reset}
  clawmem budget [--session ID]        Token utilization
  clawmem log [--last N]               Session history
  clawmem profile                      Show user profile
  clawmem profile rebuild              Force profile rebuild

${c.bold}Hooks:${c.reset}
  clawmem hook <name>                  Run hook (stdin JSON)

${c.bold}Integration:${c.reset}
  clawmem mcp                          Start stdio MCP server
  clawmem update-context               Regenerate all directory CLAUDE.md files
  clawmem doctor                       Full health check

${c.bold}Options:${c.reset}
  -n, --num <N>        Number of results
  -c, --collection     Filter by collection
  --json               JSON output
  --min-score <N>      Minimum score threshold
  -f, --force          Force re-embed/reindex all
  --pull               Run update commands before indexing
`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
