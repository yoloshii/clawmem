/**
 * Per-Folder CLAUDE.md Generation
 *
 * Automatically maintains CLAUDE.md files in project directories with
 * relevant decisions and recent activity extracted from ClawMem.
 * Opt-in via `directoryContext: true` in ~/.config/clawmem/index.yml.
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { dirname, join, resolve } from "path";
import type { Store, DocumentRow, SessionRecord } from "./store.ts";
import { listCollections } from "./collections.ts";

// =============================================================================
// Config
// =============================================================================

const MARKER = "<!-- ClawMem Auto-Generated Context — do not edit below this line -->";
const MAX_DECISIONS_PER_DIR = 5;
const MAX_SESSIONS_PER_DIR = 3;
const DECISION_LOOKBACK_DAYS = 30;
const EXCLUDED_PATHS = ["_clawmem/", "_PRIVATE/", "node_modules/", ".git/"];

// =============================================================================
// Public API
// =============================================================================

/**
 * Update CLAUDE.md files for directories containing the given touched paths.
 * Returns the number of directories updated.
 */
export function updateDirectoryContext(
  store: Store,
  touchedPaths: string[]
): number {
  if (touchedPaths.length === 0) return 0;

  const collections = listCollections();
  if (collections.length === 0) return 0;

  // Group touched paths by directory
  const dirs = new Set<string>();
  for (const filePath of touchedPaths) {
    const dir = dirname(filePath);
    if (dir && dir !== "." && !isExcluded(dir)) {
      dirs.add(dir);
    }
  }

  if (dirs.size === 0) return 0;

  // For each directory, check if it's within a collection
  let updatedCount = 0;
  for (const dir of dirs) {
    const absDir = resolve(dir);

    // Find the collection this directory belongs to (with path boundary check)
    const col = collections.find(c => {
      const colPath = resolve(c.path);
      return absDir === colPath || absDir.startsWith(colPath + "/");
    });
    if (!col) continue;

    // Get decisions mentioning files in this directory
    const decisions = getDecisionsForDirectory(store, dir);

    // Get recent sessions touching this directory
    const sessions = getSessionsForDirectory(store, dir);

    if (decisions.length === 0 && sessions.length === 0) continue;

    // Generate the context block
    const block = generateDirectoryBlock(decisions, sessions, dir);
    if (!block) continue;

    // Write to CLAUDE.md
    const claudeMdPath = join(absDir, "CLAUDE.md");
    writeClaudeMd(claudeMdPath, block);
    updatedCount++;
  }

  return updatedCount;
}

/**
 * Regenerate CLAUDE.md for all directories that have relevant context.
 * Used by `clawmem update-context` CLI command.
 */
export function regenerateAllDirectoryContexts(store: Store): number {
  const collections = listCollections();
  if (collections.length === 0) return 0;

  // Get all directories from active documents
  const allDirs = new Set<string>();
  for (const col of collections) {
    const paths = store.getActiveDocumentPaths(col.name);
    for (const p of paths) {
      const dir = dirname(p);
      if (dir && dir !== "." && !isExcluded(dir)) {
        allDirs.add(join(col.path, dir));
      }
    }
  }

  let updatedCount = 0;
  for (const absDir of allDirs) {
    const decisions = getDecisionsForDirectory(store, absDir);
    const sessions = getSessionsForDirectory(store, absDir);

    if (decisions.length === 0 && sessions.length === 0) continue;

    const block = generateDirectoryBlock(decisions, sessions, absDir);
    if (!block) continue;

    const claudeMdPath = join(absDir, "CLAUDE.md");
    writeClaudeMd(claudeMdPath, block);
    updatedCount++;
  }

  return updatedCount;
}

// =============================================================================
// Context Generation
// =============================================================================

export function generateDirectoryBlock(
  decisions: DocumentRow[],
  sessions: SessionRecord[],
  dirPath: string
): string | null {
  const lines: string[] = [];

  if (decisions.length > 0) {
    lines.push("## Decisions");
    lines.push("");
    for (const d of decisions.slice(0, MAX_DECISIONS_PER_DIR)) {
      lines.push(`- **${d.title}** (${d.modifiedAt.slice(0, 10)})`);
    }
    lines.push("");
  }

  if (sessions.length > 0) {
    lines.push("## Recent Activity");
    lines.push("");
    for (const s of sessions.slice(0, MAX_SESSIONS_PER_DIR)) {
      const date = (s.endedAt || s.startedAt).slice(0, 10);
      const summary = s.summary || "Session activity";
      lines.push(`- ${date}: ${summary}`);
    }
    lines.push("");
  }

  return lines.length > 0 ? lines.join("\n") : null;
}

// =============================================================================
// Data Retrieval
// =============================================================================

function getDecisionsForDirectory(store: Store, dirPath: string): DocumentRow[] {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - DECISION_LOOKBACK_DAYS);
  const cutoffStr = cutoff.toISOString();

  const allDecisions = store.getDocumentsByType("decision", 50);
  const results: DocumentRow[] = [];

  for (const d of allDecisions) {
    if (d.modifiedAt < cutoffStr) continue;

    // Check if any files_modified in this decision are in the target directory
    // files_modified is stored as JSON array in the observation columns
    const body = store.getDocumentBody({
      filepath: `${d.collection}/${d.path}`,
      displayPath: `${d.collection}/${d.path}`,
    } as any);

    if (!body) continue;

    // Check if the decision body mentions any files in this directory
    const normalizedDir = dirPath.replace(/\/$/, "");
    if (body.includes(normalizedDir) || body.includes(dirname(normalizedDir))) {
      results.push(d);
      if (results.length >= MAX_DECISIONS_PER_DIR) break;
    }
  }

  return results;
}

function getSessionsForDirectory(store: Store, dirPath: string): SessionRecord[] {
  const sessions = store.getRecentSessions(10);
  const results: SessionRecord[] = [];

  const normalizedDir = dirPath.replace(/\/$/, "");

  for (const s of sessions) {
    if (s.filesChanged.length === 0) continue;

    // Check if any changed files are in this directory (with path boundary)
    const resolvedDir = resolve(normalizedDir);
    const hasMatch = s.filesChanged.some(f => {
      const resolvedFile = resolve(f);
      return resolvedFile.startsWith(resolvedDir + "/") || dirname(resolvedFile) === resolvedDir;
    });

    if (hasMatch) {
      results.push(s);
      if (results.length >= MAX_SESSIONS_PER_DIR) break;
    }
  }

  return results;
}

// =============================================================================
// File I/O
// =============================================================================

function writeClaudeMd(filePath: string, generatedBlock: string): void {
  const dir = dirname(filePath);
  if (!existsSync(dir)) return; // Don't create directories

  let existing = "";
  if (existsSync(filePath)) {
    existing = readFileSync(filePath, "utf-8");
  }

  const markerIdx = existing.indexOf(MARKER);
  const userContent = markerIdx >= 0
    ? existing.slice(0, markerIdx).trimEnd()
    : existing.trimEnd();

  const output = userContent
    ? `${userContent}\n\n${MARKER}\n\n${generatedBlock}`
    : `${MARKER}\n\n${generatedBlock}`;

  writeFileSync(filePath, output, "utf-8");
}

// =============================================================================
// Helpers
// =============================================================================

function isExcluded(path: string): boolean {
  return EXCLUDED_PATHS.some(p => path.includes(p));
}
