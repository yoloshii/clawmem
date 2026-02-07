/**
 * Handoff Generator Hook - Stop
 *
 * Fires when a Claude Code session ends. Analyzes the transcript
 * to generate a handoff note summarizing: what was done, current state,
 * decisions made, and next steps. Stored in _clawmem collection.
 */

import type { Store } from "../store.ts";
import type { HookInput, HookOutput } from "../hooks.ts";
import {
  makeContextOutput,
  makeEmptyOutput,
  readTranscript,
  validateTranscriptPath,
  type TranscriptMessage,
} from "../hooks.ts";
import { hashContent } from "../indexer.ts";
import { extractSummary, type SessionSummary } from "../observer.ts";
import { updateDirectoryContext } from "../directory-context.ts";
import { loadConfig } from "../collections.ts";

// =============================================================================
// Config
// =============================================================================

const MIN_MESSAGES_FOR_HANDOFF = 4;

// =============================================================================
// Handler
// =============================================================================

export async function handoffGenerator(
  store: Store,
  input: HookInput
): Promise<HookOutput> {
  const transcriptPath = validateTranscriptPath(input.transcriptPath);
  if (!transcriptPath) return makeEmptyOutput("handoff-generator");

  const messages = readTranscript(transcriptPath, 200);
  if (messages.length < MIN_MESSAGES_FOR_HANDOFF) return makeEmptyOutput("handoff-generator");

  const sessionId = input.sessionId || `session-${Date.now()}`;
  const now = new Date();
  const timestamp = now.toISOString();
  const dateStr = timestamp.slice(0, 10);

  // Try observer for rich summary, fall back to regex
  const summary = await extractSummary(messages);
  const handoff = summary
    ? buildHandoffFromSummary(summary, messages, sessionId, dateStr)
    : buildHandoff(messages, sessionId, dateStr);
  const handoffHash = hashContent(handoff);

  // Store in _clawmem collection
  store.insertContent(handoffHash, handoff, timestamp);

  const handoffPath = `handoffs/${dateStr}-${sessionId.slice(0, 8)}.md`;
  try {
    store.insertDocument(
      "_clawmem",
      handoffPath,
      `Handoff ${dateStr}`,
      handoffHash,
      timestamp,
      timestamp
    );

    const doc = store.findActiveDocument("_clawmem", handoffPath);
    if (doc) {
      store.updateDocumentMeta(doc.id, {
        content_type: "handoff",
        confidence: 0.60,
      });
    }
  } catch {
    // May already exist; update
    const existing = store.findActiveDocument("_clawmem", handoffPath);
    if (existing) {
      store.db.prepare(
        "UPDATE documents SET hash = ?, modified_at = ? WHERE id = ?"
      ).run(handoffHash, timestamp, existing.id);
    }
  }

  // Update session record with handoff path
  try {
    store.updateSession(sessionId, {
      endedAt: timestamp,
      handoffPath,
      summary: extractSummaryLine(messages),
    });
  } catch {
    // Non-fatal
  }

  // Extract files changed from transcript
  const filesChanged = extractFilesChanged(messages);
  if (filesChanged.length > 0) {
    try {
      store.updateSession(sessionId, { filesChanged });
    } catch { /* non-fatal */ }

    // Trigger directory context update if enabled
    const config = loadConfig();
    if (config.directoryContext) {
      try {
        updateDirectoryContext(store, filesChanged);
      } catch { /* non-fatal */ }
    }
  }

  return makeContextOutput(
    "handoff-generator",
    `<vault-handoff>Handoff note saved: ${handoffPath}</vault-handoff>`
  );
}

// =============================================================================
// Observer-based Handoff Builder
// =============================================================================

function buildHandoffFromSummary(
  summary: SessionSummary,
  messages: TranscriptMessage[],
  sessionId: string,
  dateStr: string
): string {
  const filesChanged = extractFilesChanged(messages);

  const lines = [
    `---`,
    `content_type: handoff`,
    `tags: [auto-generated, observer]`,
    `---`,
    ``,
    `# Session Handoff — ${dateStr}`,
    ``,
    `Session: \`${sessionId.slice(0, 8)}\``,
    ``,
  ];

  if (summary.request !== "None") {
    lines.push(`## Request`, ``, summary.request, ``);
  }

  if (summary.investigated !== "None") {
    lines.push(`## What Was Investigated`, ``, summary.investigated, ``);
  }

  if (summary.learned !== "None") {
    lines.push(`## What Was Learned`, ``, summary.learned, ``);
  }

  if (summary.completed !== "None") {
    lines.push(`## What Was Done`, ``, summary.completed, ``);
  }

  if (filesChanged.length > 0) {
    lines.push(`## Files Changed`, ``);
    for (const f of filesChanged.slice(0, 20)) {
      lines.push(`- \`${f}\``);
    }
    lines.push(``);
  }

  if (summary.nextSteps !== "None") {
    lines.push(`## Next Session Should`, ``, summary.nextSteps, ``);
  }

  return lines.join("\n");
}

// =============================================================================
// Regex-based Handoff Builder (Fallback)
// =============================================================================

function buildHandoff(
  messages: TranscriptMessage[],
  sessionId: string,
  dateStr: string
): string {
  const topics = extractTopics(messages);
  const actions = extractActions(messages);
  const nextSteps = extractNextSteps(messages);
  const filesChanged = extractFilesChanged(messages);

  const lines = [
    `---`,
    `content_type: handoff`,
    `tags: [auto-generated]`,
    `---`,
    ``,
    `# Session Handoff — ${dateStr}`,
    ``,
    `Session: \`${sessionId.slice(0, 8)}\``,
    ``,
  ];

  if (topics.length > 0) {
    lines.push(`## Current State`, ``);
    for (const topic of topics) {
      lines.push(`- ${topic}`);
    }
    lines.push(``);
  }

  if (actions.length > 0) {
    lines.push(`## What Was Done`, ``);
    for (const action of actions) {
      lines.push(`- ${action}`);
    }
    lines.push(``);
  }

  if (filesChanged.length > 0) {
    lines.push(`## Files Changed`, ``);
    for (const f of filesChanged.slice(0, 20)) {
      lines.push(`- \`${f}\``);
    }
    lines.push(``);
  }

  if (nextSteps.length > 0) {
    lines.push(`## Next Session Should`, ``);
    for (const step of nextSteps) {
      lines.push(`- ${step}`);
    }
    lines.push(``);
  }

  return lines.join("\n");
}

// =============================================================================
// Content Extraction
// =============================================================================

function extractTopics(messages: TranscriptMessage[]): string[] {
  const topics: string[] = [];
  const seen = new Set<string>();

  // Get themes from user messages
  for (const msg of messages) {
    if (msg.role !== "user") continue;
    const first = msg.content.split("\n")[0]?.trim();
    if (!first || first.length < 10 || first.length > 200) continue;
    if (first.startsWith("/")) continue; // slash commands

    const key = first.slice(0, 50).toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    topics.push(first);
  }

  return topics.slice(0, 5);
}

function extractActions(messages: TranscriptMessage[]): string[] {
  const actions: string[] = [];
  const seen = new Set<string>();

  const actionPatterns = [
    /\b(?:created|wrote|added|implemented|built|set up|configured|installed|fixed|updated|modified|refactored|deleted|removed)\b/i,
  ];

  for (const msg of messages) {
    if (msg.role !== "assistant") continue;

    const sentences = msg.content.split(/(?<=[.!?])\s+/);
    for (const sentence of sentences) {
      if (sentence.length < 15 || sentence.length > 300) continue;
      if (!actionPatterns.some(p => p.test(sentence))) continue;

      const key = sentence.slice(0, 60).toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      actions.push(sentence.trim());
    }
  }

  return actions.slice(0, 10);
}

function extractNextSteps(messages: TranscriptMessage[]): string[] {
  const nextSteps: string[] = [];
  const seen = new Set<string>();

  const nextPatterns = [
    /\bnext\s+(?:step|task|we\s+(?:need|should|can)|up|thing)\b/i,
    /\btodo\b/i,
    /\bremaining\b/i,
    /\blater\b.*\b(?:we|you)\s+(?:can|should|need)\b/i,
    /\bstill\s+need\s+to\b/i,
    /\bnot\s+yet\s+(?:done|implemented|completed)\b/i,
  ];

  // Scan last 30 messages (most relevant for next steps)
  const tail = messages.slice(-30);
  for (const msg of tail) {
    if (msg.role !== "assistant") continue;

    const sentences = msg.content.split(/(?<=[.!?])\s+/);
    for (const sentence of sentences) {
      if (sentence.length < 15 || sentence.length > 300) continue;
      if (!nextPatterns.some(p => p.test(sentence))) continue;

      const key = sentence.slice(0, 60).toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      nextSteps.push(sentence.trim());
    }
  }

  return nextSteps.slice(0, 5);
}

const MAX_FILES_EXTRACTED = 200;

function extractFilesChanged(messages: TranscriptMessage[]): string[] {
  const files = new Set<string>();

  const filePatterns = [
    /(?:created|wrote|edited|modified|updated|deleted)\s+(?:file\s+)?[`"]?([^\s`"]+\.\w{1,10})[`"]?/gi,
    /(?:Write|Edit|Read)\s+tool.*?[`"]([^\s`"]+\.\w{1,10})[`"]?/gi,
    /^\s*(?:[-+]){3}\s+(a|b)\/(.+\.\w{1,10})/gm,
  ];

  for (const msg of messages) {
    if (msg.role !== "assistant") continue;
    if (files.size >= MAX_FILES_EXTRACTED) break;
    for (const pattern of filePatterns) {
      pattern.lastIndex = 0;
      let match;
      while ((match = pattern.exec(msg.content)) !== null) {
        if (files.size >= MAX_FILES_EXTRACTED) break;
        const file = match[2] || match[1];
        if (file && !file.includes("*") && file.length < 200) {
          files.add(file);
        }
      }
    }
  }

  return [...files];
}

function extractSummaryLine(messages: TranscriptMessage[]): string {
  // Get first user message as summary theme
  const firstUser = messages.find(m => m.role === "user");
  if (!firstUser) return "Unknown session";

  const first = firstUser.content.split("\n")[0]?.trim() || "";
  return first.length > 100 ? first.slice(0, 100) + "..." : first;
}
