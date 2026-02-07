/**
 * Session Bootstrap Hook - SessionStart
 *
 * Fires at the start of each Claude Code session. Surfaces:
 * 0. User profile (highest priority, ~800 tokens)
 * 1. Latest handoff note (~600 tokens)
 * 2. Recent decisions (~400 tokens)
 * 3. Stale notes reminder (~200 tokens)
 *
 * Total budget: ~2000 tokens (~8000 chars).
 */

import type { Store } from "../store.ts";
import type { HookInput, HookOutput } from "../hooks.ts";
import {
  makeContextOutput,
  makeEmptyOutput,
  smartTruncate,
  estimateTokens,
  logInjection,
} from "../hooks.ts";
import { sanitizeSnippet } from "../promptguard.ts";
import { getProfile } from "../profile.ts";
import { hostname } from "os";

// =============================================================================
// Config
// =============================================================================

const TOTAL_TOKEN_BUDGET = 2000;
const PROFILE_TOKEN_BUDGET = 800;
const HANDOFF_TOKEN_BUDGET = 600;
const DECISION_TOKEN_BUDGET = 400;
const STALE_TOKEN_BUDGET = 200;
const DECISION_LOOKBACK_DAYS = 7;
const STALE_LOOKBACK_DAYS = 30;

// =============================================================================
// Handler
// =============================================================================

export async function sessionBootstrap(
  store: Store,
  input: HookInput
): Promise<HookOutput> {
  const sessionId = input.sessionId || `session-${Date.now()}`;
  const now = new Date().toISOString();

  // Register the session
  try {
    store.insertSession(sessionId, now, hostname());
  } catch {
    // Session may already exist (duplicate hook fire)
  }

  const sections: string[] = [];
  const paths: string[] = [];
  let totalTokens = 0;

  // 0. User profile (highest priority)
  const profileSection = getProfileSection(store, PROFILE_TOKEN_BUDGET);
  if (profileSection) {
    const tokens = estimateTokens(profileSection.text);
    if (totalTokens + tokens <= TOTAL_TOKEN_BUDGET) {
      sections.push(profileSection.text);
      paths.push(...profileSection.paths);
      totalTokens += tokens;
    }
  }

  // 1. Latest handoff
  const handoffSection = getLatestHandoff(store, HANDOFF_TOKEN_BUDGET);
  if (handoffSection) {
    const tokens = estimateTokens(handoffSection.text);
    if (totalTokens + tokens <= TOTAL_TOKEN_BUDGET) {
      sections.push(handoffSection.text);
      paths.push(...handoffSection.paths);
      totalTokens += tokens;
    }
  }

  // 2. Recent decisions
  const decisionSection = getRecentDecisions(store, DECISION_TOKEN_BUDGET);
  if (decisionSection) {
    const tokens = estimateTokens(decisionSection.text);
    if (totalTokens + tokens <= TOTAL_TOKEN_BUDGET) {
      sections.push(decisionSection.text);
      paths.push(...decisionSection.paths);
      totalTokens += tokens;
    }
  }

  // 3. Stale notes reminder
  const staleSection = getStaleNotes(store, STALE_TOKEN_BUDGET);
  if (staleSection) {
    const tokens = estimateTokens(staleSection.text);
    if (totalTokens + tokens <= TOTAL_TOKEN_BUDGET) {
      sections.push(staleSection.text);
      paths.push(...staleSection.paths);
      totalTokens += tokens;
    }
  }

  if (sections.length === 0) return makeEmptyOutput("session-bootstrap");

  // Log the injection
  logInjection(store, sessionId, "session-bootstrap", paths, totalTokens);

  return makeContextOutput(
    "session-bootstrap",
    `<vault-session>\n${sections.join("\n\n---\n\n")}\n</vault-session>`
  );
}

// =============================================================================
// Section Builders
// =============================================================================

function getProfileSection(
  store: Store,
  maxTokens: number
): { text: string; paths: string[] } | null {
  const profile = getProfile(store);
  if (!profile) return null;
  if (profile.static.length === 0 && profile.dynamic.length === 0) return null;

  const maxChars = maxTokens * 4;
  const lines: string[] = ["### User Profile"];
  let charCount = 20;

  if (profile.static.length > 0) {
    lines.push("**Known Context:**");
    charCount += 20;
    for (const fact of profile.static) {
      if (charCount + fact.length + 4 > maxChars) break;
      lines.push(`- ${fact}`);
      charCount += fact.length + 4;
    }
  }

  if (profile.dynamic.length > 0) {
    lines.push("", "**Current Focus:**");
    charCount += 22;
    for (const item of profile.dynamic) {
      if (charCount + item.length + 4 > maxChars) break;
      lines.push(`- ${item}`);
      charCount += item.length + 4;
    }
  }

  return { text: lines.join("\n"), paths: ["_clawmem/profile.md"] };
}

function getLatestHandoff(
  store: Store,
  maxTokens: number
): { text: string; paths: string[] } | null {
  // Get most recent session with a handoff
  const sessions = store.getRecentSessions(5);
  const withHandoff = sessions.find(s => s.handoffPath);
  if (!withHandoff?.handoffPath) {
    // Fall back: get most recent handoff-type document
    const handoffs = store.getDocumentsByType("handoff", 1);
    if (handoffs.length === 0) return null;

    const doc = handoffs[0]!;
    const body = store.getDocumentBody({ filepath: `${doc.collection}/${doc.path}`, displayPath: `${doc.collection}/${doc.path}` } as any);
    if (!body) return null;

    const text = formatHandoff(doc.title, body, maxTokens);
    return { text, paths: [`${doc.collection}/${doc.path}`] };
  }

  // Try to read the handoff note from the DB
  const handoffPath = withHandoff.handoffPath;
  const parts = handoffPath.split("/");
  if (parts.length >= 2) {
    const collection = parts[0]!;
    const path = parts.slice(1).join("/");
    const docInfo = store.findActiveDocument(collection, path);
    if (docInfo) {
      const body = store.getDocumentBody({ filepath: handoffPath, displayPath: handoffPath } as any);
      if (body) {
        const text = formatHandoff(docInfo.title, body, maxTokens);
        return { text, paths: [handoffPath] };
      }
    }
  }

  // Fall back to session summary
  if (withHandoff.summary) {
    const text = `### Last Session\n${smartTruncate(withHandoff.summary, maxTokens * 4)}`;
    return { text, paths: [] };
  }

  return null;
}

function formatHandoff(title: string, body: string, maxTokens: number): string {
  // Prompt injection guard
  body = sanitizeSnippet(body);
  if (body === "[content filtered for security]") return "";
  const maxChars = maxTokens * 4;

  // Extract key sections in priority order
  const prioritySections = [
    "Next Session Should",
    "Next Session",
    "Next Steps",
    "Current State",
    "Request",
    "What Was Done",
    "What Was Learned",
    "What Was Investigated",
    "Accomplishments",
    "Decisions Made",
    "Files Changed",
  ];

  const extracted: string[] = [];
  let remaining = maxChars;

  for (const sectionName of prioritySections) {
    if (remaining <= 100) break;
    const section = extractSection(body, sectionName);
    if (section) {
      const truncated = smartTruncate(section, remaining);
      extracted.push(truncated);
      remaining -= truncated.length;
    }
  }

  if (extracted.length > 0) {
    return `### Last Handoff: ${title}\n${extracted.join("\n\n")}`;
  }

  // No structured sections, use raw content
  return `### Last Handoff: ${title}\n${smartTruncate(body, maxChars)}`;
}

function extractSection(body: string, sectionName: string): string | null {
  const regex = new RegExp(`^#{1,3}\\s+${escapeRegex(sectionName)}\\b[^\n]*\n([\\s\\S]*?)(?=^#{1,3}\\s|$)`, "mi");
  const match = body.match(regex);
  if (!match?.[1]) return null;
  const text = match[1].trim();
  return text.length > 10 ? `**${sectionName}:**\n${text}` : null;
}

function getRecentDecisions(
  store: Store,
  maxTokens: number
): { text: string; paths: string[] } | null {
  const decisions = store.getDocumentsByType("decision", 5);
  if (decisions.length === 0) return null;

  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - DECISION_LOOKBACK_DAYS);
  const cutoffStr = cutoff.toISOString();

  // Filter to recent decisions
  const recent = decisions.filter(d => d.modifiedAt >= cutoffStr);
  if (recent.length === 0) return null;

  const maxChars = maxTokens * 4;
  const lines: string[] = ["### Recent Decisions"];
  const paths: string[] = [];
  let charCount = 25; // header

  for (const d of recent) {
    if (charCount >= maxChars) break;
    let body = store.getDocumentBody({ filepath: `${d.collection}/${d.path}`, displayPath: `${d.collection}/${d.path}` } as any);
    if (body) body = sanitizeSnippet(body);
    if (body === "[content filtered for security]") continue;
    const snippet = body ? smartTruncate(body, 200) : d.title;
    const entry = `- **${d.title}** (${d.modifiedAt.slice(0, 10)})\n  ${snippet}`;
    const entryLen = entry.length;
    if (charCount + entryLen > maxChars && lines.length > 1) break;
    lines.push(entry);
    paths.push(`${d.collection}/${d.path}`);
    charCount += entryLen;
  }

  return lines.length > 1 ? { text: lines.join("\n"), paths } : null;
}

function getStaleNotes(
  store: Store,
  maxTokens: number
): { text: string; paths: string[] } | null {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - STALE_LOOKBACK_DAYS);
  const stale = store.getStaleDocuments(cutoff.toISOString());

  if (stale.length === 0) return null;

  const maxChars = maxTokens * 4;
  const lines: string[] = ["### Notes to Review"];
  const paths: string[] = [];
  let charCount = 25;

  for (const d of stale.slice(0, 5)) {
    const entry = `- ${d.title} (${d.collection}/${d.path}) — last modified ${d.modifiedAt.slice(0, 10)}`;
    if (charCount + entry.length > maxChars && lines.length > 1) break;
    lines.push(entry);
    paths.push(`${d.collection}/${d.path}`);
    charCount += entry.length;
  }

  return lines.length > 1 ? { text: lines.join("\n"), paths } : null;
}

// =============================================================================
// Utilities
// =============================================================================

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
