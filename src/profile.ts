/**
 * User Profile Abstraction - Two-tier profile (static facts + dynamic context)
 *
 * Builds a profile document from vault contents:
 * - Static: persistent facts extracted from decisions, hubs, and notes
 * - Dynamic: recent context from last sessions and progress docs
 *
 * Stored at _clawmem/profile.md, injected at session start.
 */

import type { Store } from "./store.ts";
import { hashContent } from "./indexer.ts";
import { smartTruncate } from "./hooks.ts";
import { MAX_LEVENSHTEIN_LENGTH } from "./limits.ts";

// =============================================================================
// Types
// =============================================================================

export type Profile = {
  static: string[];
  dynamic: string[];
  updatedAt: string;
};

// =============================================================================
// Config
// =============================================================================

const STATIC_MAX_TOKENS = 500;
const DYNAMIC_MAX_TOKENS = 300;
const STATIC_MAX_FACTS = 30;
const DYNAMIC_MAX_ITEMS = 10;
const PROFILE_PATH = "profile.md";
const PROFILE_COLLECTION = "_clawmem";
const STALE_SESSION_THRESHOLD = 5;

// =============================================================================
// Profile Building
// =============================================================================

export function buildStaticProfile(store: Store): string[] {
  const facts: string[] = [];
  const seen = new Set<string>();

  // Extract from decisions
  const decisions = store.getDocumentsByType("decision", 20);
  for (const d of decisions) {
    const body = store.getDocumentBody({
      filepath: `${d.collection}/${d.path}`,
      displayPath: `${d.collection}/${d.path}`,
    } as any);
    if (!body) continue;

    const bullets = extractBullets(body);
    for (const bullet of bullets) {
      const key = bullet.toLowerCase().trim().slice(0, 60);
      if (seen.has(key)) continue;
      if (isTooSimilar(key, seen)) continue;
      seen.add(key);
      facts.push(bullet);
    }
  }

  // Extract from hub documents
  const hubs = store.getDocumentsByType("hub", 10);
  for (const h of hubs) {
    const body = store.getDocumentBody({
      filepath: `${h.collection}/${h.path}`,
      displayPath: `${h.collection}/${h.path}`,
    } as any);
    if (!body) continue;

    const bullets = extractBullets(body);
    for (const bullet of bullets) {
      const key = bullet.toLowerCase().trim().slice(0, 60);
      if (seen.has(key)) continue;
      if (isTooSimilar(key, seen)) continue;
      seen.add(key);
      facts.push(bullet);
    }
  }

  // Truncate to budget
  const maxChars = STATIC_MAX_TOKENS * 4;
  let charCount = 0;
  const result: string[] = [];
  for (const fact of facts.slice(0, STATIC_MAX_FACTS)) {
    if (charCount + fact.length > maxChars) break;
    result.push(fact);
    charCount += fact.length;
  }

  return result;
}

export function buildDynamicProfile(store: Store): string[] {
  const items: string[] = [];

  // Recent sessions
  const sessions = store.getRecentSessions(5);
  for (const s of sessions) {
    if (!s.handoffPath) continue;

    const body = store.getDocumentBody({
      filepath: s.handoffPath,
      displayPath: s.handoffPath,
    } as any);
    if (!body) continue;

    // Extract "Current State" and "Next Session Should" sections
    const currentState = extractSection(body, "Current State");
    const nextSession = extractSection(body, "Next Session Should");

    if (currentState) {
      items.push(`Current: ${smartTruncate(currentState, 150)}`);
    }
    if (nextSession) {
      items.push(`Next: ${smartTruncate(nextSession, 150)}`);
    }
  }

  // Recent progress documents
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - 7);
  const progress = store.getDocumentsByType("progress", 5);
  const recent = progress.filter(p => p.modifiedAt >= cutoff.toISOString());

  for (const p of recent) {
    items.push(`Progress: ${p.title} (${p.modifiedAt.slice(0, 10)})`);
  }

  // Truncate to budget
  const maxChars = DYNAMIC_MAX_TOKENS * 4;
  let charCount = 0;
  const result: string[] = [];
  for (const item of items.slice(0, DYNAMIC_MAX_ITEMS)) {
    if (charCount + item.length > maxChars) break;
    result.push(item);
    charCount += item.length;
  }

  return result;
}

// =============================================================================
// Profile Persistence
// =============================================================================

export function updateProfile(store: Store): void {
  const staticFacts = buildStaticProfile(store);
  const dynamicItems = buildDynamicProfile(store);
  const now = new Date().toISOString();

  const body = formatProfileDocument(staticFacts, dynamicItems);
  const hash = hashContent(body);

  // Store content
  store.insertContent(hash, body, now);

  // Upsert document (handle active, inactive, or missing)
  const existing = store.findActiveDocument(PROFILE_COLLECTION, PROFILE_PATH);
  if (existing) {
    store.updateDocument(existing.id, "User Profile", hash, now);
  } else {
    // Check for inactive row (UNIQUE(collection, path) prevents re-insert)
    const inactive = store.findAnyDocument(PROFILE_COLLECTION, PROFILE_PATH);
    if (inactive) {
      // Reactivate and update
      store.reactivateDocument(inactive.id, "User Profile", hash, now);
      store.updateDocumentMeta(inactive.id, {
        content_type: "hub",
        tags: JSON.stringify(["auto-generated", "profile"]),
      });
    } else {
      try {
        store.insertDocument(PROFILE_COLLECTION, PROFILE_PATH, "User Profile", hash, now, now);
        const doc = store.findActiveDocument(PROFILE_COLLECTION, PROFILE_PATH);
        if (doc) {
          store.updateDocumentMeta(doc.id, {
            content_type: "hub",
            tags: JSON.stringify(["auto-generated", "profile"]),
          });
        }
      } catch {
        // Collection may not exist yet
      }
    }
  }
}

export function getProfile(store: Store): Profile | null {
  const doc = store.findActiveDocument(PROFILE_COLLECTION, PROFILE_PATH);
  if (!doc) return null;

  const body = store.getDocumentBody({
    filepath: `${PROFILE_COLLECTION}/${PROFILE_PATH}`,
    displayPath: `${PROFILE_COLLECTION}/${PROFILE_PATH}`,
  } as any);
  if (!body) return null;

  return parseProfileDocument(body);
}

export function isProfileStale(store: Store): boolean {
  const doc = store.findActiveDocument(PROFILE_COLLECTION, PROFILE_PATH);
  if (!doc) return true;

  // Check how many sessions since last profile update
  const sessions = store.getRecentSessions(STALE_SESSION_THRESHOLD + 1);
  if (sessions.length === 0) return false;

  // Get the profile's modification timestamp from the document row
  const rows = store.getDocumentsByType("hub", 50);
  const profileRow = rows.find(r => r.path === PROFILE_PATH && r.collection === PROFILE_COLLECTION);
  if (!profileRow) return true;

  const profileDate = profileRow.modifiedAt;
  const sessionsSince = sessions.filter(s => s.startedAt > profileDate);
  return sessionsSince.length >= STALE_SESSION_THRESHOLD;
}

// =============================================================================
// Formatting
// =============================================================================

function formatProfileDocument(staticFacts: string[], dynamicItems: string[]): string {
  const lines = [
    "---",
    "content_type: hub",
    "tags: [auto-generated, profile]",
    "---",
    "",
    "# User Profile",
    "",
  ];

  if (staticFacts.length > 0) {
    lines.push("## Known Context", "");
    for (const fact of staticFacts) {
      lines.push(`- ${fact}`);
    }
    lines.push("");
  }

  if (dynamicItems.length > 0) {
    lines.push("## Current Focus", "");
    for (const item of dynamicItems) {
      lines.push(`- ${item}`);
    }
    lines.push("");
  }

  return lines.join("\n");
}

function parseProfileDocument(body: string): Profile {
  const staticFacts: string[] = [];
  const dynamicItems: string[] = [];
  let updatedAt = "";

  let section = "";
  for (const line of body.split("\n")) {
    if (line.startsWith("## Known Context")) {
      section = "static";
      continue;
    }
    if (line.startsWith("## Current Focus")) {
      section = "dynamic";
      continue;
    }
    if (line.startsWith("## ")) {
      section = "";
      continue;
    }

    const bullet = line.match(/^-\s+(.+)/);
    if (!bullet?.[1]) continue;

    if (section === "static") staticFacts.push(bullet[1]);
    else if (section === "dynamic") dynamicItems.push(bullet[1]);
  }

  return { static: staticFacts, dynamic: dynamicItems, updatedAt };
}

// =============================================================================
// Helpers
// =============================================================================

function extractBullets(body: string): string[] {
  const bullets: string[] = [];
  for (const line of body.split("\n")) {
    const match = line.match(/^[-*]\s+(.{10,200})/);
    if (match?.[1]) {
      bullets.push(match[1].trim());
    }
  }
  return bullets;
}

function extractSection(body: string, sectionName: string): string | null {
  const regex = new RegExp(
    `^#{1,3}\\s+${escapeRegex(sectionName)}\\b[^\\n]*\\n([\\s\\S]*?)(?=^#{1,3}\\s|$)`,
    "mi"
  );
  const match = body.match(regex);
  if (!match?.[1]) return null;
  const text = match[1].trim();
  return text.length > 10 ? text : null;
}

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function isTooSimilar(key: string, existing: Set<string>): boolean {
  for (const e of existing) {
    if (levenshteinDistance(key, e) < 5) return true;
  }
  return false;
}

function levenshteinDistance(a: string, b: string): number {
  // Bound inputs to prevent O(n²) memory blowup
  if (a.length > MAX_LEVENSHTEIN_LENGTH || b.length > MAX_LEVENSHTEIN_LENGTH) return Math.abs(a.length - b.length);
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;

  const matrix: number[][] = [];
  for (let i = 0; i <= b.length; i++) matrix[i] = [i];
  for (let j = 0; j <= a.length; j++) matrix[0]![j] = j;

  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      const cost = b[i - 1] === a[j - 1] ? 0 : 1;
      matrix[i]![j] = Math.min(
        matrix[i - 1]![j]! + 1,
        matrix[i]![j - 1]! + 1,
        matrix[i - 1]![j - 1]! + cost
      );
    }
  }

  return matrix[b.length]![a.length]!;
}
