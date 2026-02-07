/**
 * ClawMem Hook Runner - stdin/stdout JSON hook dispatch for Claude Code
 *
 * Claude Code hooks send JSON on stdin and expect JSON on stdout.
 * This module provides the I/O layer and dispatches to individual hook handlers.
 */

import type { Store } from "./store.ts";
import { createHash } from "node:crypto";

// =============================================================================
// Types
// =============================================================================

export type HookInput = {
  sessionId?: string;
  prompt?: string;
  transcriptPath?: string;
  hookEventName?: string;
};

export type HookOutput = {
  hookSpecificOutput: {
    hookEventName?: string;
    additionalContext?: string;
  };
};

// =============================================================================
// I/O
// =============================================================================

/**
 * Read hook input from stdin (Claude Code sends JSON with snake_case keys).
 * Maps snake_case → camelCase to match HookInput type.
 */
export async function readHookInput(): Promise<HookInput> {
  const chunks: Uint8Array[] = [];
  for await (const chunk of Bun.stdin.stream()) {
    chunks.push(chunk);
  }
  const raw = Buffer.concat(chunks).toString("utf-8").trim();
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return {
      sessionId: parsed.session_id ?? parsed.sessionId,
      prompt: parsed.prompt,
      transcriptPath: parsed.transcript_path ?? parsed.transcriptPath,
      hookEventName: parsed.hook_event_name ?? parsed.hookEventName,
    };
  } catch {
    return {};
  }
}

/**
 * Write hook output to stdout (Claude Code reads JSON).
 */
export function writeHookOutput(output: HookOutput): void {
  console.log(JSON.stringify(output));
}

/**
 * Create a successful output with additional context injected into Claude's prompt.
 */
export function makeContextOutput(
  hookEventName: string,
  context: string
): HookOutput {
  return {
    hookSpecificOutput: {
      hookEventName,
      additionalContext: context,
    },
  };
}

/**
 * Create an empty output (no context to inject).
 */
export function makeEmptyOutput(hookEventName?: string): HookOutput {
  return {
    hookSpecificOutput: {
      ...(hookEventName && { hookEventName }),
    },
  };
}

// =============================================================================
// Token Estimation
// =============================================================================

/**
 * Estimate token count (~4 chars per token).
 */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

// =============================================================================
// Heartbeat / Dedupe Suppression (IO4)
// =============================================================================

const DEFAULT_HEARTBEAT_SUBSTRINGS = [
  "heartbeat",
  "health check",
  "keepalive",
  "keep-alive",
  "status check",
  "are you alive",
  "still alive",
  "ping",
  "pong",
];

function getHeartbeatSubstrings(): string[] {
  const raw = (Bun.env.CLAWMEM_HEARTBEAT_PATTERNS || "").trim();
  const extra = raw
    ? raw.split(",").map(s => s.trim().toLowerCase()).filter(Boolean)
    : [];
  // Deduplicate while preserving order.
  const seen = new Set<string>();
  const out: string[] = [];
  for (const s of [...DEFAULT_HEARTBEAT_SUBSTRINGS, ...extra]) {
    if (!seen.has(s)) {
      seen.add(s);
      out.push(s);
    }
  }
  return out;
}

export function isHeartbeatPrompt(prompt: string): boolean {
  if (Bun.env.CLAWMEM_DISABLE_HEARTBEAT_SUPPRESSION === "true") return false;
  const p = (prompt || "").trim().toLowerCase();
  if (!p) return true;
  if (p.startsWith("/")) return true;

  // Exact tiny pings.
  if (p === "ping" || p === "pong" || p === "heartbeat") return true;

  const subs = getHeartbeatSubstrings();
  return subs.some(s => p.includes(s));
}

export function wasPromptSeenRecently(store: Store, hookName: string, prompt: string): boolean {
  const windowSecRaw = (Bun.env.CLAWMEM_HOOK_DEDUP_WINDOW_SEC || "").trim();
  const windowSec = windowSecRaw ? parseInt(windowSecRaw, 10) : 600;
  if (!Number.isFinite(windowSec) || windowSec <= 0) return false;

  const normalized = (prompt || "").trim();
  if (!normalized) return false;

  const hash = createHash("sha256").update(normalized).digest("hex");
  const now = new Date();
  const nowIso = now.toISOString();

  const row = store.db
    .prepare("SELECT last_seen_at FROM hook_dedupe WHERE hook_name = ? AND prompt_hash = ? LIMIT 1")
    .get(hookName, hash) as { last_seen_at: string } | null;

  let recent = false;
  if (row?.last_seen_at) {
    const lastMs = Date.parse(row.last_seen_at);
    if (!Number.isNaN(lastMs)) {
      recent = (now.getTime() - lastMs) < windowSec * 1000;
    }
  }

  const preview = normalized.slice(0, 120);
  store.db.prepare(`
    INSERT INTO hook_dedupe (hook_name, prompt_hash, prompt_preview, last_seen_at)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(hook_name, prompt_hash) DO UPDATE SET
      prompt_preview = excluded.prompt_preview,
      last_seen_at = excluded.last_seen_at
  `).run(hookName, hash, preview, nowIso);

  return recent;
}

// =============================================================================
// Transcript Parsing
// =============================================================================

export type TranscriptMessage = {
  role: "user" | "assistant" | "system";
  content: string;
};

/**
 * Read and parse a Claude Code transcript (.jsonl file).
 * Returns the last N messages.
 */
export function readTranscript(
  transcriptPath: string,
  lastN: number = 200,
  roleFilter?: "user" | "assistant"
): TranscriptMessage[] {
  try {
    const content = require("fs").readFileSync(transcriptPath, "utf-8");
    const lines = content.split("\n").filter((l: string) => l.trim());
    const messages: TranscriptMessage[] = [];

    for (const line of lines) {
      try {
        const entry = JSON.parse(line);
        // Claude Code transcript: {type, message: {role, content}} or flat {role, content}
        const msg = entry.message ?? entry;
        if (msg.role && msg.content) {
          const role = msg.role as TranscriptMessage["role"];
          const text = typeof msg.content === "string"
            ? msg.content
            : Array.isArray(msg.content)
              ? msg.content
                  .filter((b: any) => b.type === "text")
                  .map((b: any) => b.text)
                  .join("\n")
              : JSON.stringify(msg.content);

          if (!roleFilter || role === roleFilter) {
            messages.push({ role, content: text });
          }
        }
      } catch {
        // Skip malformed lines
      }
    }

    return messages.slice(-lastN);
  } catch {
    return [];
  }
}

/**
 * Validate a transcript path (security: must be absolute, .jsonl, regular file, <50MB).
 */
export function validateTranscriptPath(path: string | undefined): string | null {
  if (!path) return null;
  if (!require("path").isAbsolute(path)) return null;
  if (!path.endsWith(".jsonl")) return null;

  try {
    const stat = require("fs").statSync(path);
    if (!stat.isFile()) return null;
    if (stat.size > 50 * 1024 * 1024) return null; // 50MB limit
    return path;
  } catch {
    return null;
  }
}

// =============================================================================
// Snippet Helpers
// =============================================================================

/**
 * Smart truncate: break at paragraph → sentence → newline → word boundary.
 */
export function smartTruncate(text: string, maxChars: number = 300): string {
  if (text.length <= maxChars) return text;

  const truncated = text.slice(0, maxChars);

  // Try paragraph break
  const paraIdx = truncated.lastIndexOf("\n\n");
  if (paraIdx > maxChars * 0.5) return truncated.slice(0, paraIdx).trimEnd();

  // Try sentence break
  const sentenceMatch = truncated.match(/^(.+[.!?])\s/s);
  if (sentenceMatch && sentenceMatch[1]!.length > maxChars * 0.5) {
    return sentenceMatch[1]!;
  }

  // Try newline break
  const nlIdx = truncated.lastIndexOf("\n");
  if (nlIdx > maxChars * 0.5) return truncated.slice(0, nlIdx).trimEnd();

  // Try word boundary
  const wordIdx = truncated.lastIndexOf(" ");
  if (wordIdx > maxChars * 0.5) return truncated.slice(0, wordIdx).trimEnd() + "...";

  return truncated.trimEnd() + "...";
}

// =============================================================================
// Logging
// =============================================================================

/**
 * Log a context injection to the usage tracking table.
 */
export function logInjection(
  store: Store,
  sessionId: string,
  hookName: string,
  injectedPaths: string[],
  estimatedTokens: number
): void {
  try {
    store.insertUsage({
      sessionId,
      timestamp: new Date().toISOString(),
      hookName,
      injectedPaths,
      estimatedTokens,
      wasReferenced: 0,
    });
  } catch {
    // Non-fatal: don't crash hook if usage logging fails
  }
}
