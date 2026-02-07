/**
 * Feedback Loop Hook - Stop
 *
 * Fires when a session ends. Detects which surfaced notes were actually
 * referenced by the assistant, and boosts their access counts.
 * This closes the learning loop: notes that prove useful rise in confidence,
 * unused notes gradually decay.
 *
 * Silent — does not inject context back to Claude.
 */

import type { Store } from "../store.ts";
import type { HookInput, HookOutput } from "../hooks.ts";
import {
  makeEmptyOutput,
  readTranscript,
  validateTranscriptPath,
} from "../hooks.ts";

// =============================================================================
// Handler
// =============================================================================

export async function feedbackLoop(
  store: Store,
  input: HookInput
): Promise<HookOutput> {
  const transcriptPath = validateTranscriptPath(input.transcriptPath);
  const sessionId = input.sessionId;
  if (!transcriptPath || !sessionId) return makeEmptyOutput("feedback-loop");

  // Get all notes injected during this session
  const usages = store.getUsageForSession(sessionId);
  if (usages.length === 0) return makeEmptyOutput("feedback-loop");

  // Collect all injected paths
  const injectedPaths = new Set<string>();
  for (const u of usages) {
    try {
      const paths = JSON.parse(u.injectedPaths) as string[];
      for (const p of paths) injectedPaths.add(p);
    } catch {
      // Skip malformed
    }
  }

  if (injectedPaths.size === 0) return makeEmptyOutput("feedback-loop");

  // Read assistant messages from transcript
  const assistantMessages = readTranscript(transcriptPath, 200, "assistant");
  if (assistantMessages.length === 0) return makeEmptyOutput("feedback-loop");

  // Build full assistant text for reference detection
  const assistantText = assistantMessages.map(m => m.content).join("\n");

  // Detect references: check if the assistant mentioned any injected path or title
  const referencedPaths: string[] = [];

  for (const path of injectedPaths) {
    // Check for path reference
    if (assistantText.includes(path)) {
      referencedPaths.push(path);
      continue;
    }

    // Check for filename reference
    const filename = path.split("/").pop()?.replace(/\.(md|txt)$/i, "");
    if (filename && filename.length > 3 && assistantText.toLowerCase().includes(filename.toLowerCase())) {
      referencedPaths.push(path);
      continue;
    }

    // Check for title reference (look up from DB)
    const titleMatch = checkTitleReference(store, path, assistantText);
    if (titleMatch) {
      referencedPaths.push(path);
    }
  }

  // Boost access counts for referenced notes
  if (referencedPaths.length > 0) {
    store.incrementAccessCount(referencedPaths);

    // Mark usage records as referenced
    for (const u of usages) {
      try {
        const paths = JSON.parse(u.injectedPaths) as string[];
        if (paths.some(p => referencedPaths.includes(p))) {
          store.markUsageReferenced(u.id);
        }
      } catch {
        // Skip
      }
    }
  }

  // Silent return — feedback loop doesn't inject context
  return makeEmptyOutput("feedback-loop");
}

// =============================================================================
// Reference Detection
// =============================================================================

function checkTitleReference(store: Store, path: string, text: string): boolean {
  try {
    const parts = path.split("/");
    if (parts.length < 2) return false;
    const collection = parts[0]!;
    const docPath = parts.slice(1).join("/");
    const doc = store.findActiveDocument(collection, docPath);
    if (!doc?.title) return false;

    // Skip generic titles
    if (doc.title.length < 5) return false;

    return text.toLowerCase().includes(doc.title.toLowerCase());
  } catch {
    return false;
  }
}
