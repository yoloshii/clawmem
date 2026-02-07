/**
 * Local Observer Agent - Structured observation extraction using local GGUF model
 *
 * Uses Qwen3-1.7B (already loaded for query expansion) with XML-formatted prompts
 * to extract structured observations and session summaries from transcripts.
 * Falls back gracefully when model is unavailable.
 */

import type { TranscriptMessage } from "./hooks.ts";
import { getDefaultLlamaCpp } from "./llm.ts";
import { MAX_LLM_GENERATE_TIMEOUT_MS } from "./limits.ts";

// =============================================================================
// Types
// =============================================================================

export type Observation = {
  type: "decision" | "bugfix" | "feature" | "refactor" | "discovery" | "change";
  title: string;
  facts: string[];
  narrative: string;
  concepts: string[];
  filesRead: string[];
  filesModified: string[];
};

export type SessionSummary = {
  request: string;
  investigated: string;
  learned: string;
  completed: string;
  nextSteps: string;
};

// =============================================================================
// Config
// =============================================================================

const MAX_TRANSCRIPT_MESSAGES = 100;
const MAX_USER_MSG_CHARS = 200;
const MAX_ASSISTANT_MSG_CHARS = 500;
const MAX_TRANSCRIPT_TOKENS = 2000;
const GENERATION_MAX_TOKENS = 2000;
const GENERATION_TEMPERATURE = 0.3;

// =============================================================================
// System Prompts
// =============================================================================

const OBSERVATION_SYSTEM_PROMPT = `You are an observer analyzing a coding session transcript. Extract structured observations.
For each significant action, decision, or discovery, output an <observation> XML element.

<observation>
  <type>one of: decision, bugfix, feature, refactor, discovery, change</type>
  <title>Brief descriptive title (max 80 chars)</title>
  <facts>
    <fact>Individual atomic fact</fact>
  </facts>
  <narrative>2-3 sentences explaining context and reasoning</narrative>
  <concepts>
    <concept>one of: how-it-works, why-it-exists, what-changed, problem-solution, gotcha, pattern, trade-off</concept>
  </concepts>
  <files_read><file>path/to/file</file></files_read>
  <files_modified><file>path/to/file</file></files_modified>
</observation>

Rules:
- Output 1-5 observations, focusing on the MOST significant events
- Each fact should be a standalone, atomic piece of information
- The narrative should explain WHY something was done, not just WHAT
- Only include files that were explicitly mentioned in the transcript
- If no significant observations, output nothing`;

const SUMMARY_SYSTEM_PROMPT = `You are a session summarizer. Analyze this coding session transcript and output a structured summary.

<summary>
  <request>What the user originally asked for (1-2 sentences)</request>
  <investigated>What was explored or researched (1-2 sentences)</investigated>
  <learned>Key insights or discoveries (1-2 sentences)</learned>
  <completed>What was actually accomplished (1-2 sentences)</completed>
  <next_steps>What should happen next (1-2 sentences)</next_steps>
</summary>

Rules:
- Be concise and specific
- Focus on outcomes, not process
- If a section has nothing relevant, write "None"`;

// =============================================================================
// Transcript Preparation
// =============================================================================

function prepareTranscript(messages: TranscriptMessage[]): string {
  const recent = messages.slice(-MAX_TRANSCRIPT_MESSAGES);
  const lines: string[] = [];
  let charCount = 0;
  const charBudget = MAX_TRANSCRIPT_TOKENS * 4; // ~4 chars per token

  for (const msg of recent) {
    if (charCount >= charBudget) break;

    const maxChars = msg.role === "user" ? MAX_USER_MSG_CHARS : MAX_ASSISTANT_MSG_CHARS;
    const content = msg.content.length > maxChars
      ? msg.content.slice(0, maxChars) + "..."
      : msg.content;

    const line = `[${msg.role}]: ${content}`;
    lines.push(line);
    charCount += line.length;
  }

  return lines.join("\n");
}

// =============================================================================
// XML Parsers
// =============================================================================

const VALID_OBSERVATION_TYPES = new Set([
  "decision", "bugfix", "feature", "refactor", "discovery", "change",
]);

const VALID_CONCEPTS = new Set([
  "how-it-works", "why-it-exists", "what-changed", "problem-solution",
  "gotcha", "pattern", "trade-off",
]);

export function parseObservationXml(xml: string): Observation | null {
  const typeMatch = xml.match(/<type>\s*(.*?)\s*<\/type>/s);
  const titleMatch = xml.match(/<title>\s*(.*?)\s*<\/title>/s);
  const narrativeMatch = xml.match(/<narrative>\s*(.*?)\s*<\/narrative>/s);

  if (!typeMatch?.[1] || !titleMatch?.[1]) return null;

  const type = typeMatch[1].trim().toLowerCase();
  if (!VALID_OBSERVATION_TYPES.has(type)) return null;

  const facts = extractMultiple(xml, "fact");
  const concepts = extractMultiple(xml, "concept")
    .filter(c => VALID_CONCEPTS.has(c.toLowerCase()))
    .map(c => c.toLowerCase());
  const filesRead = extractMultiple(xml, "file", "files_read");
  const filesModified = extractMultiple(xml, "file", "files_modified");

  return {
    type: type as Observation["type"],
    title: titleMatch[1].trim().slice(0, 80),
    facts: facts.filter(f => f.length >= 5),
    narrative: narrativeMatch?.[1]?.trim() || "",
    concepts,
    filesRead,
    filesModified,
  };
}

export function parseSummaryXml(xml: string): SessionSummary | null {
  const request = extractSingle(xml, "request");
  const investigated = extractSingle(xml, "investigated");
  const learned = extractSingle(xml, "learned");
  const completed = extractSingle(xml, "completed");
  const nextSteps = extractSingle(xml, "next_steps");

  if (!request && !completed) return null;

  return {
    request: request || "Unknown",
    investigated: investigated || "None",
    learned: learned || "None",
    completed: completed || "None",
    nextSteps: nextSteps || "None",
  };
}

function extractSingle(xml: string, tag: string): string | null {
  const match = xml.match(new RegExp(`<${tag}>\\s*(.*?)\\s*</${tag}>`, "s"));
  return match?.[1]?.trim() || null;
}

function extractMultiple(xml: string, tag: string, parentTag?: string): string[] {
  let scope = xml;
  if (parentTag) {
    const parentMatch = xml.match(new RegExp(`<${parentTag}>([\\s\\S]*?)</${parentTag}>`, "s"));
    if (!parentMatch?.[1]) return [];
    scope = parentMatch[1];
  }

  const results: string[] = [];
  const regex = new RegExp(`<${tag}>\\s*(.*?)\\s*</${tag}>`, "gs");
  let match;
  while ((match = regex.exec(scope)) !== null) {
    const text = match[1]?.trim();
    if (text) results.push(text);
  }
  return results;
}

// =============================================================================
// Core Extraction Functions
// =============================================================================

export async function extractObservations(
  messages: TranscriptMessage[]
): Promise<Observation[]> {
  if (messages.length < 4) return [];

  const transcript = prepareTranscript(messages);
  const prompt = `${OBSERVATION_SYSTEM_PROMPT}\n\n--- TRANSCRIPT ---\n${transcript}\n--- END TRANSCRIPT ---\n\nExtract observations:`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), MAX_LLM_GENERATE_TIMEOUT_MS);
  try {
    const llm = getDefaultLlamaCpp();
    const result = await llm.generate(prompt, {
      maxTokens: GENERATION_MAX_TOKENS,
      temperature: GENERATION_TEMPERATURE,
      signal: controller.signal,
    });

    if (!result?.text) return [];

    // Parse all <observation>...</observation> blocks
    const observations: Observation[] = [];
    const regex = /<observation>([\s\S]*?)<\/observation>/g;
    let match;
    while ((match = regex.exec(result.text)) !== null) {
      const obs = parseObservationXml(match[1]!);
      if (obs) observations.push(obs);
    }

    return observations;
  } catch (err) {
    console.error("Observer: observation extraction failed:", err);
    return [];
  } finally {
    clearTimeout(timer);
  }
}

export async function extractSummary(
  messages: TranscriptMessage[]
): Promise<SessionSummary | null> {
  if (messages.length < 4) return null;

  const transcript = prepareTranscript(messages);
  const prompt = `${SUMMARY_SYSTEM_PROMPT}\n\n--- TRANSCRIPT ---\n${transcript}\n--- END TRANSCRIPT ---\n\nGenerate summary:`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), MAX_LLM_GENERATE_TIMEOUT_MS);
  try {
    const llm = getDefaultLlamaCpp();
    const result = await llm.generate(prompt, {
      maxTokens: 500,
      temperature: GENERATION_TEMPERATURE,
      signal: controller.signal,
    });

    if (!result?.text) return null;

    const summaryMatch = result.text.match(/<summary>([\s\S]*?)<\/summary>/);
    if (!summaryMatch?.[1]) return null;

    return parseSummaryXml(summaryMatch[1]);
  } catch (err) {
    console.error("Observer: summary extraction failed:", err);
    return null;
  } finally {
    clearTimeout(timer);
  }
}
