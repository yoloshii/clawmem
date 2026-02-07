/**
 * ClawMem Prompt Injection Guard
 *
 * Multi-layer detection system ported from SAME's go-promptguard integration.
 * Checks vault content for prompt injection attempts before context injection.
 * Pure pattern-based (no LLM) for sub-ms latency.
 */

// =============================================================================
// Types
// =============================================================================

export interface DetectionResult {
  safe: boolean;
  detector: string | null;
  score: number; // 0.0 = safe, 1.0 = definite injection
}

// =============================================================================
// Detection Layers
// =============================================================================

/**
 * Layer 1: Legacy string patterns from SAME (13 patterns).
 * Case-insensitive substring match. Score: 1.0 on match.
 */
const LEGACY_PATTERNS = [
  "ignore previous",
  "ignore all previous",
  "ignore above",
  "disregard previous",
  "disregard all previous",
  "you are now",
  "new instructions",
  "system prompt",
  "<system>",
  "</system>",
  "IMPORTANT:",
  "CRITICAL:",
  "override",
];

/**
 * Layer 2: Role injection patterns. Score: 0.9 on match.
 */
const ROLE_INJECTION_PATTERNS = [
  /you are (?:now |a |an |the )/i,
  /act as (?:a |an |the |if )/i,
  /pretend (?:you(?:'re| are) |to be )/i,
  /(?:switch|change) (?:to |into )(?:a |an |the )?(?:new |different )?(?:role|mode|persona)/i,
  /your (?:new |real |true )(?:role|purpose|function|task)/i,
];

/**
 * Layer 3: Instruction override patterns. Score: 0.85 on match.
 */
const INSTRUCTION_OVERRIDE_PATTERNS = [
  /(?:ignore|forget|discard|disregard) (?:all |any )?(?:previous|prior|above|earlier)/i,
  /(?:new|updated|revised|real) (?:instructions?|directives?|rules?|guidelines?)/i,
  /(?:do not|don't|never) (?:follow|obey|listen to|adhere to)/i,
  /(?:bypass|circumvent|override|skip) (?:the |any |all )?(?:rules?|restrictions?|guidelines?|filters?|safety)/i,
];

/**
 * Layer 4: Delimiter injection patterns. Score: 0.8 on match.
 */
const DELIMITER_PATTERNS = [
  /<\/?(?:system|user|assistant|human|ai|bot|prompt|instruction)>/i,
  /\[(?:SYSTEM|INST|\/INST|SYS)\]/i,
  /```(?:system|instructions?|prompt)\s*\n/i,
  /={3,}(?:SYSTEM|PROMPT|INSTRUCTIONS?)={3,}/i,
];

/**
 * Layer 5: Unicode obfuscation detection. Score: 0.7 on match.
 */
const ZERO_WIDTH_CHARS = /[\u200B\u200C\u200D\uFEFF\u00AD\u2060\u2061\u2062\u2063\u2064]/;

// Cyrillic characters that look like Latin
const CYRILLIC_LOOKALIKES = /[\u0400-\u04FF]/;
// Greek characters that look like Latin
const GREEK_LOOKALIKES = /[\u0370-\u03FF]/;

// =============================================================================
// Detection Functions
// =============================================================================

/**
 * Multi-layer prompt injection detection.
 * Checks layers in order, short-circuits on first match.
 * Default threshold: 0.6 (same as SAME's go-promptguard config).
 */
export function detectInjection(text: string, threshold: number = 0.6): DetectionResult {
  if (!text || text.length === 0) {
    return { safe: true, detector: null, score: 0 };
  }

  // Cap input length for performance
  const input = text.slice(0, 2000);
  const lower = input.toLowerCase();

  // Layer 1: Legacy string patterns
  for (const pattern of LEGACY_PATTERNS) {
    if (lower.includes(pattern.toLowerCase())) {
      const result = { safe: false, detector: "legacy_pattern", score: 1.0 };
      return result.score >= threshold ? result : { safe: true, detector: null, score: result.score };
    }
  }

  // Layer 2: Role injection
  for (const pattern of ROLE_INJECTION_PATTERNS) {
    if (pattern.test(input)) {
      const result = { safe: false, detector: "role_injection", score: 0.9 };
      return result.score >= threshold ? result : { safe: true, detector: null, score: result.score };
    }
  }

  // Layer 3: Instruction override
  for (const pattern of INSTRUCTION_OVERRIDE_PATTERNS) {
    if (pattern.test(input)) {
      const result = { safe: false, detector: "instruction_override", score: 0.85 };
      return result.score >= threshold ? result : { safe: true, detector: null, score: result.score };
    }
  }

  // Layer 4: Delimiter injection
  for (const pattern of DELIMITER_PATTERNS) {
    if (pattern.test(input)) {
      const result = { safe: false, detector: "delimiter_injection", score: 0.8 };
      return result.score >= threshold ? result : { safe: true, detector: null, score: result.score };
    }
  }

  // Layer 5: Unicode obfuscation
  if (ZERO_WIDTH_CHARS.test(input)) {
    const result = { safe: false, detector: "unicode_obfuscation", score: 0.7 };
    return result.score >= threshold ? result : { safe: true, detector: null, score: result.score };
  }

  // Check for mixed scripts (Latin + Cyrillic/Greek in same word — homoglyph attack)
  if (hasMixedScripts(input)) {
    const result = { safe: false, detector: "homoglyph", score: 0.7 };
    return result.score >= threshold ? result : { safe: true, detector: null, score: result.score };
  }

  // Check normalization deviation
  if (hasNormalizationDeviation(input)) {
    const result = { safe: false, detector: "normalization", score: 0.7 };
    return result.score >= threshold ? result : { safe: true, detector: null, score: result.score };
  }

  return { safe: true, detector: null, score: 0 };
}

/**
 * Sanitize a snippet for safe injection into context.
 * Returns the original text if safe, or a placeholder if injection detected.
 */
export function sanitizeSnippet(text: string, threshold: number = 0.6): string {
  const result = detectInjection(text, threshold);
  if (!result.safe) {
    return "[content filtered for security]";
  }
  return text;
}

// =============================================================================
// Helpers
// =============================================================================

/**
 * Check for mixed Latin + Cyrillic/Greek within individual words.
 * This detects homoglyph attacks where Cyrillic 'а' replaces Latin 'a'.
 */
function hasMixedScripts(text: string): boolean {
  // Only check if both scripts are present at all
  const hasLatin = /[a-zA-Z]/.test(text);
  const hasCyrillic = CYRILLIC_LOOKALIKES.test(text);
  const hasGreek = GREEK_LOOKALIKES.test(text);

  if (!hasLatin || (!hasCyrillic && !hasGreek)) return false;

  // Check individual words for mixed scripts
  const words = text.split(/\s+/);
  for (const word of words) {
    if (word.length < 3) continue;
    const wordHasLatin = /[a-zA-Z]/.test(word);
    const wordHasCyrillic = CYRILLIC_LOOKALIKES.test(word);
    const wordHasGreek = GREEK_LOOKALIKES.test(word);

    if (wordHasLatin && (wordHasCyrillic || wordHasGreek)) {
      return true;
    }
  }

  return false;
}

/**
 * Check if NFKD normalization changes the text significantly.
 * Catches confusable characters and encoding tricks.
 */
function hasNormalizationDeviation(text: string): boolean {
  const normalized = text.normalize('NFKD');
  if (normalized === text) return false;

  // Count character changes — small diacritic changes are fine,
  // significant changes suggest obfuscation
  let changes = 0;
  const minLen = Math.min(text.length, normalized.length);
  for (let i = 0; i < minLen; i++) {
    if (text[i] !== normalized[i]) changes++;
  }
  changes += Math.abs(text.length - normalized.length);

  // Flag if >5% of characters changed (threshold to avoid false positives on accented text)
  return changes / text.length > 0.05;
}
