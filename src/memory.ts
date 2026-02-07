/**
 * ClawMem Memory Module - SAME composite scoring layer
 *
 * Provides recency decay, confidence scoring, and composite scoring
 * that overlays on top of QMD's raw search results.
 */

// =============================================================================
// Content Type Half-Lives (days until score drops to 50%)
// =============================================================================

export const HALF_LIVES: Record<string, number> = {
  handoff: 30,
  progress: 45,
  note: 60,
  research: 90,
  project: 120,
  decision: Infinity,
  hub: Infinity,
};

// =============================================================================
// Confidence Baselines by Content Type
// =============================================================================

export const TYPE_BASELINES: Record<string, number> = {
  decision: 0.85,
  hub: 0.80,
  research: 0.70,
  project: 0.65,
  handoff: 0.60,
  progress: 0.50,
  note: 0.50,
};

// =============================================================================
// Content Type Inference
// =============================================================================

export type ContentType = "decision" | "hub" | "research" | "project" | "handoff" | "progress" | "note";

export function inferContentType(path: string, explicitType?: string): ContentType {
  if (explicitType && explicitType in TYPE_BASELINES) return explicitType as ContentType;

  const lower = path.toLowerCase();
  if (lower.includes("decision") || lower.includes("adr/") || lower.includes("adr-")) return "decision";
  if (lower.includes("hub") || lower.includes("moc") || lower.match(/\/index\.md$/)) return "hub";
  if (lower.includes("research") || lower.includes("investigation") || lower.includes("analysis")) return "research";
  if (lower.includes("project") || lower.includes("epic") || lower.includes("initiative")) return "project";
  if (lower.includes("handoff") || lower.includes("handover") || lower.includes("session")) return "handoff";
  if (lower.includes("progress") || lower.includes("status") || lower.includes("standup") || lower.includes("changelog")) return "progress";
  return "note";
}

// =============================================================================
// Recency Score
// =============================================================================

export function recencyScore(modifiedAt: Date | string, contentType: string, now: Date = new Date()): number {
  const halfLife = HALF_LIVES[contentType] ?? 60;
  if (!isFinite(halfLife)) return 1.0;

  const modified = typeof modifiedAt === "string" ? new Date(modifiedAt) : modifiedAt;
  if (isNaN(modified.getTime())) return 0.5; // Invalid date → neutral score
  const daysSince = (now.getTime() - modified.getTime()) / (1000 * 60 * 60 * 24);
  if (daysSince <= 0) return 1.0;
  const result = Math.pow(0.5, daysSince / halfLife);
  return Number.isFinite(result) ? result : 0;
}

// =============================================================================
// Confidence Score
// =============================================================================

export function confidenceScore(
  contentType: string,
  modifiedAt: Date | string,
  accessCount: number,
  now: Date = new Date()
): number {
  const baseline = TYPE_BASELINES[contentType] ?? 0.5;
  const recency = recencyScore(modifiedAt, contentType, now);
  const safeAccess = Number.isFinite(accessCount) && accessCount >= 0 ? accessCount : 0;
  const accessBoost = Math.min(1.5, 1 + Math.log2(1 + safeAccess) * 0.1);
  const result = Math.min(1.0, baseline * recency * accessBoost);
  return Number.isFinite(result) ? result : 0;
}

// =============================================================================
// Composite Scoring
// =============================================================================

export type CompositeWeights = {
  search: number;
  recency: number;
  confidence: number;
};

export const DEFAULT_WEIGHTS: CompositeWeights = { search: 0.5, recency: 0.25, confidence: 0.25 };
export const RECENCY_WEIGHTS: CompositeWeights = { search: 0.1, recency: 0.7, confidence: 0.2 };

const RECENCY_PATTERNS = [
  /\brecent(ly)?\b/i,
  /\blast\s+(session|time|week|month|few\s+days)\b/i,
  /\bleft\s+off\b/i,
  /\bwhere\s+(was|were)\s+[wi]\b/i,
  /\bpick\s+up\b/i,
  /\bcontinue\b/i,
  /\byesterday\b/i,
  /\btoday\b/i,
  /\bwhat\s+(was|were)\s+(we|i)\s+(doing|working)\b/i,
];

export function hasRecencyIntent(query: string): boolean {
  return RECENCY_PATTERNS.some(p => p.test(query));
}

export function compositeScore(
  searchScore: number,
  recency: number,
  confidence: number,
  weights: CompositeWeights = DEFAULT_WEIGHTS
): number {
  // Guard against NaN propagation
  const s = Number.isFinite(searchScore) ? searchScore : 0;
  const r = Number.isFinite(recency) ? recency : 0;
  const c = Number.isFinite(confidence) ? confidence : 0;
  const result = weights.search * s + weights.recency * r + weights.confidence * c;
  return Number.isFinite(result) ? result : 0;
}

// =============================================================================
// Apply Composite Scoring to Search Results
// =============================================================================

export type EnrichedResult = {
  filepath: string;
  displayPath: string;
  title: string;
  score: number;
  body?: string;
  contentType: string;
  modifiedAt: string;
  accessCount: number;
  confidence: number;
  context: string | null;
  hash: string;
  docid: string;
  collectionName: string;
  bodyLength: number;
  source: "fts" | "vec";
  chunkPos?: number;
  fragmentType?: string;
  fragmentLabel?: string;
};

export type ScoredResult = EnrichedResult & {
  compositeScore: number;
  recencyScore: number;
};

export function applyCompositeScoring(
  results: EnrichedResult[],
  query: string
): ScoredResult[] {
  const weights = hasRecencyIntent(query) ? RECENCY_WEIGHTS : DEFAULT_WEIGHTS;
  const now = new Date();

  const scored = results.map(r => {
    const recency = recencyScore(r.modifiedAt, r.contentType, now);
    const conf = confidenceScore(r.contentType, r.modifiedAt, r.accessCount, now);
    const composite = compositeScore(r.score, recency, conf, weights);
    return { ...r, compositeScore: composite, recencyScore: recency };
  });

  // Sort by composite score descending
  scored.sort((a, b) => b.compositeScore - a.compositeScore);

  // Boost handoff/decision types when recency intent detected
  if (hasRecencyIntent(query)) {
    const priority = new Set<string>(["handoff", "decision", "progress"]);
    scored.sort((a, b) => {
      const aPriority = priority.has(a.contentType) ? 1 : 0;
      const bPriority = priority.has(b.contentType) ? 1 : 0;
      if (aPriority !== bPriority) return bPriority - aPriority;
      return b.compositeScore - a.compositeScore;
    });
  }

  return scored;
}
