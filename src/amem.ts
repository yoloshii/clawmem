/**
 * A-MEM: Self-Evolving Memory System
 *
 * Constructs memory notes, generates typed links, and tracks memory evolution.
 * All operations are non-fatal and log errors with [amem] prefix.
 */

import type { Database } from "bun:sqlite";
import type { LlamaCpp } from "./llm.ts";
import type { Store } from "./store.ts";

export interface MemoryNote {
  keywords: string[];
  tags: string[];
  context: string;
}

const EMPTY_NOTE: MemoryNote = {
  keywords: [],
  tags: [],
  context: ""
};

/**
 * Extract and parse JSON from LLM output, handling:
 * - Markdown code blocks (```json ... ```)
 * - Leading/trailing prose around JSON
 * - Truncated JSON from token limits (repairs arrays/objects)
 */
function extractJsonFromLLM(raw: string): any | null {
  let text = raw.trim();

  // Strip markdown code blocks
  const codeBlock = text.match(/```(?:json)?\s*\n?([\s\S]*?)(?:\n```|$)/);
  if (codeBlock) {
    text = codeBlock[1]!.trim();
  }

  // Find the first [ or { to skip leading prose
  const arrStart = text.indexOf('[');
  const objStart = text.indexOf('{');
  if (arrStart === -1 && objStart === -1) return null;

  const start = arrStart === -1 ? objStart : objStart === -1 ? arrStart : Math.min(arrStart, objStart);
  text = text.slice(start);

  // Try parsing as-is first
  try {
    return JSON.parse(text);
  } catch {
    // Attempt truncated JSON repair
  }

  // Repair truncated arrays: find last complete object, close the array
  if (text.startsWith('[')) {
    const lastBrace = text.lastIndexOf('}');
    if (lastBrace > 0) {
      const repaired = text.slice(0, lastBrace + 1) + ']';
      try { return JSON.parse(repaired); } catch { /* continue */ }
    }
    // Might be an empty or trivial array
    try { return JSON.parse(text.replace(/,\s*$/, '') + ']'); } catch { /* continue */ }
  }

  // Repair truncated objects: find last complete value, close the object
  if (text.startsWith('{')) {
    // Try closing at each } from the end
    for (let i = text.length - 1; i > 0; i--) {
      if (text[i] === '}' || text[i] === '"' || text[i] === '0' || text[i] === '1' ||
          text[i] === '2' || text[i] === '3' || text[i] === '4' || text[i] === '5' ||
          text[i] === '6' || text[i] === '7' || text[i] === '8' || text[i] === '9' ||
          text[i] === 'e' || text[i] === 'l') {
        const candidate = text.slice(0, i + 1) + '}';
        try { return JSON.parse(candidate); } catch { /* continue */ }
      }
    }
  }

  return null;
}

/**
 * Construct a memory note for a document using LLM analysis.
 * Extracts keywords, tags, and context summary.
 *
 * @param store - Store instance
 * @param llm - LLM instance
 * @param docId - Document numeric ID
 * @returns Memory note with keywords, tags, and context
 */
export async function constructMemoryNote(
  store: Store,
  llm: LlamaCpp,
  docId: number
): Promise<MemoryNote> {
  try {
    // Get document info
    const doc = store.db.prepare(`
      SELECT d.collection, d.path, d.title, c.doc as body
      FROM documents d
      JOIN content c ON c.hash = d.hash
      WHERE d.id = ? AND d.active = 1
    `).get(docId) as { collection: string; path: string; title: string; body: string } | null;

    if (!doc) {
      console.log(`[amem] Document ${docId} not found or inactive`);
      return EMPTY_NOTE;
    }

    // Truncate content to 2000 chars
    const content = doc.body.slice(0, 2000);

    // LLM prompt for memory note construction
    const prompt = `Analyze this document and extract structured memory metadata.

Title: ${doc.title}
Path: ${doc.collection}/${doc.path}

Content:
${content}

Extract:
1. keywords: 3-7 key concepts or terms
2. tags: 2-5 categorical labels
3. context: 1-2 sentence summary of what this document is about

Return ONLY valid JSON in this exact format:
{
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "tags": ["tag1", "tag2"],
  "context": "Brief summary of the document."
}`;

    const result = await llm.generate(prompt, {
      temperature: 0.3,
      max_tokens: 300,
    });

    if (!result) {
      console.log(`[amem] LLM returned null for docId ${docId}`);
      return EMPTY_NOTE;
    }

    const parsed = extractJsonFromLLM(result.text) as MemoryNote | null;

    if (!parsed || !Array.isArray(parsed.keywords) || !Array.isArray(parsed.tags) || typeof parsed.context !== 'string') {
      console.log(`[amem] Invalid/unparseable JSON for docId ${docId}`);
      return EMPTY_NOTE;
    }

    return {
      keywords: parsed.keywords,
      tags: parsed.tags,
      context: parsed.context
    };
  } catch (err) {
    console.log(`[amem] Error constructing memory note for docId ${docId}:`, err);
    return EMPTY_NOTE;
  }
}

/**
 * Store memory note in the documents table.
 * Updates amem_keywords, amem_tags, and amem_context columns.
 *
 * @param store - Store instance
 * @param docId - Document numeric ID
 * @param note - Memory note to store
 */
export function storeMemoryNote(
  store: Store,
  docId: number,
  note: MemoryNote
): void {
  try {
    store.db.prepare(`
      UPDATE documents
      SET amem_keywords = ?,
          amem_tags = ?,
          amem_context = ?
      WHERE id = ?
    `).run(
      JSON.stringify(note.keywords),
      JSON.stringify(note.tags),
      note.context,
      docId
    );
  } catch (err) {
    console.log(`[amem] Error storing memory note for docId ${docId}:`, err);
  }
}

export interface MemoryLink {
  target_id: number;
  link_type: 'semantic' | 'supporting' | 'contradicts';
  confidence: number;
  reasoning: string;
}

/**
 * Generate typed memory links for a document based on semantic similarity.
 * Finds k-nearest neighbors and uses LLM to determine relationship types.
 *
 * @param store - Store instance
 * @param llm - LLM instance
 * @param docId - Source document numeric ID
 * @param kNeighbors - Number of neighbors to find (default 8)
 * @returns Number of links created
 */
export async function generateMemoryLinks(
  store: Store,
  llm: LlamaCpp,
  docId: number,
  kNeighbors: number = 8
): Promise<number> {
  try {
    // Get source document info
    const sourceDoc = store.db.prepare(`
      SELECT d.id, d.hash, d.title, d.collection, d.path, d.amem_context
      FROM documents d
      WHERE d.id = ? AND d.active = 1
    `).get(docId) as { id: number; hash: string; title: string; collection: string; path: string; amem_context: string | null } | null;

    if (!sourceDoc) {
      console.log(`[amem] Source document ${docId} not found or inactive`);
      return 0;
    }

    // Find k-nearest neighbors using vector similarity
    const neighbors = store.db.prepare(`
      SELECT
        d2.id as target_id,
        d2.title as target_title,
        d2.amem_context as target_context,
        vec_distance_cosine(v1.embedding, v2.embedding) as distance
      FROM vectors_vec v1, vectors_vec v2
      JOIN documents d2 ON v2.hash_seq = d2.hash || '_0'
      WHERE v1.hash_seq = ? || '_0'
        AND d2.id != ?
        AND d2.active = 1
      ORDER BY distance
      LIMIT ?
    `).all(sourceDoc.hash, sourceDoc.id, kNeighbors) as {
      target_id: number;
      target_title: string;
      target_context: string | null;
      distance: number;
    }[];

    if (neighbors.length === 0) {
      console.log(`[amem] No neighbors found for docId ${docId}`);
      return 0;
    }

    // Build LLM prompt to analyze relationships
    const neighborsText = neighbors.map((n, idx) =>
      `${idx + 1}. "${n.target_title}": ${n.target_context || 'No context available'}`
    ).join('\n');

    const prompt = `Analyze the relationship between a source document and its semantic neighbors.

Source Document:
Title: ${sourceDoc.title}
Context: ${sourceDoc.amem_context || 'No context available'}

Semantically Similar Documents:
${neighborsText}

For each neighbor, determine the relationship type:
- "semantic": General topical similarity, related concepts
- "supporting": Provides evidence, examples, or elaboration for the source
- "contradicts": Presents conflicting information or opposing views

Also assign a confidence score (0.0-1.0) for each relationship.

Return ONLY valid JSON array in this exact format:
[
  {
    "target_idx": 1,
    "link_type": "semantic",
    "confidence": 0.85,
    "reasoning": "Brief explanation"
  },
  {
    "target_idx": 2,
    "link_type": "supporting",
    "confidence": 0.92,
    "reasoning": "Brief explanation"
  }
]

Include all ${neighbors.length} neighbors in your response.`;

    const result = await llm.generate(prompt, {
      temperature: 0.3,
      max_tokens: 500,
    });

    if (!result) {
      console.log(`[amem] LLM returned null for link generation docId ${docId}`);
      return 0;
    }

    const parsed = extractJsonFromLLM(result.text) as Array<{
      target_idx: number;
      link_type: 'semantic' | 'supporting' | 'contradicts';
      confidence: number;
      reasoning: string;
    }> | null;

    if (!Array.isArray(parsed)) {
      console.log(`[amem] Invalid/unparseable JSON for link generation docId ${docId}`);
      return 0;
    }

    // Insert links into memory_relations
    let linksCreated = 0;
    const now = new Date().toISOString();

    for (const link of parsed) {
      // Validate link structure
      if (typeof link.target_idx !== 'number' ||
          link.target_idx < 1 ||
          link.target_idx > neighbors.length ||
          !['semantic', 'supporting', 'contradicts'].includes(link.link_type) ||
          typeof link.confidence !== 'number') {
        continue;
      }

      const neighbor = neighbors[link.target_idx - 1];
      if (!neighbor) continue;

      // Insert link with INSERT OR IGNORE for idempotency
      store.db.prepare(`
        INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type, weight, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
      `).run(
        sourceDoc.id,
        neighbor.target_id,
        link.link_type,
        link.confidence,
        JSON.stringify({ reasoning: link.reasoning }),
        now
      );
      linksCreated++;
    }

    console.log(`[amem] Created ${linksCreated} links for docId ${docId}`);
    return linksCreated;
  } catch (err) {
    console.log(`[amem] Error generating memory links for docId ${docId}:`, err);
    return 0;
  }
}

export interface MemoryEvolution {
  should_evolve: boolean;
  new_keywords: string[];
  new_tags: string[];
  new_context: string;
  reasoning: string;
}

/**
 * Evolve a memory note based on new evidence from linked neighbors.
 * Tracks evolution history in memory_evolution table.
 *
 * @param store - Store instance
 * @param llm - LLM instance
 * @param memoryId - Memory document numeric ID
 * @param triggeredBy - Document ID that triggered this evolution
 * @returns True if evolution occurred, false otherwise
 */
export async function evolveMemories(
  store: Store,
  llm: LlamaCpp,
  memoryId: number,
  triggeredBy: number
): Promise<boolean> {
  try {
    // Get current memory state
    const memory = store.db.prepare(`
      SELECT id, title, amem_keywords, amem_tags, amem_context
      FROM documents
      WHERE id = ? AND active = 1
    `).get(memoryId) as {
      id: number;
      title: string;
      amem_keywords: string | null;
      amem_tags: string | null;
      amem_context: string | null;
    } | null;

    if (!memory || !memory.amem_context) {
      console.log(`[amem] Memory ${memoryId} not found or has no context`);
      return false;
    }

    // Get linked neighbors for context
    const neighbors = store.db.prepare(`
      SELECT
        d.id,
        d.title,
        d.amem_context,
        mr.relation_type,
        mr.weight
      FROM memory_relations mr
      JOIN documents d ON d.id = mr.target_id
      WHERE mr.source_id = ?
        AND d.active = 1
        AND d.amem_context IS NOT NULL
      ORDER BY mr.weight DESC
      LIMIT 5
    `).all(memoryId) as Array<{
      id: number;
      title: string;
      amem_context: string;
      relation_type: string;
      weight: number;
    }>;

    if (neighbors.length === 0) {
      console.log(`[amem] No linked neighbors for memory ${memoryId}`);
      return false;
    }

    // Build LLM prompt for evolution analysis
    const currentKeywords = memory.amem_keywords ? JSON.parse(memory.amem_keywords) : [];
    const currentTags = memory.amem_tags ? JSON.parse(memory.amem_tags) : [];

    const neighborsText = neighbors.map((n, idx) =>
      `${idx + 1}. [${n.relation_type}, conf=${n.weight.toFixed(2)}] "${n.title}": ${n.amem_context}`
    ).join('\n');

    const prompt = `Analyze if a memory note should evolve based on new evidence from linked documents.

Current Memory:
Title: ${memory.title}
Keywords: ${JSON.stringify(currentKeywords)}
Tags: ${JSON.stringify(currentTags)}
Context: ${memory.amem_context}

Linked Evidence:
${neighborsText}

Determine if the memory should evolve based on:
1. New contradictory information that changes understanding
2. Supporting evidence that strengthens or refines the context
3. New concepts that should be incorporated

If evolution is warranted, provide:
- new_keywords: Updated keyword list (maintain 3-7 items)
- new_tags: Updated tags (maintain 2-5 items)
- new_context: Refined context incorporating new evidence
- reasoning: Why this evolution is necessary

Return ONLY valid JSON in this exact format:
{
  "should_evolve": true,
  "new_keywords": ["keyword1", "keyword2", "keyword3"],
  "new_tags": ["tag1", "tag2"],
  "new_context": "Updated context summary.",
  "reasoning": "Explanation of why evolution occurred."
}

If no evolution is needed:
{
  "should_evolve": false,
  "new_keywords": [],
  "new_tags": [],
  "new_context": "",
  "reasoning": "No significant new information."
}`;

    const result = await llm.generate(prompt, {
      temperature: 0.4,
      max_tokens: 400,
    });

    if (!result) {
      console.log(`[amem] LLM returned null for evolution of memory ${memoryId}`);
      return false;
    }

    const evolution = extractJsonFromLLM(result.text) as MemoryEvolution | null;

    if (!evolution || typeof evolution.should_evolve !== 'boolean') {
      console.log(`[amem] Invalid evolution JSON for memory ${memoryId}`);
      return false;
    }

    if (!evolution.should_evolve) {
      console.log(`[amem] No evolution needed for memory ${memoryId}`);
      return false;
    }

    // Validate evolution data
    if (!Array.isArray(evolution.new_keywords) ||
        !Array.isArray(evolution.new_tags) ||
        typeof evolution.new_context !== 'string' ||
        typeof evolution.reasoning !== 'string') {
      console.log(`[amem] Invalid evolution data for memory ${memoryId}`);
      return false;
    }

    // Get current version number
    const versionRow = store.db.prepare(`
      SELECT COALESCE(MAX(version), 0) as max_version
      FROM memory_evolution
      WHERE memory_id = ?
    `).get(memoryId) as { max_version: number } | null;

    const nextVersion = (versionRow?.max_version || 0) + 1;

    // Perform transactional update
    const updateStmt = store.db.prepare(`
      UPDATE documents
      SET amem_keywords = ?,
          amem_tags = ?,
          amem_context = ?
      WHERE id = ?
    `);

    const historyStmt = store.db.prepare(`
      INSERT INTO memory_evolution (
        memory_id,
        triggered_by,
        version,
        previous_keywords,
        new_keywords,
        previous_context,
        new_context,
        reasoning
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);

    try {
      store.db.exec("BEGIN TRANSACTION");

      updateStmt.run(
        JSON.stringify(evolution.new_keywords),
        JSON.stringify(evolution.new_tags),
        evolution.new_context,
        memoryId
      );

      historyStmt.run(
        memoryId,
        triggeredBy,
        nextVersion,
        memory.amem_keywords,
        JSON.stringify(evolution.new_keywords),
        memory.amem_context,
        evolution.new_context,
        evolution.reasoning
      );

      store.db.exec("COMMIT");
      console.log(`[amem] Evolved memory ${memoryId} to version ${nextVersion}`);
      return true;
    } catch (err) {
      store.db.exec("ROLLBACK");
      console.log(`[amem] Transaction failed for memory ${memoryId}:`, err);
      return false;
    }
  } catch (err) {
    console.log(`[amem] Error evolving memory ${memoryId}:`, err);
    return false;
  }
}

/**
 * Post-index enrichment orchestrator.
 * Runs A-MEM processing after document indexing.
 *
 * For new documents:
 *   - Construct memory note
 *   - Generate memory links
 *   - Evolve memories based on new evidence
 *
 * For updated documents:
 *   - Refresh memory note only (skip links/evolution to avoid churn)
 *
 * All operations are non-fatal and gated by CLAWMEM_ENABLE_AMEM feature flag.
 *
 * @param store - Store instance
 * @param llm - LLM instance
 * @param docId - Document numeric ID
 * @param isNew - True if this is a new document, false if update
 */
export async function postIndexEnrich(
  store: Store,
  llm: LlamaCpp,
  docId: number,
  isNew: boolean
): Promise<void> {
  try {
    // Check feature flag
    if (Bun.env.CLAWMEM_ENABLE_AMEM === 'false') {
      return;
    }

    console.log(`[amem] Starting enrichment for docId ${docId} (isNew=${isNew})`);

    // Step 1: Construct and store memory note (always)
    const note = await constructMemoryNote(store, llm, docId);
    storeMemoryNote(store, docId, note);

    // For updated documents, stop here to avoid churn
    if (!isNew) {
      console.log(`[amem] Completed note refresh for docId ${docId}`);
      return;
    }

    // Step 2: Generate memory links (new documents only)
    const linksCreated = await generateMemoryLinks(store, llm, docId);
    console.log(`[amem] Created ${linksCreated} links for docId ${docId}`);

    // Step 3: Evolve memories based on new evidence (new documents only)
    // The new document triggers evolution of its linked neighbors
    if (linksCreated > 0) {
      // Get neighbors this new document links to (outbound links from generateMemoryLinks)
      const neighbors = store.db.prepare(`
        SELECT DISTINCT target_id
        FROM memory_relations
        WHERE source_id = ?
      `).all(docId) as Array<{ target_id: number }>;

      for (const neighbor of neighbors) {
        await evolveMemories(store, llm, neighbor.target_id, docId);
      }
    }

    console.log(`[amem] Completed full enrichment for docId ${docId}`);
  } catch (err) {
    console.log(`[amem] Error in postIndexEnrich for docId ${docId}:`, err);
  }
}

/**
 * Observation with document ID for causal inference
 */
export interface ObservationWithDoc {
  docId: number;
  facts: string[];
}

/**
 * Causal link identified by LLM
 */
interface CausalLink {
  source_fact_idx: number;
  target_fact_idx: number;
  confidence: number;
  reasoning: string;
}

/**
 * Infer causal relationships between facts from observations.
 * Analyzes facts using LLM and creates causal edges in memory_relations.
 *
 * @param store - Store instance
 * @param llm - LLM instance
 * @param observations - Array of observations with docId and facts
 * @returns Number of causal links created
 */
export async function inferCausalLinks(
  store: Store,
  llm: LlamaCpp,
  observations: ObservationWithDoc[]
): Promise<number> {
  try {
    // Build flat list of facts with source document mapping
    const factMap: Array<{ fact: string; docId: number }> = [];
    for (const obs of observations) {
      for (const fact of obs.facts) {
        factMap.push({ fact, docId: obs.docId });
      }
    }

    // Need at least 2 facts to infer causality
    if (factMap.length < 2) {
      console.log(`[amem] Insufficient facts (${factMap.length}) for causal inference`);
      return 0;
    }

    console.log(`[amem] Inferring causal links from ${factMap.length} facts across ${observations.length} observations`);

    // Build LLM prompt
    const factsText = factMap.map((f, idx) =>
      `${idx}. ${f.fact}`
    ).join('\n');

    const prompt = `Analyze the following facts from a session and identify causal relationships.

Facts:
${factsText}

Identify cause-effect relationships where one fact directly or indirectly caused another.
Consider:
- Temporal ordering (causes precede effects)
- Logical dependencies (one fact enables or triggers another)
- Problem-solution patterns (a discovery leads to an action)

Return ONLY valid JSON array in this exact format:
[
  {
    "source_fact_idx": 0,
    "target_fact_idx": 2,
    "confidence": 0.85,
    "reasoning": "Brief explanation of causal relationship"
  },
  {
    "source_fact_idx": 1,
    "target_fact_idx": 3,
    "confidence": 0.72,
    "reasoning": "Brief explanation of causal relationship"
  }
]

Only include relationships with confidence >= 0.6. Return empty array [] if no causal relationships found.`;

    const result = await llm.generate(prompt, {
      temperature: 0.3,
      max_tokens: 600,
    });

    if (!result) {
      console.log(`[amem] LLM returned null for causal inference`);
      return 0;
    }

    const links = extractJsonFromLLM(result.text) as CausalLink[] | null;

    if (!Array.isArray(links)) {
      console.log(`[amem] Invalid JSON for causal inference (not an array)`);
      return 0;
    }

    // Filter by confidence threshold and insert causal links
    let linksCreated = 0;
    const timestamp = new Date().toISOString();
    const insertStmt = store.db.prepare(`
      INSERT OR IGNORE INTO memory_relations (
        source_id, target_id, relation_type, weight, metadata, created_at
      ) VALUES (?, ?, 'causal', ?, ?, ?)
    `);

    for (const link of links) {
      // Validate link structure
      if (typeof link.source_fact_idx !== 'number' ||
          typeof link.target_fact_idx !== 'number' ||
          typeof link.confidence !== 'number' ||
          typeof link.reasoning !== 'string') {
        console.log(`[amem] Invalid causal link structure, skipping`);
        continue;
      }

      // Filter by confidence threshold
      if (link.confidence < 0.6) {
        continue;
      }

      // Validate indices
      if (link.source_fact_idx < 0 || link.source_fact_idx >= factMap.length ||
          link.target_fact_idx < 0 || link.target_fact_idx >= factMap.length) {
        console.log(`[amem] Invalid fact indices: ${link.source_fact_idx} -> ${link.target_fact_idx}`);
        continue;
      }

      // Get document IDs
      const sourceDocId = factMap[link.source_fact_idx].docId;
      const targetDocId = factMap[link.target_fact_idx].docId;

      // Skip self-links (same document)
      if (sourceDocId === targetDocId) {
        continue;
      }

      // Insert causal relation
      const metadata = JSON.stringify({
        reasoning: link.reasoning,
        source_fact: factMap[link.source_fact_idx].fact,
        target_fact: factMap[link.target_fact_idx].fact,
      });

      insertStmt.run(sourceDocId, targetDocId, link.confidence, metadata, timestamp);
      linksCreated++;
    }

    console.log(`[amem] Created ${linksCreated} causal links from ${links.length} identified relationships`);
    return linksCreated;
  } catch (err) {
    console.log(`[amem] Error in inferCausalLinks:`, err);
    return 0;
  }
}
