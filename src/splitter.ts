/**
 * ClawMem Document Splitter — Granular Fragment Indexing
 *
 * Splits markdown documents into semantic fragments (sections, bullet lists,
 * code blocks, frontmatter facts) for per-fragment embedding. Each fragment
 * gets its own vector, dramatically improving recall for specific facts
 * buried in larger documents.
 */

// =============================================================================
// Types
// =============================================================================

export interface Fragment {
  type: 'full' | 'section' | 'list' | 'code' | 'frontmatter' | 'fact' | 'narrative';
  label: string | null;
  content: string;
  startLine: number;
}

// =============================================================================
// Config
// =============================================================================

import { MAX_FRAGMENTS_PER_DOC, MAX_SPLITTER_INPUT_CHARS } from "./limits.ts";

const MIN_FRAGMENT_CHARS = 50;
const MAX_FRAGMENT_CHARS = 2000;
const MIN_DOC_CHARS_FOR_SPLIT = 200;

// =============================================================================
// Main Splitter
// =============================================================================

/**
 * Split a markdown document into semantic fragments for embedding.
 * Always includes a 'full' fragment (entire body). Additional fragments
 * are only generated if the document is large enough to benefit from splitting.
 */
export function splitDocument(
  body: string,
  frontmatter?: Record<string, any>
): Fragment[] {
  // Bound input size to prevent memory blowup
  const boundedBody = body.length > MAX_SPLITTER_INPUT_CHARS
    ? body.slice(0, MAX_SPLITTER_INPUT_CHARS)
    : body;

  const fragments: Fragment[] = [];

  // Always include full document as first fragment
  fragments.push({ type: 'full', label: null, content: boundedBody, startLine: 1 });

  // Skip splitting for very short documents
  if (boundedBody.length < MIN_DOC_CHARS_FOR_SPLIT) return fragments;

  const lines = boundedBody.split('\n');
  const remaining = () => MAX_FRAGMENTS_PER_DOC - fragments.length;

  // Extract sections (## headings)
  const sections = extractSections(lines);
  fragments.push(...sections.slice(0, remaining()));

  // Extract bullet lists
  if (remaining() > 0) {
    const lists = extractLists(lines);
    fragments.push(...lists.slice(0, remaining()));
  }

  // Extract code blocks
  if (remaining() > 0) {
    const blocks = extractCodeBlocks(lines);
    fragments.push(...blocks.slice(0, remaining()));
  }

  // Extract frontmatter facts
  if (frontmatter && remaining() > 0) {
    const fmFrags = extractFrontmatter(frontmatter);
    fragments.push(...fmFrags.slice(0, remaining()));
  }

  return fragments;
}

/**
 * Split observer-generated observations into fact and narrative fragments.
 * Used for documents that have structured `facts` and `narrative` fields.
 */
export function splitObservation(
  body: string,
  meta: { facts?: string; narrative?: string }
): Fragment[] {
  // Bound input size
  const boundedBody = body.length > MAX_SPLITTER_INPUT_CHARS
    ? body.slice(0, MAX_SPLITTER_INPUT_CHARS)
    : body;

  const fragments: Fragment[] = [];

  // Full document
  fragments.push({ type: 'full', label: null, content: boundedBody, startLine: 1 });

  // Individual facts
  if (meta.facts && fragments.length < MAX_FRAGMENTS_PER_DOC) {
    try {
      const facts = JSON.parse(meta.facts) as string[];
      for (const fact of facts) {
        if (fragments.length >= MAX_FRAGMENTS_PER_DOC) break;
        if (fact.length >= MIN_FRAGMENT_CHARS) {
          fragments.push({ type: 'fact', label: null, content: fact, startLine: 0 });
        }
      }
    } catch { /* invalid JSON, skip */ }
  }

  // Narrative
  if (meta.narrative && meta.narrative.length >= MIN_FRAGMENT_CHARS && fragments.length < MAX_FRAGMENTS_PER_DOC) {
    fragments.push({ type: 'narrative', label: null, content: meta.narrative, startLine: 0 });
  }

  return fragments;
}

// =============================================================================
// Section Extraction
// =============================================================================

function extractSections(lines: string[]): Fragment[] {
  const sections: Fragment[] = [];
  let currentHeading: string | null = null;
  let currentLines: string[] = [];
  let currentStartLine = 1;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const headingMatch = line.match(/^(#{1,3})\s+(.+)/);

    if (headingMatch) {
      // Flush previous section
      if (currentHeading !== null && currentLines.length > 0) {
        const content = currentLines.join('\n').trim();
        if (content.length >= MIN_FRAGMENT_CHARS) {
          sections.push({
            type: 'section',
            label: currentHeading,
            content: maybeSplitLarge(content),
            startLine: currentStartLine,
          });
        }
      }

      currentHeading = headingMatch[2]!.trim();
      currentLines = [line];
      currentStartLine = i + 1;
    } else {
      currentLines.push(line);
    }
  }

  // Flush last section
  if (currentHeading !== null && currentLines.length > 0) {
    const content = currentLines.join('\n').trim();
    if (content.length >= MIN_FRAGMENT_CHARS) {
      sections.push({
        type: 'section',
        label: currentHeading,
        content: maybeSplitLarge(content),
        startLine: currentStartLine,
      });
    }
  }

  return sections;
}

// =============================================================================
// List Extraction
// =============================================================================

function extractLists(lines: string[]): Fragment[] {
  const lists: Fragment[] = [];
  let currentList: string[] = [];
  let listStartLine = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const isBullet = /^\s*[-*+]\s/.test(line) || /^\s*\d+\.\s/.test(line);
    // Indented continuation of a list item
    const isContinuation = currentList.length > 0 && /^\s{2,}/.test(line) && line.trim().length > 0;

    if (isBullet || isContinuation) {
      if (currentList.length === 0) listStartLine = i + 1;
      currentList.push(line);
    } else {
      if (currentList.length >= 2) {
        const content = currentList.join('\n').trim();
        if (content.length >= MIN_FRAGMENT_CHARS) {
          lists.push({
            type: 'list',
            label: null,
            content: maybeSplitLarge(content),
            startLine: listStartLine,
          });
        }
      }
      currentList = [];
    }
  }

  // Flush trailing list
  if (currentList.length >= 2) {
    const content = currentList.join('\n').trim();
    if (content.length >= MIN_FRAGMENT_CHARS) {
      lists.push({
        type: 'list',
        label: null,
        content: maybeSplitLarge(content),
        startLine: listStartLine,
      });
    }
  }

  return lists;
}

// =============================================================================
// Code Block Extraction
// =============================================================================

function extractCodeBlocks(lines: string[]): Fragment[] {
  const blocks: Fragment[] = [];
  let inBlock = false;
  let blockLines: string[] = [];
  let blockLang: string | null = null;
  let blockStartLine = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;

    if (!inBlock && line.match(/^```(\w*)/)) {
      inBlock = true;
      blockLang = line.match(/^```(\w+)/)?.[1] || null;
      blockLines = [line];
      blockStartLine = i + 1;
    } else if (inBlock && line.startsWith('```')) {
      blockLines.push(line);
      const content = blockLines.join('\n').trim();
      if (content.length >= MIN_FRAGMENT_CHARS) {
        blocks.push({
          type: 'code',
          label: blockLang,
          content: maybeSplitLarge(content),
          startLine: blockStartLine,
        });
      }
      inBlock = false;
      blockLines = [];
      blockLang = null;
    } else if (inBlock) {
      blockLines.push(line);
    }
  }

  return blocks;
}

// =============================================================================
// Frontmatter Extraction
// =============================================================================

function extractFrontmatter(fm: Record<string, any>): Fragment[] {
  const fragments: Fragment[] = [];

  for (const [key, value] of Object.entries(fm)) {
    if (key === 'content_type' || key === 'tags') continue; // skip metadata-only fields

    let text: string;
    if (typeof value === 'string') {
      text = `${key}: ${value}`;
    } else if (typeof value === 'number' || typeof value === 'boolean') {
      text = `${key}: ${String(value)}`;
    } else if (Array.isArray(value)) {
      text = `${key}: ${value.join(', ')}`;
    } else {
      continue;
    }

    if (text.length >= 10) {
      fragments.push({
        type: 'frontmatter',
        label: key,
        content: text,
        startLine: 0,
      });
    }
  }

  return fragments;
}

// =============================================================================
// Helpers
// =============================================================================

/**
 * If content exceeds MAX_FRAGMENT_CHARS, truncate at a paragraph boundary.
 */
function maybeSplitLarge(content: string): string {
  if (content.length <= MAX_FRAGMENT_CHARS) return content;

  // Try to split at paragraph boundary
  const paragraphBreak = content.lastIndexOf('\n\n', MAX_FRAGMENT_CHARS);
  if (paragraphBreak > MAX_FRAGMENT_CHARS * 0.5) {
    return content.slice(0, paragraphBreak);
  }

  // Fall back to line boundary
  const lineBreak = content.lastIndexOf('\n', MAX_FRAGMENT_CHARS);
  if (lineBreak > MAX_FRAGMENT_CHARS * 0.5) {
    return content.slice(0, lineBreak);
  }

  // Hard truncate
  return content.slice(0, MAX_FRAGMENT_CHARS);
}
