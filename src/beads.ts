/**
 * Beads Integration - Git-backed task dependency graph
 *
 * Parses Beads JSONL format and syncs issues to ClawMem.
 * Reference: https://github.com/steveyegge/beads
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

// =============================================================================
// Types
// =============================================================================

export interface BeadsIssue {
  id: string;              // bd-a3f8
  type: string;            // task, bug, epic, feature
  title: string;
  status: string;          // open, in_progress, blocked, closed
  priority: number;        // 0-3
  tags: string[];
  assignee?: string;
  parent?: string;         // bd-a3f8 for bd-a3f8.1
  blocks: string[];        // Array of issue IDs this blocks
  notes?: string;
  created_at: string;
  closed_at?: string;
}

// =============================================================================
// Parser
// =============================================================================

/**
 * Parse Beads JSONL file into array of issues.
 */
export function parseBeadsJsonl(path: string): BeadsIssue[] {
  const content = readFileSync(path, 'utf-8');
  const lines = content.trim().split('\n');

  return lines
    .filter(line => line.trim())
    .map(line => {
      try {
        return JSON.parse(line) as BeadsIssue;
      } catch (err) {
        console.warn(`[beads] Failed to parse line: ${line}`);
        return null;
      }
    })
    .filter((issue): issue is BeadsIssue => issue !== null);
}

/**
 * Detect if a directory contains a Beads project.
 * Returns path to beads.jsonl if found, null otherwise.
 */
export function detectBeadsProject(cwd: string): string | null {
  const beadsPath = join(cwd, '.beads', 'beads.jsonl');
  return existsSync(beadsPath) ? beadsPath : null;
}

/**
 * Format a Beads issue as markdown for ClawMem indexing.
 */
export function formatBeadsIssueAsMarkdown(issue: BeadsIssue): string {
  const lines = [
    `# ${issue.title}`,
    ``,
    `**ID**: ${issue.id}`,
    `**Type**: ${issue.type}`,
    `**Status**: ${issue.status}`,
    `**Priority**: P${issue.priority}`,
  ];

  if (issue.assignee) lines.push(`**Assignee**: ${issue.assignee}`);
  if (issue.parent) lines.push(`**Parent**: ${issue.parent}`);
  if (issue.tags && issue.tags.length > 0) lines.push(`**Tags**: ${issue.tags.join(', ')}`);
  if (issue.blocks && issue.blocks.length > 0) lines.push(`**Blocks**: ${issue.blocks.join(', ')}`);

  if (issue.notes) {
    lines.push('', '## Notes', '', issue.notes);
  }

  return lines.join('\n');
}
