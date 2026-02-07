#!/usr/bin/env bun
/**
 * ClawMem MCP Server - Model Context Protocol server
 *
 * Exposes ClawMem search and document retrieval as MCP tools and resources.
 * Includes all QMD tools + SAME memory tools (find_similar, session_log, reindex, index_stats).
 * Documents are accessible via clawmem:// URIs.
 */

import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import {
  createStore,
  extractSnippet,
  DEFAULT_EMBED_MODEL,
  DEFAULT_QUERY_MODEL,
  DEFAULT_RERANK_MODEL,
  DEFAULT_MULTI_GET_MAX_BYTES,
  type SearchResult,
  type CausalLink,
  type EvolutionEntry,
} from "./store.ts";
import {
  applyCompositeScoring,
  hasRecencyIntent,
  type EnrichedResult,
} from "./memory.ts";
import { enrichResults, reciprocalRankFusion, toRanked, type RankedResult } from "./search-utils.ts";
import { indexCollection, type IndexStats } from "./indexer.ts";
import { listCollections } from "./collections.ts";
import { classifyIntent, type IntentType } from "./intent.ts";
import { adaptiveTraversal, mergeTraversalResults } from "./graph-traversal.ts";
import { getDefaultLlamaCpp } from "./llm.ts";
import { startConsolidationWorker, stopConsolidationWorker } from "./consolidation.ts";

// =============================================================================
// Types
// =============================================================================

type SearchResultItem = {
  docid: string;
  file: string;
  title: string;
  score: number;
  context: string | null;
  snippet: string;
  contentType?: string;
  compositeScore?: number;
};

type StatusResult = {
  totalDocuments: number;
  needsEmbedding: number;
  hasVectorIndex: boolean;
  collections: {
    name: string;
    path: string;
    pattern: string;
    documents: number;
    lastUpdated: string;
  }[];
};

// =============================================================================
// Helpers
// =============================================================================

function encodeClawmemPath(path: string): string {
  return path.split('/').map(segment => encodeURIComponent(segment)).join('/');
}

function formatSearchSummary(results: SearchResultItem[], query: string): string {
  if (results.length === 0) return `No results found for "${query}"`;
  const lines = [`Found ${results.length} result${results.length === 1 ? '' : 's'} for "${query}":\n`];
  for (const r of results) {
    const scoreStr = r.compositeScore !== undefined
      ? `${Math.round(r.compositeScore * 100)}%`
      : `${Math.round(r.score * 100)}%`;
    const typeTag = r.contentType && r.contentType !== "note" ? ` [${r.contentType}]` : "";
    lines.push(`${r.docid} ${scoreStr} ${r.file} - ${r.title}${typeTag}`);
  }
  return lines.join('\n');
}

function addLineNumbers(text: string, startLine: number = 1): string {
  const lines = text.split('\n');
  return lines.map((line, i) => `${startLine + i}: ${line}`).join('\n');
}

// =============================================================================
// MCP Server
// =============================================================================

export async function startMcpServer(): Promise<void> {
  const store = createStore();

  const server = new McpServer({
    name: "clawmem",
    version: "0.1.0",
  });

  // ---------------------------------------------------------------------------
  // Tool: __IMPORTANT (workflow instructions)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "__IMPORTANT",
    {
      title: "READ THIS FIRST: Memory search workflow",
      description: "Instructions for efficient memory search. Read this before searching.",
      inputSchema: {},
    },
    async () => ({
      content: [{ type: "text" as const, text: `## ClawMem Search Workflow
1a. General recall → query(query, compact=true) — hybrid + expansion + deep rerank (4000 char)
1b. Causal/why/when/entity → intent_search(query, enable_graph_traversal=true) — graph traversal
    Choose 1a or 1b by query type. They are parallel options, not sequential fallback.
2. Progressive disclosure → multi_get("path1,path2") for full content of top hits
3. Spot checks → search(query) or vsearch(query) — fast, narrow
4. Before irreversible actions → check memory for prior decisions on the topic.
Only escalate when injected <vault-context> is insufficient. Do not re-search what hooks already surfaced.` }]
    })
  );

  // ---------------------------------------------------------------------------
  // Resource: clawmem://{path}
  // ---------------------------------------------------------------------------

  server.registerResource(
    "document",
    new ResourceTemplate("clawmem://{+path}", { list: undefined }),
    {
      title: "ClawMem Document",
      description: "A document from your ClawMem knowledge base.",
      mimeType: "text/markdown",
    },
    async (uri, { path }) => {
      const pathStr = Array.isArray(path) ? path.join('/') : (path || '');
      const decodedPath = decodeURIComponent(pathStr);
      const parts = decodedPath.split('/');
      const collection = parts[0] || '';
      const relativePath = parts.slice(1).join('/');

      let doc = store.db.prepare(`
        SELECT d.collection, d.path, d.title, c.doc as body
        FROM documents d JOIN content c ON c.hash = d.hash
        WHERE d.collection = ? AND d.path = ? AND d.active = 1
      `).get(collection, relativePath) as { collection: string; path: string; title: string; body: string } | null;

      if (!doc) {
        doc = store.db.prepare(`
          SELECT d.collection, d.path, d.title, c.doc as body
          FROM documents d JOIN content c ON c.hash = d.hash
          WHERE d.path LIKE ? AND d.active = 1 LIMIT 1
        `).get(`%${relativePath}`) as typeof doc;
      }

      if (!doc) {
        return { contents: [{ uri: uri.href, text: `Document not found: ${decodedPath}` }] };
      }

      const virtualPath = `clawmem://${doc.collection}/${doc.path}`;
      const context = store.getContextForFile(virtualPath);
      let text = addLineNumbers(doc.body);
      if (context) text = `<!-- Context: ${context} -->\n\n` + text;

      return {
        contents: [{
          uri: uri.href,
          name: `${doc.collection}/${doc.path}`,
          title: doc.title || doc.path,
          mimeType: "text/markdown",
          text,
        }],
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: search (BM25 + composite)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "search",
    {
      title: "Search (BM25 + Memory)",
      description: "Keyword search with SAME composite scoring. Results ranked by relevance + recency + confidence.",
      inputSchema: {
        query: z.string().describe("Search query"),
        limit: z.number().optional().default(10),
        minScore: z.number().optional().default(0),
        collection: z.string().optional().describe("Filter to collection"),
        compact: z.boolean().optional().default(false).describe("Return compact results (id, path, title, score, snippet) instead of full content"),
      },
    },
    async ({ query, limit, minScore, collection, compact }) => {
      const results = store.searchFTS(query, limit || 10)
        .filter(r => !collection || r.collectionName === collection);

      const enriched = enrichResults(store, results, query);
      const scored = applyCompositeScoring(enriched, query)
        .filter(r => r.compositeScore >= (minScore || 0));

      if (compact) {
        const items = scored.map(r => ({
          docid: `#${r.docid}`, path: r.displayPath, title: r.title,
          score: Math.round((r.compositeScore ?? r.score) * 100) / 100,
          snippet: (r.body || "").substring(0, 150), content_type: r.contentType, modified_at: r.modifiedAt,
          fragment: r.fragmentType ? { type: r.fragmentType, label: r.fragmentLabel } : undefined,
        }));
        return { content: [{ type: "text", text: formatSearchSummary(items.map(i => ({ ...i, file: i.path, compositeScore: i.score, context: null })), query) }], structuredContent: { results: items } };
      }

      const filtered: SearchResultItem[] = scored.map(r => {
        const { line, snippet } = extractSnippet(r.body || "", query, 300, r.chunkPos);
        return {
          docid: `#${r.docid}`,
          file: r.displayPath,
          title: r.title,
          score: r.score,
          compositeScore: Math.round(r.compositeScore * 100) / 100,
          contentType: r.contentType,
          context: store.getContextForFile(r.filepath),
          snippet: addLineNumbers(snippet, line),
        };
      });

      return {
        content: [{ type: "text", text: formatSearchSummary(filtered, query) }],
        structuredContent: { results: filtered },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: vsearch (Vector + composite)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "vsearch",
    {
      title: "Vector Search (Semantic + Memory)",
      description: "Semantic similarity search with SAME composite scoring.",
      inputSchema: {
        query: z.string().describe("Natural language query"),
        limit: z.number().optional().default(10),
        minScore: z.number().optional().default(0.3),
        collection: z.string().optional().describe("Filter to collection"),
        compact: z.boolean().optional().default(false).describe("Return compact results (id, path, title, score, snippet) instead of full content"),
      },
    },
    async ({ query, limit, minScore, collection, compact }) => {
      const tableExists = store.db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
      if (!tableExists) {
        return { content: [{ type: "text", text: "Vector index not found. Run 'clawmem embed' first." }], isError: true };
      }

      const results = await store.searchVec(query, DEFAULT_EMBED_MODEL, limit || 10);
      const filtered = results.filter(r => !collection || r.collectionName === collection);

      const enriched = enrichResults(store, filtered, query);
      const scored = applyCompositeScoring(enriched, query)
        .filter(r => r.compositeScore >= (minScore || 0.3));

      if (compact) {
        const items = scored.map(r => ({
          docid: `#${r.docid}`, path: r.displayPath, title: r.title,
          score: Math.round((r.compositeScore ?? r.score) * 100) / 100,
          snippet: (r.body || "").substring(0, 150), content_type: r.contentType, modified_at: r.modifiedAt,
          fragment: r.fragmentType ? { type: r.fragmentType, label: r.fragmentLabel } : undefined,
        }));
        return { content: [{ type: "text", text: formatSearchSummary(items.map(i => ({ ...i, file: i.path, compositeScore: i.score, context: null })), query) }], structuredContent: { results: items } };
      }

      const items: SearchResultItem[] = scored.map(r => {
        const { line, snippet } = extractSnippet(r.body || "", query, 300, r.chunkPos);
        return {
          docid: `#${r.docid}`,
          file: r.displayPath,
          title: r.title,
          score: r.score,
          compositeScore: Math.round(r.compositeScore * 100) / 100,
          contentType: r.contentType,
          context: store.getContextForFile(r.filepath),
          snippet: addLineNumbers(snippet, line),
        };
      });

      return {
        content: [{ type: "text", text: formatSearchSummary(items, query) }],
        structuredContent: { results: items },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: query (Hybrid + rerank + composite)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "query",
    {
      title: "Hybrid Query (Best Quality)",
      description: "Highest quality: BM25 + vector + query expansion + reranking + SAME composite scoring.",
      inputSchema: {
        query: z.string().describe("Natural language query"),
        limit: z.number().optional().default(10),
        minScore: z.number().optional().default(0),
        collection: z.string().optional().describe("Filter to collection"),
        compact: z.boolean().optional().default(false).describe("Return compact results (id, path, title, score, snippet) instead of full content"),
      },
    },
    async ({ query, limit, minScore, collection, compact }) => {
      const queries = await store.expandQuery(query, DEFAULT_QUERY_MODEL);
      const rankedLists: RankedResult[][] = [];
      const docidMap = new Map<string, string>();
      const hasVectors = !!store.db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();

      for (const q of queries) {
        const ftsResults = store.searchFTS(q, 20).filter(r => !collection || r.collectionName === collection);
        if (ftsResults.length > 0) {
          for (const r of ftsResults) docidMap.set(r.filepath, r.docid);
          rankedLists.push(ftsResults.map(r => ({ file: r.filepath, displayPath: r.displayPath, title: r.title, body: r.body || "", score: r.score })));
        }
        if (hasVectors) {
          const vecResults = await store.searchVec(q, DEFAULT_EMBED_MODEL, 20);
          const filteredVec = vecResults.filter(r => !collection || r.collectionName === collection);
          if (filteredVec.length > 0) {
            for (const r of filteredVec) docidMap.set(r.filepath, r.docid);
            rankedLists.push(filteredVec.map(r => ({ file: r.filepath, displayPath: r.displayPath, title: r.title, body: r.body || "", score: r.score })));
          }
        }
      }

      const weights = rankedLists.map((_, i) => i < 2 ? 2.0 : 1.0);
      const fused = reciprocalRankFusion(rankedLists, weights);
      const candidates = fused.slice(0, 30);

      const reranked = await store.rerank(
        query,
        candidates.map(c => ({ file: c.file, text: c.body.slice(0, 4000) })),
        DEFAULT_RERANK_MODEL
      );

      const candidateMap = new Map(candidates.map(c => [c.file, c]));
      const rrfRankMap = new Map(candidates.map((c, i) => [c.file, i + 1]));

      // Blend RRF + reranker scores
      const blended = reranked.map(r => {
        const rrfRank = rrfRankMap.get(r.file) || candidates.length;
        const rrfWeight = rrfRank <= 3 ? 0.75 : rrfRank <= 10 ? 0.60 : 0.40;
        const blendedScore = rrfWeight * (1 / rrfRank) + (1 - rrfWeight) * r.score;
        return { file: r.file, score: blendedScore };
      });
      blended.sort((a, b) => b.score - a.score);

      // Map to SearchResults for composite scoring
      const allSearchResults = [...store.searchFTS(query, 30)];
      const resultMap = new Map(allSearchResults.map(r => [r.filepath, r]));
      const searchResults = blended
        .map(b => {
          const r = resultMap.get(b.file) || candidateMap.get(b.file);
          if (!r) return null;
          return { ...r, score: b.score, filepath: b.file } as SearchResult;
        })
        .filter((r): r is SearchResult => r !== null);

      const enriched = enrichResults(store, searchResults, query);
      const scored = applyCompositeScoring(enriched, query)
        .filter(r => r.compositeScore >= (minScore || 0))
        .slice(0, limit || 10);

      if (compact) {
        const items = scored.map(r => ({
          docid: `#${docidMap.get(r.filepath) || r.docid}`, path: r.displayPath, title: r.title,
          score: Math.round((r.compositeScore ?? r.score) * 100) / 100,
          snippet: (r.body || "").substring(0, 150), content_type: r.contentType, modified_at: r.modifiedAt,
          fragment: r.fragmentType ? { type: r.fragmentType, label: r.fragmentLabel } : undefined,
        }));
        return { content: [{ type: "text", text: formatSearchSummary(items.map(i => ({ ...i, file: i.path, compositeScore: i.score, context: null })), query) }], structuredContent: { results: items } };
      }

      const items: SearchResultItem[] = scored.map(r => {
        const { line, snippet } = extractSnippet(r.body || "", query, 300, r.chunkPos);
        return {
          docid: `#${docidMap.get(r.filepath) || r.docid}`,
          file: r.displayPath,
          title: r.title,
          score: r.score,
          compositeScore: Math.round(r.compositeScore * 100) / 100,
          contentType: r.contentType,
          context: store.getContextForFile(r.filepath),
          snippet: addLineNumbers(snippet, line),
        };
      });

      return {
        content: [{ type: "text", text: formatSearchSummary(items, query) }],
        structuredContent: { results: items },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: memory_forget
  // ---------------------------------------------------------------------------

  server.registerTool(
    "memory_forget",
    {
      title: "Forget Memory",
      description: "Remove a memory by searching for the closest match and deactivating it.",
      inputSchema: {
        query: z.string().describe("What to forget — searches for the closest match"),
        confirm: z.boolean().optional().default(true).describe("If true, deactivates the best match. If false, just shows what would be forgotten."),
      },
    },
    async ({ query, confirm }) => {
      const results = store.searchFTS(query, 5);
      if (results.length === 0) {
        return { content: [{ type: "text", text: `No matching memory found for "${query}"` }] };
      }

      const best = results[0]!;
      const parts = best.displayPath.split("/");
      const collection = parts[0]!;
      const path = parts.slice(1).join("/");

      if (!confirm) {
        return {
          content: [{ type: "text", text: `Would forget: ${best.displayPath} — "${best.title}" (score ${Math.round(best.score * 100)}%)` }],
          structuredContent: { path: best.displayPath, title: best.title, score: best.score, action: "preview" },
        };
      }

      store.deactivateDocument(collection, path);

      // Log the deletion as audit trail
      store.insertUsage({
        sessionId: "mcp-forget",
        timestamp: new Date().toISOString(),
        hookName: "memory_forget",
        injectedPaths: [best.displayPath],
        estimatedTokens: 0,
        wasReferenced: 0,
      });

      return {
        content: [{ type: "text", text: `Forgotten: ${best.displayPath} — "${best.title}"` }],
        structuredContent: { path: best.displayPath, title: best.title, action: "deactivated" },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: profile
  // ---------------------------------------------------------------------------

  server.registerTool(
    "profile",
    {
      title: "User Profile",
      description: "Get the current user profile (static facts + dynamic context). Rebuild if stale.",
      inputSchema: {
        rebuild: z.boolean().optional().default(false).describe("Force rebuild the profile"),
      },
    },
    async ({ rebuild }) => {
      const { getProfile: gp, updateProfile: up, isProfileStale: ips } = await import("./profile.ts");

      if (rebuild || ips(store)) {
        up(store);
      }

      const profile = gp(store);
      if (!profile) {
        return { content: [{ type: "text", text: "No profile available. Try: profile(rebuild=true)" }] };
      }

      const lines: string[] = [];
      if (profile.static.length > 0) {
        lines.push("## Known Context");
        for (const f of profile.static) lines.push(`- ${f}`);
      }
      if (profile.dynamic.length > 0) {
        lines.push("", "## Current Focus");
        for (const d of profile.dynamic) lines.push(`- ${d}`);
      }

      return { content: [{ type: "text", text: lines.join("\n") || "Profile is empty." }] };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: get (Retrieve document)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "get",
    {
      title: "Get Document",
      description: "Retrieve document by file path or docid.",
      inputSchema: {
        file: z.string().describe("File path or docid (#abc123)"),
        fromLine: z.number().optional(),
        maxLines: z.number().optional(),
        lineNumbers: z.boolean().optional().default(false),
      },
    },
    async ({ file, fromLine, maxLines, lineNumbers }) => {
      let parsedFromLine = fromLine;
      let lookup = file;
      const colonMatch = lookup.match(/:(\d+)$/);
      if (colonMatch?.[1] && parsedFromLine === undefined) {
        parsedFromLine = parseInt(colonMatch[1], 10);
        lookup = lookup.slice(0, -colonMatch[0].length);
      }

      const result = store.findDocument(lookup, { includeBody: false });
      if ("error" in result) {
        let msg = `Document not found: ${file}`;
        if (result.similarFiles.length > 0) {
          msg += `\n\nDid you mean?\n${result.similarFiles.map(s => `  - ${s}`).join('\n')}`;
        }
        return { content: [{ type: "text", text: msg }], isError: true };
      }

      const body = store.getDocumentBody(result, parsedFromLine, maxLines) ?? "";
      let text = body;
      if (lineNumbers) text = addLineNumbers(text, parsedFromLine || 1);
      if (result.context) text = `<!-- Context: ${result.context} -->\n\n` + text;

      return {
        content: [{
          type: "resource",
          resource: {
            uri: `clawmem://${encodeClawmemPath(result.displayPath)}`,
            name: result.displayPath,
            title: result.title,
            mimeType: "text/markdown",
            text,
          },
        }],
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: multi_get (Retrieve multiple documents)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "multi_get",
    {
      title: "Multi-Get Documents",
      description: "Retrieve multiple documents by glob pattern or comma-separated list.",
      inputSchema: {
        pattern: z.string().describe("Glob pattern or comma-separated paths"),
        maxLines: z.number().optional(),
        maxBytes: z.number().optional().default(10240),
        lineNumbers: z.boolean().optional().default(false),
      },
    },
    async ({ pattern, maxLines, maxBytes, lineNumbers }) => {
      const { docs, errors } = store.findDocuments(pattern, { includeBody: true, maxBytes: maxBytes || DEFAULT_MULTI_GET_MAX_BYTES });
      if (docs.length === 0 && errors.length === 0) {
        return { content: [{ type: "text", text: `No files matched: ${pattern}` }], isError: true };
      }

      const content: any[] = [];
      if (errors.length > 0) content.push({ type: "text", text: `Errors:\n${errors.join('\n')}` });

      for (const result of docs) {
        if (result.skipped) {
          content.push({ type: "text", text: `[SKIPPED: ${result.doc.displayPath} - ${result.skipReason}]` });
          continue;
        }
        let text = result.doc.body || "";
        if (maxLines !== undefined) {
          const lines = text.split("\n");
          text = lines.slice(0, maxLines).join("\n");
          if (lines.length > maxLines) text += `\n\n[... truncated ${lines.length - maxLines} more lines]`;
        }
        if (lineNumbers) text = addLineNumbers(text);
        if (result.doc.context) text = `<!-- Context: ${result.doc.context} -->\n\n` + text;

        content.push({
          type: "resource",
          resource: {
            uri: `clawmem://${encodeClawmemPath(result.doc.displayPath)}`,
            name: result.doc.displayPath,
            title: result.doc.title,
            mimeType: "text/markdown",
            text,
          },
        });
      }
      return { content };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: status
  // ---------------------------------------------------------------------------

  server.registerTool(
    "status",
    {
      title: "Index Status",
      description: "Show ClawMem index status with content type distribution.",
      inputSchema: {},
    },
    async () => {
      const status: StatusResult = store.getStatus();

      // Add content type distribution
      const typeCounts = store.db.prepare(`
        SELECT content_type, COUNT(*) as count FROM documents WHERE active = 1 GROUP BY content_type ORDER BY count DESC
      `).all() as { content_type: string; count: number }[];

      const summary = [
        `ClawMem Index Status:`,
        `  Total documents: ${status.totalDocuments}`,
        `  Needs embedding: ${status.needsEmbedding}`,
        `  Vector index: ${status.hasVectorIndex ? 'yes' : 'no'}`,
        `  Collections: ${status.collections.length}`,
      ];
      for (const col of status.collections) {
        summary.push(`    - ${col.name}: ${col.path} (${col.documents} docs)`);
      }
      if (typeCounts.length > 0) {
        summary.push(`  Content types:`);
        for (const t of typeCounts) {
          summary.push(`    - ${t.content_type}: ${t.count}`);
        }
      }

      return {
        content: [{ type: "text", text: summary.join('\n') }],
        structuredContent: { ...status, contentTypes: typeCounts },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: find_similar (NEW - SAME)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "find_similar",
    {
      title: "Find Similar Notes",
      description: "Find notes similar to a reference document using vector proximity.",
      inputSchema: {
        file: z.string().describe("Path of reference document"),
        limit: z.number().optional().default(5),
      },
    },
    async ({ file, limit }) => {
      const tableExists = store.db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
      if (!tableExists) {
        return { content: [{ type: "text", text: "Vector index not found. Run 'clawmem embed' first." }], isError: true };
      }

      // Get the reference document's body
      const result = store.findDocument(file, { includeBody: false });
      if ("error" in result) {
        return { content: [{ type: "text", text: `Document not found: ${file}` }], isError: true };
      }

      const body = store.getDocumentBody(result) ?? "";
      const title = result.title || file;

      // Use the document's content as the search query
      const queryText = `${title}\n${body.slice(0, 1000)}`;
      const vecResults = await store.searchVec(queryText, DEFAULT_EMBED_MODEL, (limit || 5) + 1);

      // Filter out the reference document itself
      const similar = vecResults
        .filter(r => r.filepath !== result.filepath)
        .slice(0, limit || 5);

      const items: SearchResultItem[] = similar.map(r => {
        const { line, snippet } = extractSnippet(r.body || "", title, 200);
        return {
          docid: `#${r.docid}`,
          file: r.displayPath,
          title: r.title,
          score: Math.round(r.score * 100) / 100,
          context: store.getContextForFile(r.filepath),
          snippet: addLineNumbers(snippet, line),
        };
      });

      return {
        content: [{ type: "text", text: `${items.length} similar to "${title}":\n${items.map(i => `  ${i.file} (${Math.round(i.score * 100)}%)`).join('\n')}` }],
        structuredContent: { reference: file, results: items },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: reindex (NEW - SAME)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "reindex",
    {
      title: "Re-index Collections",
      description: "Trigger a re-scan of all collections. Detects new, changed, and deleted documents.",
      inputSchema: {},
    },
    async () => {
      const collections = listCollections();
      const totalStats: IndexStats = { added: 0, updated: 0, unchanged: 0, removed: 0 };

      for (const col of collections) {
        const stats = await indexCollection(store, col.name, col.path, col.pattern);
        totalStats.added += stats.added;
        totalStats.updated += stats.updated;
        totalStats.unchanged += stats.unchanged;
        totalStats.removed += stats.removed;
      }

      const summary = `Reindex complete: +${totalStats.added} added, ~${totalStats.updated} updated, =${totalStats.unchanged} unchanged, -${totalStats.removed} removed`;
      return {
        content: [{ type: "text" as const, text: summary }],
        structuredContent: { ...totalStats } as Record<string, unknown>,
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: index_stats (NEW - SAME)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "index_stats",
    {
      title: "Index Statistics",
      description: "Detailed index statistics with content type distribution, staleness info, and memory health.",
      inputSchema: {},
    },
    async () => {
      const status = store.getStatus();
      const typeCounts = store.db.prepare(
        `SELECT content_type, COUNT(*) as count FROM documents WHERE active = 1 GROUP BY content_type ORDER BY count DESC`
      ).all() as { content_type: string; count: number }[];

      const staleCount = store.db.prepare(
        `SELECT COUNT(*) as count FROM documents WHERE active = 1 AND review_by IS NOT NULL AND review_by <= ?`
      ).get(new Date().toISOString()) as { count: number };

      const recentSessions = store.getRecentSessions(5);
      const avgAccessCount = store.db.prepare(
        `SELECT AVG(access_count) as avg FROM documents WHERE active = 1`
      ).get() as { avg: number | null };

      const stats = {
        totalDocuments: status.totalDocuments,
        needsEmbedding: status.needsEmbedding,
        hasVectorIndex: status.hasVectorIndex,
        collections: status.collections.length,
        contentTypes: typeCounts,
        staleDocuments: staleCount.count,
        recentSessions: recentSessions.length,
        avgAccessCount: Math.round((avgAccessCount.avg ?? 0) * 100) / 100,
      };

      const summary = [
        `Index Statistics:`,
        `  Documents: ${stats.totalDocuments} (${stats.needsEmbedding} need embedding)`,
        `  Stale documents: ${stats.staleDocuments}`,
        `  Recent sessions: ${stats.recentSessions}`,
        `  Avg access count: ${stats.avgAccessCount}`,
        `  Content types:`,
        ...typeCounts.map(t => `    ${t.content_type}: ${t.count}`),
      ];

      return {
        content: [{ type: "text", text: summary.join('\n') }],
        structuredContent: stats,
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: session_log (NEW - SAME)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "session_log",
    {
      title: "Session Log",
      description: "View recent agent sessions with handoff info and file changes.",
      inputSchema: {
        limit: z.number().optional().default(10),
      },
    },
    async ({ limit }) => {
      const sessions = store.getRecentSessions(limit || 10);
      if (sessions.length === 0) {
        return { content: [{ type: "text", text: "No sessions tracked yet." }] };
      }

      const lines: string[] = [];
      for (const s of sessions) {
        const duration = s.endedAt
          ? `${Math.round((new Date(s.endedAt).getTime() - new Date(s.startedAt).getTime()) / 60000)}min`
          : "active";
        lines.push(`${s.sessionId.slice(0, 8)} ${s.startedAt} (${duration})`);
        if (s.handoffPath) lines.push(`  Handoff: ${s.handoffPath}`);
        if (s.summary) lines.push(`  ${s.summary.slice(0, 100)}`);
        if (s.filesChanged.length > 0) lines.push(`  Files: ${s.filesChanged.slice(0, 5).join(", ")}`);
      }

      return {
        content: [{ type: "text", text: lines.join('\n') }],
        structuredContent: { sessions },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: beads_sync
  // ---------------------------------------------------------------------------

  server.registerTool(
    "beads_sync",
    {
      title: "Sync Beads Issues",
      description: "Import Beads issues from .beads/beads.jsonl into ClawMem search index. Syncs task dependency graph for searchability.",
      inputSchema: {
        project_path: z.string().optional().describe("Path to project with .beads/ directory (default: cwd)"),
      },
    },
    async ({ project_path }) => {
      const cwd = project_path || process.cwd();
      const beadsPath = store.detectBeadsProject(cwd);

      if (!beadsPath) {
        return {
          content: [{ type: "text", text: "No Beads project found. Expected .beads/beads.jsonl in project directory." }],
        };
      }

      try {
        const result = await store.syncBeadsIssues(beadsPath);

        // A-MEM enrichment for newly created docs (generates semantic/entity edges)
        if (result.newDocIds.length > 0) {
          try {
            const llm = getDefaultLlamaCpp();
            for (const docId of result.newDocIds) {
              await store.postIndexEnrich(llm, docId, true);
            }
          } catch (enrichErr) {
            console.error(`[beads] A-MEM enrichment failed (non-fatal):`, enrichErr);
          }
        }

        return {
          content: [{
            type: "text",
            text: `Beads sync complete:\n  - ${result.created} new issues indexed\n  - ${result.synced} existing issues updated\n  - ${result.newDocIds.length} docs enriched with A-MEM\n  - Total: ${result.created + result.synced} issues`,
          }],
          structuredContent: { ...result, beads_path: beadsPath },
        };
      } catch (err) {
        return {
          content: [{
            type: "text",
            text: `Beads sync failed: ${err instanceof Error ? err.message : String(err)}`,
          }],
          isError: true,
        };
      }
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: build_graphs
  // ---------------------------------------------------------------------------

  server.registerTool(
    "build_graphs",
    {
      title: "Build Memory Graphs",
      description: "Build temporal and semantic graphs for MAGMA multi-graph memory. Run after indexing documents.",
      inputSchema: {
        graph_types: z.array(z.enum(['temporal', 'semantic', 'all'])).optional().default(['all']),
        semantic_threshold: z.number().optional().default(0.7).describe("Similarity threshold for semantic edges (0.0-1.0)"),
      },
    },
    async ({ graph_types, semantic_threshold }) => {
      const types = graph_types || ['all'];
      const shouldBuildTemporal = types.includes('temporal') || types.includes('all');
      const shouldBuildSemantic = types.includes('semantic') || types.includes('all');

      const results: { temporal?: number; semantic?: number } = {};

      if (shouldBuildTemporal) {
        results.temporal = store.buildTemporalBackbone();
      }

      if (shouldBuildSemantic) {
        results.semantic = await store.buildSemanticGraph(semantic_threshold);
      }

      const lines = [];
      if (results.temporal !== undefined) lines.push(`Temporal graph: ${results.temporal} edges`);
      if (results.semantic !== undefined) lines.push(`Semantic graph: ${results.semantic} edges`);

      return {
        content: [{
          type: "text",
          text: `Graph building complete:\n  ${lines.join('\n  ')}`,
        }],
        structuredContent: results,
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: intent_search
  // ---------------------------------------------------------------------------

  server.registerTool(
    "intent_search",
    {
      title: "Intent-Aware Search",
      description: "Search with automatic intent detection and graph-enhanced retrieval (MAGMA). Uses WHY/WHEN/ENTITY/WHAT classification to route queries.",
      inputSchema: {
        query: z.string().describe("Search query"),
        limit: z.number().optional().default(10),
        force_intent: z.enum(['WHY', 'WHEN', 'ENTITY', 'WHAT']).optional().describe("Override automatic intent detection"),
        enable_graph_traversal: z.boolean().optional().default(true).describe("Enable multi-hop graph expansion"),
      },
    },
    async ({ query, limit, force_intent, enable_graph_traversal }) => {
      const llm = getDefaultLlamaCpp();

      // Step 1: Intent classification
      const intent = force_intent
        ? { intent: force_intent as IntentType, confidence: 1.0 }
        : await classifyIntent(query, llm, store.db);

      // Step 2: Baseline search (BM25 + Vector)
      const bm25Results = store.searchFTS(query, 30);
      const vecResults = await store.searchVec(query, DEFAULT_EMBED_MODEL, 30);

      // Step 3: Intent-weighted RRF
      const rrfWeights = intent.intent === 'WHEN'
        ? [1.5, 1.0]  // Boost BM25 for temporal (dates in text)
        : intent.intent === 'WHY'
        ? [1.0, 1.5]  // Boost vector for causal (semantic)
        : [1.0, 1.0]; // Balanced

      const fusedRanked = reciprocalRankFusion([bm25Results.map(toRanked), vecResults.map(toRanked)], rrfWeights);

      // Map RRF results back to SearchResult with updated scores
      const allSearchResults = [...bm25Results, ...vecResults];
      const fused: SearchResult[] = fusedRanked.map(fr => {
        const original = allSearchResults.find(r => r.filepath === fr.file);
        return original ? { ...original, score: fr.score } : null;
      }).filter((r): r is SearchResult => r !== null);

      // Step 4: Graph expansion (if enabled and intent allows)
      let expanded = fused;
      if (enable_graph_traversal && (intent.intent === 'WHY' || intent.intent === 'ENTITY')) {
        const anchorEmbeddingResult = await llm.embed(query);
        if (anchorEmbeddingResult) {
          const traversed = adaptiveTraversal(store.db, fused.slice(0, 10).map(r => ({ hash: r.hash, score: r.score })), {
            maxDepth: 2,
            beamWidth: 5,
            budget: 30,
            intent: intent.intent,
            queryEmbedding: anchorEmbeddingResult.embedding,
          });

          // Merge traversed nodes with original results
          const merged = mergeTraversalResults(
            store.db,
            fused.map(r => ({ hash: r.hash, score: r.score })),
            traversed
          );

          // Convert back to SearchResult format
          expanded = merged.map(m => {
            const original = fused.find(f => f.hash === m.hash);
            return original
              ? { ...original, score: m.score }
              : { ...fused[0]!, hash: m.hash, score: m.score };
          }).filter((r): r is SearchResult => r !== null);
        }
      }

      // Step 5: Rerank top 30
      const toRerank = expanded.slice(0, 30);
      const rerankDocs = toRerank.map(r => ({
        file: r.filepath,
        text: r.body?.slice(0, 200) || r.title,
      }));

      const reranked = await store.rerank(query, rerankDocs);

      // Step 6: Composite scoring
      const enriched = enrichResults(store, toRerank.map((r, i) => ({
        ...r,
        rerankScore: reranked[i]?.score || 0,
      })), query);

      const scored = applyCompositeScoring(enriched, query);

      // Format results
      const results = scored.slice(0, limit || 10).map(r => ({
        docid: r.docid,
        file: r.filepath,
        title: r.title,
        score: r.score,
        compositeScore: r.compositeScore,
        context: r.context,
        snippet: r.body?.slice(0, 300) || '',
        contentType: r.contentType,
      }));

      return {
        content: [{
          type: "text",
          text: `Intent: ${intent.intent} (${Math.round(intent.confidence * 100)}% confidence)\n\n${formatSearchSummary(results, query)}`,
        }],
        structuredContent: {
          intent: intent.intent,
          confidence: intent.confidence,
          results,
        },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: find_causal_links (A-MEM)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "find_causal_links",
    {
      title: "Find Causal Links",
      description: "Traverse the causal graph to find documents causally related to a given document. Returns causal chains with reasoning.",
      inputSchema: {
        docid: z.string().describe("Document ID (e.g., '#123' or path)"),
        direction: z.enum(['causes', 'caused_by', 'both']).optional().default('both').describe("Direction: 'causes' (outbound), 'caused_by' (inbound), or 'both'"),
        depth: z.number().optional().default(5).describe("Maximum traversal depth (1-10)"),
      },
    },
    async ({ docid, direction, depth }) => {
      // Resolve docid to document
      const resolved = store.findDocumentByDocid(docid);
      if (!resolved) {
        return {
          content: [{ type: "text", text: `Document not found: ${docid}` }],
        };
      }

      // Get the numeric docId
      const doc = store.db.prepare(`
        SELECT id, title, collection, path
        FROM documents
        WHERE hash = ? AND active = 1
        LIMIT 1
      `).get(resolved.hash) as { id: number; title: string; collection: string; path: string } | undefined;

      if (!doc) {
        return {
          content: [{ type: "text", text: `Document not found: ${docid}` }],
        };
      }

      // Find causal links
      const links = store.findCausalLinks(doc.id, direction, depth);

      if (links.length === 0) {
        return {
          content: [{ type: "text", text: `No causal links found for "${doc.title}" (${direction})` }],
          structuredContent: { source: doc, links: [] },
        };
      }

      // Format summary
      const directionLabel = direction === 'causes' ? 'causes' : direction === 'caused_by' ? 'is caused by' : 'is causally related to';
      const lines = [`"${doc.title}" ${directionLabel} ${links.length} document(s):\n`];

      for (const link of links) {
        const confidence = Math.round(link.weight * 100);
        const reasoning = link.reasoning ? ` - ${link.reasoning}` : '';
        lines.push(`[Depth ${link.depth}] ${confidence}% ${link.title} (${link.filepath})${reasoning}`);
      }

      return {
        content: [{ type: "text", text: lines.join('\n') }],
        structuredContent: {
          source: {
            id: doc.id,
            title: doc.title,
            filepath: `${doc.collection}/${doc.path}`,
          },
          direction,
          links: links.map(l => ({
            id: l.docId,
            title: l.title,
            filepath: l.filepath,
            depth: l.depth,
            confidence: Math.round(l.weight * 100),
            reasoning: l.reasoning,
          })),
        },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: memory_evolution_status (A-MEM)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "memory_evolution_status",
    {
      title: "Memory Evolution Status",
      description: "Get the evolution timeline for a memory document, showing how its keywords and context have changed over time based on new evidence.",
      inputSchema: {
        docid: z.string().describe("Document ID (e.g., '#123' or path)"),
        limit: z.number().optional().default(10).describe("Maximum number of evolution entries to return (1-100)"),
      },
    },
    async ({ docid, limit }) => {
      // Resolve docid to document
      const resolved = store.findDocumentByDocid(docid);
      if (!resolved) {
        return {
          content: [{ type: "text", text: `Document not found: ${docid}` }],
        };
      }

      // Get the numeric docId
      const doc = store.db.prepare(`
        SELECT id, title, collection, path
        FROM documents
        WHERE hash = ? AND active = 1
        LIMIT 1
      `).get(resolved.hash) as { id: number; title: string; collection: string; path: string } | undefined;

      if (!doc) {
        return {
          content: [{ type: "text", text: `Document not found: ${docid}` }],
        };
      }

      // Get evolution timeline
      const timeline = store.getEvolutionTimeline(doc.id, limit);

      if (timeline.length === 0) {
        return {
          content: [{ type: "text", text: `No evolution history found for "${doc.title}"` }],
          structuredContent: { document: doc, timeline: [] },
        };
      }

      // Format summary
      const lines = [`Evolution timeline for "${doc.title}" (${timeline.length} version${timeline.length === 1 ? '' : 's'}):\n`];

      for (const entry of timeline) {
        lines.push(`\nVersion ${entry.version} (${entry.createdAt})`);
        lines.push(`Triggered by: ${entry.triggeredBy.title} (${entry.triggeredBy.filepath})`);

        // Keywords delta
        if (entry.previousKeywords || entry.newKeywords) {
          const prev = entry.previousKeywords?.join(', ') || 'none';
          const next = entry.newKeywords?.join(', ') || 'none';
          lines.push(`Keywords: ${prev} → ${next}`);
        }

        // Context delta
        if (entry.previousContext || entry.newContext) {
          const prevCtx = entry.previousContext || 'none';
          const newCtx = entry.newContext || 'none';
          const prevPreview = prevCtx.substring(0, 50) + (prevCtx.length > 50 ? '...' : '');
          const newPreview = newCtx.substring(0, 50) + (newCtx.length > 50 ? '...' : '');
          lines.push(`Context: ${prevPreview} → ${newPreview}`);
        }

        // Reasoning
        if (entry.reasoning) {
          lines.push(`Reasoning: ${entry.reasoning}`);
        }
      }

      return {
        content: [{ type: "text", text: lines.join('\n') }],
        structuredContent: {
          document: {
            id: doc.id,
            title: doc.title,
            filepath: `${doc.collection}/${doc.path}`,
          },
          timeline: timeline.map(e => ({
            version: e.version,
            triggeredBy: {
              id: e.triggeredBy.docId,
              title: e.triggeredBy.title,
              filepath: e.triggeredBy.filepath,
            },
            previousKeywords: e.previousKeywords,
            newKeywords: e.newKeywords,
            previousContext: e.previousContext,
            newContext: e.newContext,
            reasoning: e.reasoning,
            createdAt: e.createdAt,
          })),
        },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Connect
  // ---------------------------------------------------------------------------

  const transport = new StdioServerTransport();
  await server.connect(transport);

  // ---------------------------------------------------------------------------
  // Consolidation Worker
  // ---------------------------------------------------------------------------

  // Start consolidation worker if enabled
  if (Bun.env.CLAWMEM_ENABLE_CONSOLIDATION === "true") {
    const llm = getDefaultLlamaCpp();
    const intervalMs = parseInt(Bun.env.CLAWMEM_CONSOLIDATION_INTERVAL || "300000", 10);
    startConsolidationWorker(store, llm, intervalMs);
  }

  // Signal handlers for graceful shutdown
  process.on("SIGINT", () => {
    console.error("\n[mcp] Received SIGINT, shutting down...");
    stopConsolidationWorker();
    process.exit(0);
  });

  process.on("SIGTERM", () => {
    console.error("\n[mcp] Received SIGTERM, shutting down...");
    stopConsolidationWorker();
    process.exit(0);
  });
}

if (import.meta.main) {
  startMcpServer().catch(console.error);
}
