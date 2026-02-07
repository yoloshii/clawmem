/**
 * ClawMem Store - Core data access and retrieval functions
 *
 * Forked from QMD store with SAME agent memory extensions:
 * - Extended documents table (domain, workstream, tags, content_type, confidence, access_count)
 * - Session tracking (session_log table)
 * - Context usage tracking (context_usage table)
 *
 * Usage:
 *   const store = createStore("/path/to/db.sqlite");
 *   // or use default path:
 *   const store = createStore();
 */

import { Database } from "bun:sqlite";
import { Glob } from "bun";
import { realpathSync } from "node:fs";
import * as sqliteVec from "sqlite-vec";
import {
  LlamaCpp,
  getDefaultLlamaCpp,
  formatQueryForEmbedding,
  formatDocForEmbedding,
  type RerankDocument,
} from "./llm.ts";
import {
  findContextForPath as collectionsFindContextForPath,
  addContext as collectionsAddContext,
  removeContext as collectionsRemoveContext,
  listAllContexts as collectionsListAllContexts,
  getCollection,
  listCollections as collectionsListCollections,
  addCollection as collectionsAddCollection,
  removeCollection as collectionsRemoveCollection,
  renameCollection as collectionsRenameCollection,
  setGlobalContext,
  loadConfig as collectionsLoadConfig,
  type NamedCollection,
} from "./collections.ts";
import {
  parseBeadsJsonl,
  formatBeadsIssueAsMarkdown,
  detectBeadsProject,
  type BeadsIssue,
} from "./beads.ts";
import {
  constructMemoryNote,
  storeMemoryNote,
  generateMemoryLinks,
  evolveMemories,
  postIndexEnrich,
  inferCausalLinks,
  type ObservationWithDoc,
} from "./amem.ts";

// =============================================================================
// Configuration
// =============================================================================

const HOME = Bun.env.HOME || "/tmp";
export const DEFAULT_EMBED_MODEL = "granite";
export const DEFAULT_RERANK_MODEL = "ExpedientFalcon/qwen3-reranker:0.6b-q8_0";
export const DEFAULT_QUERY_MODEL = "tobil/qmd-query-expansion-1.7B";
export const DEFAULT_GLOB = "**/*.md";
export const DEFAULT_MULTI_GET_MAX_BYTES = 10 * 1024; // 10KB

// Chunking: 800 tokens per chunk with 15% overlap
export const CHUNK_SIZE_TOKENS = 800;
export const CHUNK_OVERLAP_TOKENS = Math.floor(CHUNK_SIZE_TOKENS * 0.15);  // 120 tokens (15% overlap)
// Fallback char-based approximation for sync chunking (~4 chars per token)
export const CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * 4;  // 3200 chars
export const CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * 4;  // 480 chars

// =============================================================================
// Path utilities
// =============================================================================

export function homedir(): string {
  return HOME;
}

export function resolve(...paths: string[]): string {
  if (paths.length === 0) {
    throw new Error("resolve: at least one path segment is required");
  }
  let result = paths[0]!.startsWith('/') ? '' : Bun.env.PWD || process.cwd();
  for (const p of paths) {
    if (p.startsWith('/')) {
      result = p;
    } else {
      result = result + '/' + p;
    }
  }
  const parts = result.split('/').filter(Boolean);
  const normalized: string[] = [];
  for (const part of parts) {
    if (part === '..') normalized.pop();
    else if (part !== '.') normalized.push(part);
  }
  return '/' + normalized.join('/');
}

// Flag to indicate production mode (set by qmd.ts at startup)
let _productionMode = false;

export function enableProductionMode(): void {
  _productionMode = true;
}

export function getDefaultDbPath(indexName: string = "index"): string {
  // Always allow override via INDEX_PATH (for testing)
  if (Bun.env.INDEX_PATH) {
    return Bun.env.INDEX_PATH;
  }

  // In non-production mode (tests), require explicit path
  if (!_productionMode) {
    throw new Error(
      "Database path not set. Tests must set INDEX_PATH env var or use createStore() with explicit path. " +
      "This prevents tests from accidentally writing to the global index."
    );
  }

  const cacheDir = Bun.env.XDG_CACHE_HOME || resolve(homedir(), ".cache");
  const clawmemCacheDir = resolve(cacheDir, "clawmem");
  try { Bun.spawnSync(["mkdir", "-p", clawmemCacheDir]); } catch { }
  return resolve(clawmemCacheDir, `${indexName}.sqlite`);
}

export function getPwd(): string {
  return process.env.PWD || process.cwd();
}

export function getRealPath(path: string): string {
  try {
    return realpathSync(path);
  } catch {
    return resolve(path);
  }
}

// =============================================================================
// Virtual Path Utilities (clawmem://)
// =============================================================================

export type VirtualPath = {
  collectionName: string;
  path: string;  // relative path within collection
};

/**
 * Normalize explicit virtual path formats to standard clawmem:// format.
 * Only handles paths that are already explicitly virtual:
 * - clawmem://collection/path.md (already normalized)
 * - clawmem:////collection/path.md (extra slashes - normalize)
 * - //collection/path.md (missing clawmem: prefix - add it)
 *
 * Does NOT handle:
 * - collection/path.md (bare paths - could be filesystem relative)
 * - :linenum suffix (should be parsed separately before calling this)
 */
export function normalizeVirtualPath(input: string): string {
  let path = input.trim();

  // Handle clawmem:// with extra slashes: clawmem:////collection/path -> clawmem://collection/path
  if (path.startsWith('clawmem:')) {
    // Remove clawmem: prefix and normalize slashes
    path = path.slice(4);
    // Remove leading slashes and re-add exactly two
    path = path.replace(/^\/+/, '');
    return `clawmem://${path}`;
  }

  // Handle //collection/path (missing clawmem: prefix)
  if (path.startsWith('//')) {
    path = path.replace(/^\/+/, '');
    return `clawmem://${path}`;
  }

  // Return as-is for other cases (filesystem paths, docids, bare collection/path, etc.)
  return path;
}

/**
 * Parse a virtual path like "clawmem://collection-name/path/to/file.md"
 * into its components.
 * Also supports collection root: "clawmem://collection-name/" or "clawmem://collection-name"
 */
export function parseVirtualPath(virtualPath: string): VirtualPath | null {
  // Normalize the path first
  const normalized = normalizeVirtualPath(virtualPath);

  // Match: clawmem://collection-name[/optional-path]
  // Allows: clawmem://name, clawmem://name/, clawmem://name/path
  const match = normalized.match(/^clawmem:\/\/([^\/]+)\/?(.*)$/);
  if (!match?.[1]) return null;
  return {
    collectionName: match[1],
    path: match[2] ?? '',  // Empty string for collection root
  };
}

/**
 * Build a virtual path from collection name and relative path.
 */
export function buildVirtualPath(collectionName: string, path: string): string {
  return `clawmem://${collectionName}/${path}`;
}

/**
 * Check if a path is explicitly a virtual path.
 * Only recognizes explicit virtual path formats:
 * - clawmem://collection/path.md
 * - //collection/path.md
 *
 * Does NOT consider bare collection/path.md as virtual - that should be
 * handled separately by checking if the first component is a collection name.
 */
export function isVirtualPath(path: string): boolean {
  const trimmed = path.trim();

  // Explicit clawmem:// prefix (with any number of slashes)
  if (trimmed.startsWith('clawmem:')) return true;

  // //collection/path format (missing clawmem: prefix)
  if (trimmed.startsWith('//')) return true;

  return false;
}

/**
 * Resolve a virtual path to absolute filesystem path.
 */
export function resolveVirtualPath(db: Database, virtualPath: string): string | null {
  const parsed = parseVirtualPath(virtualPath);
  if (!parsed) return null;

  const coll = getCollectionByName(db, parsed.collectionName);
  if (!coll) return null;

  return resolve(coll.pwd, parsed.path);
}

/**
 * Convert an absolute filesystem path to a virtual path.
 * Returns null if the file is not in any indexed collection.
 */
export function toVirtualPath(db: Database, absolutePath: string): string | null {
  // Get all collections from YAML config
  const collections = collectionsListCollections();

  // Find which collection this absolute path belongs to
  for (const coll of collections) {
    if (absolutePath.startsWith(coll.path + '/') || absolutePath === coll.path) {
      // Extract relative path
      const relativePath = absolutePath.startsWith(coll.path + '/')
        ? absolutePath.slice(coll.path.length + 1)
        : '';

      // Verify this document exists in the database
      const doc = db.prepare(`
        SELECT d.path
        FROM documents d
        WHERE d.collection = ? AND d.path = ? AND d.active = 1
        LIMIT 1
      `).get(coll.name, relativePath) as { path: string } | null;

      if (doc) {
        return buildVirtualPath(coll.name, relativePath);
      }
    }
  }

  return null;
}

// =============================================================================
// Database initialization
// =============================================================================

// On macOS, use Homebrew's SQLite which supports extensions
if (process.platform === "darwin") {
  const homebrewSqlitePath = "/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib";
  try {
    if (Bun.file(homebrewSqlitePath).size > 0) {
      Database.setCustomSQLite(homebrewSqlitePath);
    }
  } catch { }
}

function initializeDatabase(db: Database): void {
  sqliteVec.load(db);
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA foreign_keys = ON");

  // Drop legacy tables that are now managed in YAML
  db.exec(`DROP TABLE IF EXISTS path_contexts`);
  db.exec(`DROP TABLE IF EXISTS collections`);

  // Content-addressable storage - the source of truth for document content
  db.exec(`
    CREATE TABLE IF NOT EXISTS content (
      hash TEXT PRIMARY KEY,
      doc TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);

  // Documents table - file system layer mapping virtual paths to content hashes
  // Extended with SAME agent memory metadata columns
  db.exec(`
    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      collection TEXT NOT NULL,
      path TEXT NOT NULL,
      title TEXT NOT NULL,
      hash TEXT NOT NULL,
      created_at TEXT NOT NULL,
      modified_at TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 1,
      domain TEXT,
      workstream TEXT,
      tags TEXT,
      content_type TEXT NOT NULL DEFAULT 'note',
      review_by TEXT,
      confidence REAL NOT NULL DEFAULT 0.5,
      access_count INTEGER NOT NULL DEFAULT 0,
      content_hash TEXT,
      FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE,
      UNIQUE(collection, path)
    )
  `);

  // Migration: add SAME columns to existing databases
  const docCols = db.prepare("PRAGMA table_info(documents)").all() as { name: string }[];
  const colNames = new Set(docCols.map(c => c.name));
  const migrations: [string, string][] = [
    ["domain", "ALTER TABLE documents ADD COLUMN domain TEXT"],
    ["workstream", "ALTER TABLE documents ADD COLUMN workstream TEXT"],
    ["tags", "ALTER TABLE documents ADD COLUMN tags TEXT"],
    ["content_type", "ALTER TABLE documents ADD COLUMN content_type TEXT NOT NULL DEFAULT 'note'"],
    ["review_by", "ALTER TABLE documents ADD COLUMN review_by TEXT"],
    ["confidence", "ALTER TABLE documents ADD COLUMN confidence REAL NOT NULL DEFAULT 0.5"],
    ["access_count", "ALTER TABLE documents ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0"],
    ["content_hash", "ALTER TABLE documents ADD COLUMN content_hash TEXT"],
  ];
  for (const [col, sql] of migrations) {
    if (!colNames.has(col)) {
      try { db.exec(sql); } catch { /* column may already exist */ }
    }
  }

  db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection, active)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path, active)`);

  // Cache table for LLM API calls
  db.exec(`
    CREATE TABLE IF NOT EXISTS llm_cache (
      hash TEXT PRIMARY KEY,
      result TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);

  // Content vectors
  const cvInfo = db.prepare(`PRAGMA table_info(content_vectors)`).all() as { name: string }[];
  const hasSeqColumn = cvInfo.some(col => col.name === 'seq');
  if (cvInfo.length > 0 && !hasSeqColumn) {
    db.exec(`DROP TABLE IF EXISTS content_vectors`);
    db.exec(`DROP TABLE IF EXISTS vectors_vec`);
  }
  db.exec(`
    CREATE TABLE IF NOT EXISTS content_vectors (
      hash TEXT NOT NULL,
      seq INTEGER NOT NULL DEFAULT 0,
      pos INTEGER NOT NULL DEFAULT 0,
      model TEXT NOT NULL,
      embedded_at TEXT NOT NULL,
      PRIMARY KEY (hash, seq)
    )
  `);

  // FTS - index filepath (collection/path), title, and content
  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
      filepath, title, body,
      tokenize='porter unicode61'
    )
  `);

  // Triggers to keep FTS in sync
  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents
    WHEN new.active = 1
    BEGIN
      INSERT INTO documents_fts(rowid, filepath, title, body)
      SELECT
        new.id,
        new.collection || '/' || new.path,
        new.title,
        (SELECT doc FROM content WHERE hash = new.hash)
      WHERE new.active = 1;
    END
  `);

  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
      DELETE FROM documents_fts WHERE rowid = old.id;
    END
  `);

  db.exec(`
    CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents
    BEGIN
      -- Delete from FTS if no longer active
      DELETE FROM documents_fts WHERE rowid = old.id AND new.active = 0;

      -- Update FTS if still/newly active
      INSERT OR REPLACE INTO documents_fts(rowid, filepath, title, body)
      SELECT
        new.id,
        new.collection || '/' || new.path,
        new.title,
        (SELECT doc FROM content WHERE hash = new.hash)
      WHERE new.active = 1;
    END
  `);

  // SAME: Session tracking
  db.exec(`
    CREATE TABLE IF NOT EXISTS session_log (
      session_id TEXT PRIMARY KEY,
      started_at TEXT NOT NULL,
      ended_at TEXT,
      handoff_path TEXT,
      machine TEXT,
      files_changed TEXT,
      summary TEXT
    )
  `);

  // SAME: Context usage tracking (feedback loop)
  db.exec(`
    CREATE TABLE IF NOT EXISTS context_usage (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      timestamp TEXT NOT NULL,
      hook_name TEXT NOT NULL,
      injected_paths TEXT NOT NULL DEFAULT '[]',
      estimated_tokens INTEGER NOT NULL DEFAULT 0,
      was_referenced INTEGER NOT NULL DEFAULT 0
    )
  `);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_context_usage_session ON context_usage(session_id)`);

  // Hook prompt dedupe: suppress duplicate/heartbeat prompts to reduce GPU churn.
  db.exec(`
    CREATE TABLE IF NOT EXISTS hook_dedupe (
      hook_name TEXT NOT NULL,
      prompt_hash TEXT NOT NULL,
      prompt_preview TEXT,
      last_seen_at TEXT NOT NULL,
      PRIMARY KEY (hook_name, prompt_hash)
    )
  `);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_hook_dedupe_last_seen ON hook_dedupe(last_seen_at)`);

  // Migration: add fragment columns to content_vectors
  const cvCols = db.prepare("PRAGMA table_info(content_vectors)").all() as { name: string }[];
  const cvColNames = new Set(cvCols.map(c => c.name));
  const cvMigrations: [string, string][] = [
    ["fragment_type", "ALTER TABLE content_vectors ADD COLUMN fragment_type TEXT"],
    ["fragment_label", "ALTER TABLE content_vectors ADD COLUMN fragment_label TEXT"],
  ];
  for (const [col, sql] of cvMigrations) {
    if (!cvColNames.has(col)) {
      try { db.exec(sql); } catch { /* column may already exist */ }
    }
  }

  // Migration: add observation columns to documents
  const obsMigrations: [string, string][] = [
    ["observation_type", "ALTER TABLE documents ADD COLUMN observation_type TEXT"],
    ["facts", "ALTER TABLE documents ADD COLUMN facts TEXT"],
    ["narrative", "ALTER TABLE documents ADD COLUMN narrative TEXT"],
    ["concepts", "ALTER TABLE documents ADD COLUMN concepts TEXT"],
    ["files_read", "ALTER TABLE documents ADD COLUMN files_read TEXT"],
    ["files_modified", "ALTER TABLE documents ADD COLUMN files_modified TEXT"],
  ];
  for (const [col, sql] of obsMigrations) {
    if (!colNames.has(col)) {
      try { db.exec(sql); } catch { /* column may already exist */ }
    }
  }

  // Migration: add A-MEM columns to documents
  const amemMigrations: [string, string][] = [
    ["amem_keywords", "ALTER TABLE documents ADD COLUMN amem_keywords TEXT"],
    ["amem_tags", "ALTER TABLE documents ADD COLUMN amem_tags TEXT"],
    ["amem_context", "ALTER TABLE documents ADD COLUMN amem_context TEXT"],
  ];
  for (const [col, sql] of amemMigrations) {
    if (!colNames.has(col)) {
      try { db.exec(sql); } catch { /* column may already exist */ }
    }
  }

  // Beads integration tables
  db.exec(`
    CREATE TABLE IF NOT EXISTS beads_issues (
      beads_id TEXT PRIMARY KEY,
      doc_id INTEGER,
      issue_type TEXT,
      status TEXT,
      priority INTEGER,
      tags TEXT,
      assignee TEXT,
      parent_id TEXT,
      created_at TEXT,
      closed_at TEXT,
      last_synced_at TEXT,
      FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
    )
  `);

  db.exec(`CREATE INDEX IF NOT EXISTS idx_beads_status ON beads_issues(status, priority)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_beads_parent ON beads_issues(parent_id)`);

  db.exec(`
    CREATE TABLE IF NOT EXISTS beads_dependencies (
      source_id TEXT NOT NULL,
      target_id TEXT NOT NULL,
      dep_type TEXT NOT NULL,
      created_at TEXT,
      PRIMARY KEY (source_id, target_id, dep_type),
      FOREIGN KEY (source_id) REFERENCES beads_issues(beads_id) ON DELETE CASCADE,
      FOREIGN KEY (target_id) REFERENCES beads_issues(beads_id) ON DELETE CASCADE
    )
  `);

  db.exec(`CREATE INDEX IF NOT EXISTS idx_beads_deps_target ON beads_dependencies(target_id, dep_type)`);

  // MAGMA: Multi-graph relational memory
  db.exec(`
    CREATE TABLE IF NOT EXISTS memory_relations (
      source_id INTEGER NOT NULL,
      target_id INTEGER NOT NULL,
      relation_type TEXT NOT NULL,
      weight REAL DEFAULT 1.0,
      metadata TEXT,
      created_at TEXT,
      PRIMARY KEY (source_id, target_id, relation_type),
      FOREIGN KEY (source_id) REFERENCES documents(id) ON DELETE CASCADE,
      FOREIGN KEY (target_id) REFERENCES documents(id) ON DELETE CASCADE
    )
  `);

  db.exec(`CREATE INDEX IF NOT EXISTS idx_relations_source ON memory_relations(source_id, relation_type)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_relations_target ON memory_relations(target_id, relation_type)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_relations_weight ON memory_relations(weight DESC) WHERE weight > 0.5`);

  // A-MEM: Memory evolution tracking
  db.exec(`
    CREATE TABLE IF NOT EXISTS memory_evolution (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      memory_id INTEGER NOT NULL,
      triggered_by INTEGER NOT NULL,
      version INTEGER NOT NULL DEFAULT 1,
      previous_keywords TEXT,
      new_keywords TEXT,
      previous_context TEXT,
      new_context TEXT,
      reasoning TEXT,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      FOREIGN KEY (memory_id) REFERENCES documents(id) ON DELETE CASCADE,
      FOREIGN KEY (triggered_by) REFERENCES documents(id) ON DELETE CASCADE
    )
  `);

  db.exec(`CREATE INDEX IF NOT EXISTS idx_memory_evolution_memory_id ON memory_evolution(memory_id)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_memory_evolution_triggered_by ON memory_evolution(triggered_by)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_memory_evolution_created_at ON memory_evolution(created_at)`);

  db.exec(`
    CREATE TABLE IF NOT EXISTS entity_nodes (
      entity_id TEXT PRIMARY KEY,
      entity_type TEXT,
      name TEXT,
      description TEXT,
      created_at TEXT
    )
  `);

  db.exec(`
    CREATE TABLE IF NOT EXISTS intent_classifications (
      query_hash TEXT PRIMARY KEY,
      query_text TEXT,
      intent TEXT,
      confidence REAL,
      temporal_start TEXT,
      temporal_end TEXT,
      cached_at TEXT
    )
  `);

  db.exec(`CREATE INDEX IF NOT EXISTS idx_intent_cache_time ON intent_classifications(cached_at)`);
}


function ensureVecTableInternal(db: Database, dimensions: number): void {
  const tableInfo = db.prepare(`SELECT sql FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get() as { sql: string } | null;
  if (tableInfo) {
    const match = tableInfo.sql.match(/float\[(\d+)\]/);
    const hasHashSeq = tableInfo.sql.includes('hash_seq');
    const hasCosine = tableInfo.sql.includes('distance_metric=cosine');
    const existingDims = match?.[1] ? parseInt(match[1], 10) : null;
    if (existingDims === dimensions && hasHashSeq && hasCosine) return;
    // Table exists but wrong schema - need to rebuild
    db.exec("DROP TABLE IF EXISTS vectors_vec");
  }
  db.exec(`CREATE VIRTUAL TABLE vectors_vec USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[${dimensions}] distance_metric=cosine)`);
}

// =============================================================================
// Store Factory
// =============================================================================

export type Store = {
  db: Database;
  dbPath: string;
  close: () => void;
  ensureVecTable: (dimensions: number) => void;

  // Index health
  getHashesNeedingEmbedding: () => number;
  getIndexHealth: () => IndexHealthInfo;
  getStatus: () => IndexStatus;

  // Caching
  getCacheKey: typeof getCacheKey;
  getCachedResult: (cacheKey: string) => string | null;
  setCachedResult: (cacheKey: string, result: string) => void;
  clearCache: () => void;

  // Cleanup and maintenance
  deleteLLMCache: () => number;
  deleteInactiveDocuments: () => number;
  cleanupOrphanedContent: () => number;
  cleanupOrphanedVectors: () => number;
  vacuumDatabase: () => void;

  // Context
  getContextForFile: (filepath: string) => string | null;
  getContextForPath: (collectionName: string, path: string) => string | null;
  getCollectionByName: (name: string) => { name: string; pwd: string; glob_pattern: string } | null;
  getCollectionsWithoutContext: () => { name: string; pwd: string; doc_count: number }[];
  getTopLevelPathsWithoutContext: (collectionName: string) => string[];

  // Virtual paths
  parseVirtualPath: typeof parseVirtualPath;
  buildVirtualPath: typeof buildVirtualPath;
  isVirtualPath: typeof isVirtualPath;
  resolveVirtualPath: (virtualPath: string) => string | null;
  toVirtualPath: (absolutePath: string) => string | null;

  // Search
  searchFTS: (query: string, limit?: number, collectionId?: number) => SearchResult[];
  searchVec: (query: string, model: string, limit?: number, collectionId?: number) => Promise<SearchResult[]>;

  // Query expansion & reranking
  expandQuery: (query: string, model?: string) => Promise<string[]>;
  rerank: (query: string, documents: { file: string; text: string }[], model?: string) => Promise<{ file: string; score: number }[]>;

  // Document retrieval
  findDocument: (filename: string, options?: { includeBody?: boolean }) => DocumentResult | DocumentNotFound;
  getDocumentBody: (doc: DocumentResult | { filepath: string }, fromLine?: number, maxLines?: number) => string | null;
  findDocuments: (pattern: string, options?: { includeBody?: boolean; maxBytes?: number }) => { docs: MultiGetResult[]; errors: string[] };

  // Fuzzy matching and docid lookup
  findSimilarFiles: (query: string, maxDistance?: number, limit?: number) => string[];
  matchFilesByGlob: (pattern: string) => { filepath: string; displayPath: string; bodyLength: number }[];
  findDocumentByDocid: (docid: string) => { filepath: string; hash: string } | null;

  // Document indexing operations
  insertContent: (hash: string, content: string, createdAt: string) => void;
  insertDocument: (collectionName: string, path: string, title: string, hash: string, createdAt: string, modifiedAt: string) => void;
  findActiveDocument: (collectionName: string, path: string) => { id: number; hash: string; title: string } | null;
  findAnyDocument: (collectionName: string, path: string) => { id: number; hash: string; title: string; active: number } | null;
  reactivateDocument: (documentId: number, title: string, hash: string, modifiedAt: string) => void;
  updateDocumentTitle: (documentId: number, title: string, modifiedAt: string) => void;
  updateDocument: (documentId: number, title: string, hash: string, modifiedAt: string) => void;
  deactivateDocument: (collectionName: string, path: string) => void;
  getActiveDocumentPaths: (collectionName: string) => string[];

  // Vector/embedding operations
  getHashesForEmbedding: () => { hash: string; body: string; path: string }[];
  getHashesNeedingFragments: () => { hash: string; body: string; path: string; title: string }[];
  clearAllEmbeddings: () => void;
  insertEmbedding: (hash: string, seq: number, pos: number, embedding: Float32Array, model: string, embeddedAt: string, fragmentType?: string, fragmentLabel?: string) => void;

  // SAME: Observation metadata
  updateObservationFields: (docPath: string, collectionName: string, fields: { observation_type?: string; facts?: string; narrative?: string; concepts?: string; files_read?: string; files_modified?: string }) => void;

  // SAME: Session tracking
  insertSession: (sessionId: string, startedAt: string, machine?: string) => void;
  updateSession: (sessionId: string, updates: { endedAt?: string; handoffPath?: string; filesChanged?: string[]; summary?: string }) => void;
  getSession: (sessionId: string) => SessionRecord | null;
  getRecentSessions: (limit: number) => SessionRecord[];

  // SAME: Context usage tracking
  insertUsage: (usage: UsageRecord) => void;
  getUsageForSession: (sessionId: string) => UsageRow[];
  markUsageReferenced: (id: number) => void;

  // SAME: Document metadata operations
  updateDocumentMeta: (docId: number, meta: { domain?: string; workstream?: string; tags?: string; content_type?: string; review_by?: string; confidence?: number }) => void;
  incrementAccessCount: (paths: string[]) => void;
  getDocumentsByType: (contentType: string, limit?: number) => DocumentRow[];
  getStaleDocuments: (beforeDate: string) => DocumentRow[];

  // Beads integration
  syncBeadsIssues: (beadsJsonlPath: string) => Promise<{ synced: number; created: number; newDocIds: number[] }>;
  detectBeadsProject: (cwd: string) => string | null;

  // MAGMA graph building
  buildTemporalBackbone: () => number;
  buildSemanticGraph: (threshold?: number) => Promise<number>;

  // A-MEM: Self-Evolving Memory
  constructMemoryNote: (llm: any, docId: number) => Promise<any>;
  storeMemoryNote: (docId: number, note: any) => void;
  generateMemoryLinks: (llm: any, docId: number, kNeighbors?: number) => Promise<number>;
  evolveMemories: (llm: any, memoryId: number, triggeredBy: number) => Promise<boolean>;
  postIndexEnrich: (llm: any, docId: number, isNew: boolean) => Promise<void>;
  inferCausalLinks: (llm: any, observations: ObservationWithDoc[]) => Promise<number>;
  findCausalLinks: (docId: number, direction?: 'causes' | 'caused_by' | 'both', maxDepth?: number) => CausalLink[];
  getEvolutionTimeline: (docId: number, limit?: number) => EvolutionEntry[];
};

/**
 * Create a new store instance with the given database path.
 * If no path is provided, uses the default path (~/.cache/qmd/index.sqlite).
 *
 * @param dbPath - Path to the SQLite database file
 * @returns Store instance with all methods bound to the database
 */
export function createStore(dbPath?: string): Store {
  const resolvedPath = dbPath || getDefaultDbPath();
  const db = new Database(resolvedPath);
  initializeDatabase(db);

  return {
    db,
    dbPath: resolvedPath,
    close: () => db.close(),
    ensureVecTable: (dimensions: number) => ensureVecTableInternal(db, dimensions),

    // Index health
    getHashesNeedingEmbedding: () => getHashesNeedingEmbedding(db),
    getIndexHealth: () => getIndexHealth(db),
    getStatus: () => getStatus(db),

    // Caching
    getCacheKey,
    getCachedResult: (cacheKey: string) => getCachedResult(db, cacheKey),
    setCachedResult: (cacheKey: string, result: string) => setCachedResult(db, cacheKey, result),
    clearCache: () => clearCache(db),

    // Cleanup and maintenance
    deleteLLMCache: () => deleteLLMCache(db),
    deleteInactiveDocuments: () => deleteInactiveDocuments(db),
    cleanupOrphanedContent: () => cleanupOrphanedContent(db),
    cleanupOrphanedVectors: () => cleanupOrphanedVectors(db),
    vacuumDatabase: () => vacuumDatabase(db),

    // Context
    getContextForFile: (filepath: string) => getContextForFile(db, filepath),
    getContextForPath: (collectionName: string, path: string) => getContextForPath(db, collectionName, path),
    getCollectionByName: (name: string) => getCollectionByName(db, name),
    getCollectionsWithoutContext: () => getCollectionsWithoutContext(db),
    getTopLevelPathsWithoutContext: (collectionName: string) => getTopLevelPathsWithoutContext(db, collectionName),

    // Virtual paths
    parseVirtualPath,
    buildVirtualPath,
    isVirtualPath,
    resolveVirtualPath: (virtualPath: string) => resolveVirtualPath(db, virtualPath),
    toVirtualPath: (absolutePath: string) => toVirtualPath(db, absolutePath),

    // Search
    searchFTS: (query: string, limit?: number, collectionId?: number) => searchFTS(db, query, limit, collectionId),
    searchVec: (query: string, model: string, limit?: number, collectionId?: number) => searchVec(db, query, model, limit, collectionId),

    // Query expansion & reranking
    expandQuery: (query: string, model?: string) => expandQuery(query, model, db),
    rerank: (query: string, documents: { file: string; text: string }[], model?: string) => rerank(query, documents, model, db),

    // Document retrieval
    findDocument: (filename: string, options?: { includeBody?: boolean }) => findDocument(db, filename, options),
    getDocumentBody: (doc: DocumentResult | { filepath: string }, fromLine?: number, maxLines?: number) => getDocumentBody(db, doc, fromLine, maxLines),
    findDocuments: (pattern: string, options?: { includeBody?: boolean; maxBytes?: number }) => findDocuments(db, pattern, options),

    // Fuzzy matching and docid lookup
    findSimilarFiles: (query: string, maxDistance?: number, limit?: number) => findSimilarFiles(db, query, maxDistance, limit),
    matchFilesByGlob: (pattern: string) => matchFilesByGlob(db, pattern),
    findDocumentByDocid: (docid: string) => findDocumentByDocid(db, docid),

    // Document indexing operations
    insertContent: (hash: string, content: string, createdAt: string) => insertContent(db, hash, content, createdAt),
    insertDocument: (collectionName: string, path: string, title: string, hash: string, createdAt: string, modifiedAt: string) => insertDocument(db, collectionName, path, title, hash, createdAt, modifiedAt),
    findActiveDocument: (collectionName: string, path: string) => findActiveDocument(db, collectionName, path),
    findAnyDocument: (collectionName: string, path: string) => findAnyDocument(db, collectionName, path),
    reactivateDocument: (documentId: number, title: string, hash: string, modifiedAt: string) => reactivateDocument(db, documentId, title, hash, modifiedAt),
    updateDocumentTitle: (documentId: number, title: string, modifiedAt: string) => updateDocumentTitle(db, documentId, title, modifiedAt),
    updateDocument: (documentId: number, title: string, hash: string, modifiedAt: string) => updateDocument(db, documentId, title, hash, modifiedAt),
    deactivateDocument: (collectionName: string, path: string) => deactivateDocument(db, collectionName, path),
    getActiveDocumentPaths: (collectionName: string) => getActiveDocumentPaths(db, collectionName),

    // Vector/embedding operations
    getHashesForEmbedding: () => getHashesForEmbedding(db),
    getHashesNeedingFragments: () => getHashesNeedingFragments(db),
    clearAllEmbeddings: () => clearAllEmbeddings(db),
    insertEmbedding: (hash: string, seq: number, pos: number, embedding: Float32Array, model: string, embeddedAt: string, fragmentType?: string, fragmentLabel?: string) => insertEmbedding(db, hash, seq, pos, embedding, model, embeddedAt, fragmentType, fragmentLabel),

    // SAME: Observation metadata
    updateObservationFields: (docPath: string, collectionName: string, fields) => updateObservationFieldsFn(db, docPath, collectionName, fields),

    // SAME: Session tracking
    insertSession: (sessionId: string, startedAt: string, machine?: string) => insertSessionFn(db, sessionId, startedAt, machine),
    updateSession: (sessionId: string, updates) => updateSessionFn(db, sessionId, updates),
    getSession: (sessionId: string) => getSessionFn(db, sessionId),
    getRecentSessions: (limit: number) => getRecentSessionsFn(db, limit),

    // SAME: Context usage tracking
    insertUsage: (usage: UsageRecord) => insertUsageFn(db, usage),
    getUsageForSession: (sessionId: string) => getUsageForSessionFn(db, sessionId),
    markUsageReferenced: (id: number) => markUsageReferencedFn(db, id),

    // SAME: Document metadata operations
    updateDocumentMeta: (docId: number, meta) => updateDocumentMetaFn(db, docId, meta),
    incrementAccessCount: (paths: string[]) => incrementAccessCountFn(db, paths),
    getDocumentsByType: (contentType: string, limit?: number) => getDocumentsByTypeFn(db, contentType, limit),
    getStaleDocuments: (beforeDate: string) => getStaleDocumentsFn(db, beforeDate),

    // Beads integration
    syncBeadsIssues: (beadsJsonlPath: string) => syncBeadsIssues(db, beadsJsonlPath),
    detectBeadsProject,

    // MAGMA graph building
    buildTemporalBackbone: () => buildTemporalBackbone(db),
    buildSemanticGraph: (threshold?: number) => buildSemanticGraph(db, threshold),

    // A-MEM: Self-Evolving Memory
    constructMemoryNote: (llm: any, docId: number) => constructMemoryNote({ db, dbPath: resolvedPath } as Store, llm, docId),
    storeMemoryNote: (docId: number, note: any) => storeMemoryNote({ db, dbPath: resolvedPath } as Store, docId, note),
    generateMemoryLinks: (llm: any, docId: number, kNeighbors?: number) => generateMemoryLinks({ db, dbPath: resolvedPath } as Store, llm, docId, kNeighbors),
    evolveMemories: (llm: any, memoryId: number, triggeredBy: number) => evolveMemories({ db, dbPath: resolvedPath } as Store, llm, memoryId, triggeredBy),
    postIndexEnrich: (llm: any, docId: number, isNew: boolean) => postIndexEnrich({ db, dbPath: resolvedPath } as Store, llm, docId, isNew),
    inferCausalLinks: (llm: any, observations: ObservationWithDoc[]) => inferCausalLinks({ db, dbPath: resolvedPath } as Store, llm, observations),
    findCausalLinks: (docId: number, direction?: 'causes' | 'caused_by' | 'both', maxDepth?: number) => findCausalLinks(db, docId, direction, maxDepth),
    getEvolutionTimeline: (docId: number, limit?: number) => getEvolutionTimeline(db, docId, limit),
  };
}

// =============================================================================
// Core Document Type
// =============================================================================

/**
 * Unified document result type with all metadata.
 * Body is optional - use getDocumentBody() to load it separately if needed.
 */
export type DocumentResult = {
  filepath: string;           // Full filesystem path
  displayPath: string;        // Short display path (e.g., "docs/readme.md")
  title: string;              // Document title (from first heading or filename)
  context: string | null;     // Folder context description if configured
  hash: string;               // Content hash for caching/change detection
  docid: string;              // Short docid (first 6 chars of hash) for quick reference
  collectionName: string;     // Parent collection name
  modifiedAt: string;         // Last modification timestamp
  bodyLength: number;         // Body length in bytes (useful before loading)
  body?: string;              // Document body (optional, load with getDocumentBody)
};

/**
 * Extract short docid from a full hash (first 6 characters).
 */
export function getDocid(hash: string): string {
  return hash.slice(0, 6);
}

/**
 * Handelize a filename to be more token-friendly.
 * - Convert triple underscore `___` to `/` (folder separator)
 * - Convert to lowercase
 * - Replace sequences of non-word chars (except /) with single dash
 * - Remove leading/trailing dashes from path segments
 * - Preserve folder structure (a/b/c/d.md stays structured)
 * - Preserve file extension
 */
export function handelize(path: string): string {
  if (!path || path.trim() === '') {
    throw new Error('handelize: path cannot be empty');
  }

  // Check for paths that are just extensions or only dots/special chars
  // A valid path must have at least one letter or digit (including Unicode)
  const segments = path.split('/').filter(Boolean);
  const lastSegment = segments[segments.length - 1] || '';
  const filenameWithoutExt = lastSegment.replace(/\.[^.]+$/, '');
  const hasValidContent = /[\p{L}\p{N}]/u.test(filenameWithoutExt);
  if (!hasValidContent) {
    throw new Error(`handelize: path "${path}" has no valid filename content`);
  }

  const result = path
    .replace(/___/g, '/')       // Triple underscore becomes folder separator
    .toLowerCase()
    .split('/')
    .map((segment, idx, arr) => {
      const isLastSegment = idx === arr.length - 1;

      if (isLastSegment) {
        // For the filename (last segment), preserve the extension
        const extMatch = segment.match(/(\.[a-z0-9]+)$/i);
        const ext = extMatch ? extMatch[1] : '';
        const nameWithoutExt = ext ? segment.slice(0, -ext.length) : segment;

        const cleanedName = nameWithoutExt
          .replace(/[^\p{L}\p{N}]+/gu, '-')  // Replace non-letter/digit chars with dash
          .replace(/^-+|-+$/g, ''); // Remove leading/trailing dashes

        return cleanedName + ext;
      } else {
        // For directories, just clean normally
        return segment
          .replace(/[^\p{L}\p{N}]+/gu, '-')
          .replace(/^-+|-+$/g, '');
      }
    })
    .filter(Boolean)
    .join('/');

  if (!result) {
    throw new Error(`handelize: path "${path}" resulted in empty string after processing`);
  }

  return result;
}

/**
 * Search result extends DocumentResult with score and source info
 */
export type SearchResult = DocumentResult & {
  score: number;              // Relevance score (0-1)
  source: "fts" | "vec";      // Search source (full-text or vector)
  chunkPos?: number;          // Character position of matching chunk (for vector search)
  fragmentType?: string;      // Fragment type (section, list, code, frontmatter, fact, narrative)
  fragmentLabel?: string;     // Fragment label (heading text, fm key, etc.)
};

/**
 * Ranked result for RRF fusion (simplified, used internally)
 */
export type RankedResult = {
  file: string;
  displayPath: string;
  title: string;
  body: string;
  score: number;
};

/**
 * Error result when document is not found
 */
export type DocumentNotFound = {
  error: "not_found";
  query: string;
  similarFiles: string[];
};

/**
 * Result from multi-get operations
 */
export type MultiGetResult = {
  doc: DocumentResult;
  skipped: false;
} | {
  doc: Pick<DocumentResult, "filepath" | "displayPath">;
  skipped: true;
  skipReason: string;
};

export type CollectionInfo = {
  name: string;
  path: string;
  pattern: string;
  documents: number;
  lastUpdated: string;
};

export type IndexStatus = {
  totalDocuments: number;
  needsEmbedding: number;
  hasVectorIndex: boolean;
  collections: CollectionInfo[];
};

// =============================================================================
// SAME: Agent Memory Types
// =============================================================================

export type SessionRecord = {
  sessionId: string;
  startedAt: string;
  endedAt: string | null;
  handoffPath: string | null;
  machine: string | null;
  filesChanged: string[];
  summary: string | null;
};

export type UsageRecord = {
  sessionId: string;
  timestamp: string;
  hookName: string;
  injectedPaths: string[];
  estimatedTokens: number;
  wasReferenced: number;
};

export type UsageRow = {
  id: number;
  sessionId: string;
  timestamp: string;
  hookName: string;
  injectedPaths: string;
  estimatedTokens: number;
  wasReferenced: number;
};

export type DocumentRow = {
  id: number;
  collection: string;
  path: string;
  title: string;
  hash: string;
  modifiedAt: string;
  domain: string | null;
  workstream: string | null;
  tags: string | null;
  contentType: string;
  reviewBy: string | null;
  confidence: number;
  accessCount: number;
  bodyLength: number;
};

// =============================================================================
// Index health
// =============================================================================

export function getHashesNeedingEmbedding(db: Database): number {
  const result = db.prepare(`
    SELECT COUNT(DISTINCT d.hash) as count
    FROM documents d
    LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
    WHERE d.active = 1 AND v.hash IS NULL
  `).get() as { count: number };
  return result.count;
}

export type IndexHealthInfo = {
  needsEmbedding: number;
  totalDocs: number;
  daysStale: number | null;
};

export function getIndexHealth(db: Database): IndexHealthInfo {
  const needsEmbedding = getHashesNeedingEmbedding(db);
  const totalDocs = (db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 1`).get() as { count: number }).count;

  const mostRecent = db.prepare(`SELECT MAX(modified_at) as latest FROM documents WHERE active = 1`).get() as { latest: string | null };
  let daysStale: number | null = null;
  if (mostRecent?.latest) {
    const lastUpdate = new Date(mostRecent.latest);
    daysStale = Math.floor((Date.now() - lastUpdate.getTime()) / (24 * 60 * 60 * 1000));
  }

  return { needsEmbedding, totalDocs, daysStale };
}

// =============================================================================
// Caching
// =============================================================================

export function getCacheKey(url: string, body: object): string {
  const hash = new Bun.CryptoHasher("sha256");
  hash.update(url);
  hash.update(JSON.stringify(body));
  return hash.digest("hex");
}

export function getCachedResult(db: Database, cacheKey: string): string | null {
  const row = db.prepare(`SELECT result FROM llm_cache WHERE hash = ?`).get(cacheKey) as { result: string } | null;
  return row?.result || null;
}

export function setCachedResult(db: Database, cacheKey: string, result: string): void {
  const now = new Date().toISOString();
  db.prepare(`INSERT OR REPLACE INTO llm_cache (hash, result, created_at) VALUES (?, ?, ?)`).run(cacheKey, result, now);
  if (Math.random() < 0.01) {
    db.exec(`DELETE FROM llm_cache WHERE hash NOT IN (SELECT hash FROM llm_cache ORDER BY created_at DESC LIMIT 1000)`);
  }
}

export function clearCache(db: Database): void {
  db.exec(`DELETE FROM llm_cache`);
}

// =============================================================================
// Cleanup and maintenance operations
// =============================================================================

/**
 * Delete cached LLM API responses.
 * Returns the number of cached responses deleted.
 */
export function deleteLLMCache(db: Database): number {
  const result = db.prepare(`DELETE FROM llm_cache`).run();
  return result.changes;
}

/**
 * Remove inactive document records (active = 0).
 * Returns the number of inactive documents deleted.
 */
export function deleteInactiveDocuments(db: Database): number {
  const result = db.prepare(`DELETE FROM documents WHERE active = 0`).run();
  return result.changes;
}

/**
 * Remove orphaned content hashes that are not referenced by any active document.
 * Returns the number of orphaned content hashes deleted.
 */
export function cleanupOrphanedContent(db: Database): number {
  const result = db.prepare(`
    DELETE FROM content
    WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
  `).run();
  return result.changes;
}

/**
 * Remove orphaned vector embeddings that are not referenced by any active document.
 * Returns the number of orphaned embedding chunks deleted.
 */
export function cleanupOrphanedVectors(db: Database): number {
  // Check if vectors_vec table exists
  const tableExists = db.prepare(`
    SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'
  `).get();

  if (!tableExists) {
    return 0;
  }

  // Count orphaned vectors first
  const countResult = db.prepare(`
    SELECT COUNT(*) as c FROM content_vectors cv
    WHERE NOT EXISTS (
      SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1
    )
  `).get() as { c: number };

  if (countResult.c === 0) {
    return 0;
  }

  // Delete from vectors_vec first
  db.exec(`
    DELETE FROM vectors_vec WHERE hash_seq IN (
      SELECT cv.hash || '_' || cv.seq FROM content_vectors cv
      WHERE NOT EXISTS (
        SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1
      )
    )
  `);

  // Delete from content_vectors
  db.exec(`
    DELETE FROM content_vectors WHERE hash NOT IN (
      SELECT hash FROM documents WHERE active = 1
    )
  `);

  return countResult.c;
}

/**
 * Run VACUUM to reclaim unused space in the database.
 * This operation rebuilds the database file to eliminate fragmentation.
 */
export function vacuumDatabase(db: Database): void {
  db.exec(`VACUUM`);
}

// =============================================================================
// Document helpers
// =============================================================================

export async function hashContent(content: string): Promise<string> {
  const hash = new Bun.CryptoHasher("sha256");
  hash.update(content);
  return hash.digest("hex");
}

export function extractTitle(content: string, filename: string): string {
  const match = content.match(/^##?\s+(.+)$/m);
  if (match) {
    const title = (match[1] ?? "").trim();
    if (title === "📝 Notes" || title === "Notes") {
      const nextMatch = content.match(/^##\s+(.+)$/m);
      if (nextMatch?.[1]) return nextMatch[1].trim();
    }
    return title;
  }
  return filename.replace(/\.md$/, "").split("/").pop() || filename;
}

// =============================================================================
// Document indexing operations
// =============================================================================

/**
 * Insert content into the content table (content-addressable storage).
 * Uses INSERT OR IGNORE so duplicate hashes are skipped.
 */
export function insertContent(db: Database, hash: string, content: string, createdAt: string): void {
  db.prepare(`INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)`)
    .run(hash, content, createdAt);
}

/**
 * Insert a new document into the documents table.
 */
export function insertDocument(
  db: Database,
  collectionName: string,
  path: string,
  title: string,
  hash: string,
  createdAt: string,
  modifiedAt: string
): void {
  db.prepare(`
    INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
    VALUES (?, ?, ?, ?, ?, ?, 1)
  `).run(collectionName, path, title, hash, createdAt, modifiedAt);
}

/**
 * Find an active document by collection name and path.
 */
export function findActiveDocument(
  db: Database,
  collectionName: string,
  path: string
): { id: number; hash: string; title: string } | null {
  return db.prepare(`
    SELECT id, hash, title FROM documents
    WHERE collection = ? AND path = ? AND active = 1
  `).get(collectionName, path) as { id: number; hash: string; title: string } | null;
}

/**
 * Find a document by collection and path, regardless of active status.
 * Used to detect inactive rows that block re-insertion (UNIQUE constraint).
 */
export function findAnyDocument(
  db: Database,
  collectionName: string,
  path: string
): { id: number; hash: string; title: string; active: number } | null {
  return db.prepare(`
    SELECT id, hash, title, active FROM documents
    WHERE collection = ? AND path = ?
  `).get(collectionName, path) as { id: number; hash: string; title: string; active: number } | null;
}

/**
 * Reactivate an inactive document with updated content.
 */
export function reactivateDocument(
  db: Database,
  documentId: number,
  title: string,
  hash: string,
  modifiedAt: string
): void {
  db.prepare(`UPDATE documents SET active = 1, title = ?, hash = ?, modified_at = ? WHERE id = ?`)
    .run(title, hash, modifiedAt, documentId);
}

/**
 * Update the title and modified_at timestamp for a document.
 */
export function updateDocumentTitle(
  db: Database,
  documentId: number,
  title: string,
  modifiedAt: string
): void {
  db.prepare(`UPDATE documents SET title = ?, modified_at = ? WHERE id = ?`)
    .run(title, modifiedAt, documentId);
}

/**
 * Update an existing document's hash, title, and modified_at timestamp.
 * Used when content changes but the file path stays the same.
 */
export function updateDocument(
  db: Database,
  documentId: number,
  title: string,
  hash: string,
  modifiedAt: string
): void {
  db.prepare(`UPDATE documents SET title = ?, hash = ?, modified_at = ? WHERE id = ?`)
    .run(title, hash, modifiedAt, documentId);
}

/**
 * Deactivate a document (mark as inactive but don't delete).
 */
export function deactivateDocument(db: Database, collectionName: string, path: string): void {
  db.prepare(`UPDATE documents SET active = 0 WHERE collection = ? AND path = ? AND active = 1`)
    .run(collectionName, path);
}

/**
 * Get all active document paths for a collection.
 */
export function getActiveDocumentPaths(db: Database, collectionName: string): string[] {
  const rows = db.prepare(`
    SELECT path FROM documents WHERE collection = ? AND active = 1
  `).all(collectionName) as { path: string }[];
  return rows.map(r => r.path);
}

export { formatQueryForEmbedding, formatDocForEmbedding };

export function chunkDocument(content: string, maxChars: number = CHUNK_SIZE_CHARS, overlapChars: number = CHUNK_OVERLAP_CHARS): { text: string; pos: number }[] {
  if (content.length <= maxChars) {
    return [{ text: content, pos: 0 }];
  }

  const chunks: { text: string; pos: number }[] = [];
  let charPos = 0;

  while (charPos < content.length) {
    // Calculate end position for this chunk
    let endPos = Math.min(charPos + maxChars, content.length);

    // If not at the end, try to find a good break point
    if (endPos < content.length) {
      const slice = content.slice(charPos, endPos);

      // Look for break points in the last 30% of the chunk
      const searchStart = Math.floor(slice.length * 0.7);
      const searchSlice = slice.slice(searchStart);

      // Priority: paragraph > sentence > line > word
      let breakOffset = -1;
      const paragraphBreak = searchSlice.lastIndexOf('\n\n');
      if (paragraphBreak >= 0) {
        breakOffset = searchStart + paragraphBreak + 2;
      } else {
        const sentenceEnd = Math.max(
          searchSlice.lastIndexOf('. '),
          searchSlice.lastIndexOf('.\n'),
          searchSlice.lastIndexOf('? '),
          searchSlice.lastIndexOf('?\n'),
          searchSlice.lastIndexOf('! '),
          searchSlice.lastIndexOf('!\n')
        );
        if (sentenceEnd >= 0) {
          breakOffset = searchStart + sentenceEnd + 2;
        } else {
          const lineBreak = searchSlice.lastIndexOf('\n');
          if (lineBreak >= 0) {
            breakOffset = searchStart + lineBreak + 1;
          } else {
            const spaceBreak = searchSlice.lastIndexOf(' ');
            if (spaceBreak >= 0) {
              breakOffset = searchStart + spaceBreak + 1;
            }
          }
        }
      }

      if (breakOffset > 0) {
        endPos = charPos + breakOffset;
      }
    }

    // Ensure we make progress
    if (endPos <= charPos) {
      endPos = Math.min(charPos + maxChars, content.length);
    }

    chunks.push({ text: content.slice(charPos, endPos), pos: charPos });

    // Move forward, but overlap with previous chunk
    // For last chunk, don't overlap (just go to the end)
    if (endPos >= content.length) {
      break;
    }
    charPos = endPos - overlapChars;
    const lastChunkPos = chunks.at(-1)!.pos;
    if (charPos <= lastChunkPos) {
      // Prevent infinite loop - move forward at least a bit
      charPos = endPos;
    }
  }

  return chunks;
}

/**
 * Chunk a document by actual token count using the LLM tokenizer.
 * More accurate than character-based chunking but requires async.
 */
export async function chunkDocumentByTokens(
  content: string,
  maxTokens: number = CHUNK_SIZE_TOKENS,
  overlapTokens: number = CHUNK_OVERLAP_TOKENS
): Promise<{ text: string; pos: number; tokens: number }[]> {
  const llm = getDefaultLlamaCpp();

  // Tokenize once upfront
  const allTokens = await llm.tokenize(content);
  const totalTokens = allTokens.length;

  if (totalTokens <= maxTokens) {
    return [{ text: content, pos: 0, tokens: totalTokens }];
  }

  const chunks: { text: string; pos: number; tokens: number }[] = [];
  const step = maxTokens - overlapTokens;
  const avgCharsPerToken = content.length / totalTokens;
  let tokenPos = 0;

  while (tokenPos < totalTokens) {
    const chunkEnd = Math.min(tokenPos + maxTokens, totalTokens);
    const chunkTokens = allTokens.slice(tokenPos, chunkEnd);
    let chunkText = await llm.detokenize(chunkTokens);

    // Find a good break point if not at end of document
    if (chunkEnd < totalTokens) {
      const searchStart = Math.floor(chunkText.length * 0.7);
      const searchSlice = chunkText.slice(searchStart);

      let breakOffset = -1;
      const paragraphBreak = searchSlice.lastIndexOf('\n\n');
      if (paragraphBreak >= 0) {
        breakOffset = paragraphBreak + 2;
      } else {
        const sentenceEnd = Math.max(
          searchSlice.lastIndexOf('. '),
          searchSlice.lastIndexOf('.\n'),
          searchSlice.lastIndexOf('? '),
          searchSlice.lastIndexOf('?\n'),
          searchSlice.lastIndexOf('! '),
          searchSlice.lastIndexOf('!\n')
        );
        if (sentenceEnd >= 0) {
          breakOffset = sentenceEnd + 2;
        } else {
          const lineBreak = searchSlice.lastIndexOf('\n');
          if (lineBreak >= 0) {
            breakOffset = lineBreak + 1;
          }
        }
      }

      if (breakOffset >= 0) {
        chunkText = chunkText.slice(0, searchStart + breakOffset);
      }
    }

    // Approximate character position based on token position
    const charPos = Math.floor(tokenPos * avgCharsPerToken);
    chunks.push({ text: chunkText, pos: charPos, tokens: chunkTokens.length });

    // Move forward
    if (chunkEnd >= totalTokens) break;

    // Advance by step tokens (maxTokens - overlap)
    tokenPos += step;
  }

  return chunks;
}

// =============================================================================
// Fuzzy matching
// =============================================================================

function levenshtein(a: string, b: string): number {
  const m = a.length, n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i]![0] = i;
  for (let j = 0; j <= n; j++) dp[0]![j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i]![j] = Math.min(
        dp[i - 1]![j]! + 1,
        dp[i]![j - 1]! + 1,
        dp[i - 1]![j - 1]! + cost
      );
    }
  }
  return dp[m]![n]!;
}

/**
 * Find a document by its short docid (first 6 characters of hash).
 * Returns the document's virtual path if found, null otherwise.
 * If multiple documents match the same short hash (collision), returns the first one.
 */
export function findDocumentByDocid(db: Database, docid: string): { filepath: string; hash: string } | null {
  // Normalize: remove leading # if present
  const shortHash = docid.startsWith('#') ? docid.slice(1) : docid;

  if (shortHash.length < 1) return null;

  // Look up documents where hash starts with the short hash
  const doc = db.prepare(`
    SELECT 'clawmem://' || d.collection || '/' || d.path as filepath, d.hash
    FROM documents d
    WHERE d.hash LIKE ? AND d.active = 1
    LIMIT 1
  `).get(`${shortHash}%`) as { filepath: string; hash: string } | null;

  return doc;
}

export function findSimilarFiles(db: Database, query: string, maxDistance: number = 3, limit: number = 5): string[] {
  const allFiles = db.prepare(`
    SELECT d.path
    FROM documents d
    WHERE d.active = 1
  `).all() as { path: string }[];
  const queryLower = query.toLowerCase();
  const scored = allFiles
    .map(f => ({ path: f.path, dist: levenshtein(f.path.toLowerCase(), queryLower) }))
    .filter(f => f.dist <= maxDistance)
    .sort((a, b) => a.dist - b.dist)
    .slice(0, limit);
  return scored.map(f => f.path);
}

export function matchFilesByGlob(db: Database, pattern: string): { filepath: string; displayPath: string; bodyLength: number }[] {
  const allFiles = db.prepare(`
    SELECT
      'clawmem://' || d.collection || '/' || d.path as virtual_path,
      LENGTH(content.doc) as body_length,
      d.path,
      d.collection
    FROM documents d
    JOIN content ON content.hash = d.hash
    WHERE d.active = 1
  `).all() as { virtual_path: string; body_length: number; path: string; collection: string }[];

  const glob = new Glob(pattern);
  return allFiles
    .filter(f => glob.match(f.virtual_path) || glob.match(f.path))
    .map(f => ({
      filepath: f.virtual_path,  // Virtual path for precise lookup
      displayPath: f.path,        // Relative path for display
      bodyLength: f.body_length
    }));
}

// =============================================================================
// Context
// =============================================================================

/**
 * Get context for a file path using hierarchical inheritance.
 * Contexts are collection-scoped and inherit from parent directories.
 * For example, context at "/talks" applies to "/talks/2024/keynote.md".
 *
 * @param db Database instance (unused - kept for compatibility)
 * @param collectionName Collection name
 * @param path Relative path within the collection
 * @returns Context string or null if no context is defined
 */
export function getContextForPath(db: Database, collectionName: string, path: string): string | null {
  const config = collectionsLoadConfig();
  const coll = getCollection(collectionName);

  if (!coll) return null;

  // Collect ALL matching contexts (global + all path prefixes)
  const contexts: string[] = [];

  // Add global context if present
  if (config.global_context) {
    contexts.push(config.global_context);
  }

  // Add all matching path contexts (from most general to most specific)
  if (coll.context) {
    const normalizedPath = path.startsWith("/") ? path : `/${path}`;

    // Collect all matching prefixes
    const matchingContexts: { prefix: string; context: string }[] = [];
    for (const [prefix, context] of Object.entries(coll.context)) {
      const normalizedPrefix = prefix.startsWith("/") ? prefix : `/${prefix}`;
      if (normalizedPath.startsWith(normalizedPrefix)) {
        matchingContexts.push({ prefix: normalizedPrefix, context });
      }
    }

    // Sort by prefix length (shortest/most general first)
    matchingContexts.sort((a, b) => a.prefix.length - b.prefix.length);

    // Add all matching contexts
    for (const match of matchingContexts) {
      contexts.push(match.context);
    }
  }

  // Join all contexts with double newline
  return contexts.length > 0 ? contexts.join('\n\n') : null;
}

/**
 * Get context for a file path (virtual or filesystem).
 * Resolves the collection and relative path using the YAML collections config.
 */
export function getContextForFile(db: Database, filepath: string): string | null {
  // Handle undefined or null filepath
  if (!filepath) return null;

  // Get all collections from YAML config
  const collections = collectionsListCollections();
  const config = collectionsLoadConfig();

  // Parse virtual path format: clawmem://collection/path
  let collectionName: string | null = null;
  let relativePath: string | null = null;

  const parsedVirtual = filepath.startsWith('clawmem://') ? parseVirtualPath(filepath) : null;
  if (parsedVirtual) {
    collectionName = parsedVirtual.collectionName;
    relativePath = parsedVirtual.path;
  } else {
    // Filesystem path: find which collection this absolute path belongs to
    for (const coll of collections) {
      // Skip collections with missing paths
      if (!coll || !coll.path) continue;

      if (filepath.startsWith(coll.path + '/') || filepath === coll.path) {
        collectionName = coll.name;
        // Extract relative path
        relativePath = filepath.startsWith(coll.path + '/')
          ? filepath.slice(coll.path.length + 1)
          : '';
        break;
      }
    }

    if (!collectionName || relativePath === null) return null;
  }

  // Get the collection from config
  const coll = getCollection(collectionName);
  if (!coll) return null;

  // Verify this document exists in the database
  const doc = db.prepare(`
    SELECT d.path
    FROM documents d
    WHERE d.collection = ? AND d.path = ? AND d.active = 1
    LIMIT 1
  `).get(collectionName, relativePath) as { path: string } | null;

  if (!doc) return null;

  // Collect ALL matching contexts (global + all path prefixes)
  const contexts: string[] = [];

  // Add global context if present
  if (config.global_context) {
    contexts.push(config.global_context);
  }

  // Add all matching path contexts (from most general to most specific)
  if (coll.context) {
    const normalizedPath = relativePath.startsWith("/") ? relativePath : `/${relativePath}`;

    // Collect all matching prefixes
    const matchingContexts: { prefix: string; context: string }[] = [];
    for (const [prefix, context] of Object.entries(coll.context)) {
      const normalizedPrefix = prefix.startsWith("/") ? prefix : `/${prefix}`;
      if (normalizedPath.startsWith(normalizedPrefix)) {
        matchingContexts.push({ prefix: normalizedPrefix, context });
      }
    }

    // Sort by prefix length (shortest/most general first)
    matchingContexts.sort((a, b) => a.prefix.length - b.prefix.length);

    // Add all matching contexts
    for (const match of matchingContexts) {
      contexts.push(match.context);
    }
  }

  // Join all contexts with double newline
  return contexts.length > 0 ? contexts.join('\n\n') : null;
}

/**
 * Get collection by name from YAML config.
 * Returns collection metadata from ~/.config/qmd/index.yml
 */
export function getCollectionByName(db: Database, name: string): { name: string; pwd: string; glob_pattern: string } | null {
  const collection = getCollection(name);
  if (!collection) return null;

  return {
    name: collection.name,
    pwd: collection.path,
    glob_pattern: collection.pattern,
  };
}

/**
 * List all collections with document counts from database.
 * Merges YAML config with database statistics.
 */
export function listCollections(db: Database): { name: string; pwd: string; glob_pattern: string; doc_count: number; active_count: number; last_modified: string | null }[] {
  const collections = collectionsListCollections();

  // Get document counts from database for each collection
  const result = collections.map(coll => {
    const stats = db.prepare(`
      SELECT
        COUNT(d.id) as doc_count,
        SUM(CASE WHEN d.active = 1 THEN 1 ELSE 0 END) as active_count,
        MAX(d.modified_at) as last_modified
      FROM documents d
      WHERE d.collection = ?
    `).get(coll.name) as { doc_count: number; active_count: number; last_modified: string | null } | null;

    return {
      name: coll.name,
      pwd: coll.path,
      glob_pattern: coll.pattern,
      doc_count: stats?.doc_count || 0,
      active_count: stats?.active_count || 0,
      last_modified: stats?.last_modified || null,
    };
  });

  return result;
}

/**
 * Remove a collection and clean up its documents.
 * Uses collections.ts to remove from YAML config and cleans up database.
 */
export function removeCollection(db: Database, collectionName: string): { deletedDocs: number; cleanedHashes: number } {
  // Delete documents from database
  const docResult = db.prepare(`DELETE FROM documents WHERE collection = ?`).run(collectionName);

  // Clean up orphaned content hashes
  const cleanupResult = db.prepare(`
    DELETE FROM content
    WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
  `).run();

  // Remove from YAML config (returns true if found and removed)
  collectionsRemoveCollection(collectionName);

  return {
    deletedDocs: docResult.changes,
    cleanedHashes: cleanupResult.changes
  };
}

/**
 * Rename a collection.
 * Updates both YAML config and database documents table.
 */
export function renameCollection(db: Database, oldName: string, newName: string): void {
  // Update all documents with the new collection name in database
  db.prepare(`UPDATE documents SET collection = ? WHERE collection = ?`)
    .run(newName, oldName);

  // Rename in YAML config
  collectionsRenameCollection(oldName, newName);
}

// =============================================================================
// Context Management Operations
// =============================================================================

/**
 * Insert or update a context for a specific collection and path prefix.
 */
export function insertContext(db: Database, collectionId: number, pathPrefix: string, context: string): void {
  // Get collection name from ID
  const coll = db.prepare(`SELECT name FROM collections WHERE id = ?`).get(collectionId) as { name: string } | null;
  if (!coll) {
    throw new Error(`Collection with id ${collectionId} not found`);
  }

  // Use collections.ts to add context
  collectionsAddContext(coll.name, pathPrefix, context);
}

/**
 * Delete a context for a specific collection and path prefix.
 * Returns the number of contexts deleted.
 */
export function deleteContext(db: Database, collectionName: string, pathPrefix: string): number {
  // Use collections.ts to remove context
  const success = collectionsRemoveContext(collectionName, pathPrefix);
  return success ? 1 : 0;
}

/**
 * Delete all global contexts (contexts with empty path_prefix).
 * Returns the number of contexts deleted.
 */
export function deleteGlobalContexts(db: Database): number {
  let deletedCount = 0;

  // Remove global context
  setGlobalContext(undefined);
  deletedCount++;

  // Remove root context (empty string) from all collections
  const collections = collectionsListCollections();
  for (const coll of collections) {
    const success = collectionsRemoveContext(coll.name, '');
    if (success) {
      deletedCount++;
    }
  }

  return deletedCount;
}

/**
 * List all contexts, grouped by collection.
 * Returns contexts ordered by collection name, then by path prefix length (longest first).
 */
export function listPathContexts(db: Database): { collection_name: string; path_prefix: string; context: string }[] {
  const allContexts = collectionsListAllContexts();

  // Convert to expected format and sort
  return allContexts.map(ctx => ({
    collection_name: ctx.collection,
    path_prefix: ctx.path,
    context: ctx.context,
  })).sort((a, b) => {
    // Sort by collection name first
    if (a.collection_name !== b.collection_name) {
      return a.collection_name.localeCompare(b.collection_name);
    }
    // Then by path prefix length (longest first)
    if (a.path_prefix.length !== b.path_prefix.length) {
      return b.path_prefix.length - a.path_prefix.length;
    }
    // Then alphabetically
    return a.path_prefix.localeCompare(b.path_prefix);
  });
}

/**
 * Get all collections (name only - from YAML config).
 */
export function getAllCollections(db: Database): { name: string }[] {
  const collections = collectionsListCollections();
  return collections.map(c => ({ name: c.name }));
}

/**
 * Check which collections don't have any context defined.
 * Returns collections that have no context entries at all (not even root context).
 */
export function getCollectionsWithoutContext(db: Database): { name: string; pwd: string; doc_count: number }[] {
  // Get all collections from YAML config
  const yamlCollections = collectionsListCollections();

  // Filter to those without context
  const collectionsWithoutContext: { name: string; pwd: string; doc_count: number }[] = [];

  for (const coll of yamlCollections) {
    // Check if collection has any context
    if (!coll.context || Object.keys(coll.context).length === 0) {
      // Get doc count from database
      const stats = db.prepare(`
        SELECT COUNT(d.id) as doc_count
        FROM documents d
        WHERE d.collection = ? AND d.active = 1
      `).get(coll.name) as { doc_count: number } | null;

      collectionsWithoutContext.push({
        name: coll.name,
        pwd: coll.path,
        doc_count: stats?.doc_count || 0,
      });
    }
  }

  return collectionsWithoutContext.sort((a, b) => a.name.localeCompare(b.name));
}

/**
 * Get top-level directories in a collection that don't have context.
 * Useful for suggesting where context might be needed.
 */
export function getTopLevelPathsWithoutContext(db: Database, collectionName: string): string[] {
  // Get all paths in the collection from database
  const paths = db.prepare(`
    SELECT DISTINCT path FROM documents
    WHERE collection = ? AND active = 1
  `).all(collectionName) as { path: string }[];

  // Get existing contexts for this collection from YAML
  const yamlColl = getCollection(collectionName);
  if (!yamlColl) return [];

  const contextPrefixes = new Set<string>();
  if (yamlColl.context) {
    for (const prefix of Object.keys(yamlColl.context)) {
      contextPrefixes.add(prefix);
    }
  }

  // Extract top-level directories (first path component)
  const topLevelDirs = new Set<string>();
  for (const { path } of paths) {
    const parts = path.split('/').filter(Boolean);
    if (parts.length > 1) {
      const dir = parts[0];
      if (dir) topLevelDirs.add(dir);
    }
  }

  // Filter out directories that already have context (exact or parent)
  const missing: string[] = [];
  for (const dir of topLevelDirs) {
    let hasContext = false;

    // Check if this dir or any parent has context
    for (const prefix of contextPrefixes) {
      if (prefix === '' || prefix === dir || dir.startsWith(prefix + '/')) {
        hasContext = true;
        break;
      }
    }

    if (!hasContext) {
      missing.push(dir);
    }
  }

  return missing.sort();
}

// =============================================================================
// FTS Search
// =============================================================================

function sanitizeFTS5Term(term: string): string {
  return term.replace(/[^\p{L}\p{N}']/gu, '').toLowerCase();
}

function buildFTS5Query(query: string): string | null {
  const terms = query.split(/\s+/)
    .map(t => sanitizeFTS5Term(t))
    .filter(t => t.length > 0);
  if (terms.length === 0) return null;
  if (terms.length === 1) return `"${terms[0]}"*`;
  return terms.map(t => `"${t}"*`).join(' AND ');
}

export function searchFTS(db: Database, query: string, limit: number = 20, collectionId?: number): SearchResult[] {
  const ftsQuery = buildFTS5Query(query);
  if (!ftsQuery) return [];

  let sql = `
    SELECT
      'clawmem://' || d.collection || '/' || d.path as filepath,
      d.collection || '/' || d.path as display_path,
      d.title,
      content.doc as body,
      d.hash,
      bm25(documents_fts, 10.0, 1.0) as bm25_score
    FROM documents_fts f
    JOIN documents d ON d.id = f.rowid
    JOIN content ON content.hash = d.hash
    WHERE documents_fts MATCH ? AND d.active = 1
  `;
  const params: (string | number)[] = [ftsQuery];

  if (collectionId) {
    // Note: collectionId is a legacy parameter that should be phased out
    // Collections are now managed in YAML. For now, we interpret it as a collection name filter.
    // This code path is likely unused as collection filtering should be done at CLI level.
    sql += ` AND d.collection = ?`;
    params.push(String(collectionId));
  }

  // bm25 lower is better; sort ascending.
  sql += ` ORDER BY bm25_score ASC LIMIT ?`;
  params.push(limit);

  const rows = db.prepare(sql).all(...params) as { filepath: string; display_path: string; title: string; body: string; hash: string; bm25_score: number }[];
  return rows.map(row => {
    const collectionName = row.filepath.split('//')[1]?.split('/')[0] || "";
    // Convert bm25 (lower is better) into a stable (0..1] score where higher is better.
    // Avoid per-query normalization so "strong signal" heuristics can work.
    const score = 1 / (1 + Math.max(0, row.bm25_score));
    return {
      filepath: row.filepath,
      displayPath: row.display_path,
      title: row.title,
      hash: row.hash,
      docid: getDocid(row.hash),
      collectionName,
      modifiedAt: "",  // Not available in FTS query
      bodyLength: row.body.length,
      body: row.body,
      context: getContextForFile(db, row.filepath),
      score,
      source: "fts" as const,
    };
  });
}

// =============================================================================
// Vector Search
// =============================================================================

export async function searchVec(db: Database, query: string, model: string, limit: number = 20, collectionId?: number): Promise<SearchResult[]> {
  const tableExists = db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
  if (!tableExists) return [];

  const embedding = await getEmbedding(query, model, true);
  if (!embedding) return [];

  // IMPORTANT: We use a two-step query approach here because sqlite-vec virtual tables
  // hang indefinitely when combined with JOINs in the same query. Do NOT try to
  // "optimize" this by combining into a single query with JOINs - it will break.
  // See: https://github.com/tobi/qmd/pull/23

  // Step 1: Get vector matches from sqlite-vec (no JOINs allowed)
  const vecResults = db.prepare(`
    SELECT hash_seq, distance
    FROM vectors_vec
    WHERE embedding MATCH ? AND k = ?
  `).all(new Float32Array(embedding), limit * 3) as { hash_seq: string; distance: number }[];

  if (vecResults.length === 0) return [];

  // Step 2: Get chunk info and document data
  const hashSeqs = vecResults.map(r => r.hash_seq);
  const distanceMap = new Map(vecResults.map(r => [r.hash_seq, r.distance]));

  // Build query for document lookup (includes fragment metadata)
  const placeholders = hashSeqs.map(() => '?').join(',');
  let docSql = `
    SELECT
      cv.hash || '_' || cv.seq as hash_seq,
      cv.hash,
      cv.pos,
      cv.fragment_type,
      cv.fragment_label,
      'clawmem://' || d.collection || '/' || d.path as filepath,
      d.collection || '/' || d.path as display_path,
      d.title,
      content.doc as body
    FROM content_vectors cv
    JOIN documents d ON d.hash = cv.hash AND d.active = 1
    JOIN content ON content.hash = d.hash
    WHERE cv.hash || '_' || cv.seq IN (${placeholders})
  `;
  const params: string[] = [...hashSeqs];

  if (collectionId) {
    docSql += ` AND d.collection = ?`;
    params.push(String(collectionId));
  }

  const docRows = db.prepare(docSql).all(...params) as {
    hash_seq: string; hash: string; pos: number; filepath: string;
    display_path: string; title: string; body: string;
    fragment_type: string | null; fragment_label: string | null;
  }[];

  // Combine with distances and dedupe by filepath (keep best-scoring fragment per doc)
  const seen = new Map<string, { row: typeof docRows[0]; bestDist: number }>();
  for (const row of docRows) {
    const distance = distanceMap.get(row.hash_seq) ?? 1;
    const existing = seen.get(row.filepath);
    if (!existing || distance < existing.bestDist) {
      seen.set(row.filepath, { row, bestDist: distance });
    }
  }

  return Array.from(seen.values())
    .sort((a, b) => a.bestDist - b.bestDist)
    .slice(0, limit)
    .map(({ row, bestDist }) => {
      const collectionName = row.filepath.split('//')[1]?.split('/')[0] || "";
      return {
        filepath: row.filepath,
        displayPath: row.display_path,
        title: row.title,
        hash: row.hash,
        docid: getDocid(row.hash),
        collectionName,
        modifiedAt: "",  // Not available in vec query
        bodyLength: row.body.length,
        body: row.body,
        context: getContextForFile(db, row.filepath),
        score: 1 - bestDist,  // Cosine similarity = 1 - cosine distance
        source: "vec" as const,
        chunkPos: row.pos,
        fragmentType: row.fragment_type ?? undefined,
        fragmentLabel: row.fragment_label ?? undefined,
      };
    });
}

// =============================================================================
// Embeddings
// =============================================================================

async function getEmbedding(text: string, model: string, isQuery: boolean): Promise<number[] | null> {
  const llm = getDefaultLlamaCpp();
  // Format text using the appropriate prompt template
  const formattedText = isQuery ? formatQueryForEmbedding(text) : formatDocForEmbedding(text);
  const result = await llm.embed(formattedText, { model, isQuery });
  return result?.embedding || null;
}

/**
 * Get all unique content hashes that need embeddings (from active documents).
 * Returns hash, document body, and a sample path for display purposes.
 */
export function getHashesForEmbedding(db: Database): { hash: string; body: string; path: string }[] {
  return db.prepare(`
    SELECT d.hash, c.doc as body, MIN(d.path) as path
    FROM documents d
    JOIN content c ON d.hash = c.hash
    LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
    WHERE d.active = 1 AND v.hash IS NULL
    GROUP BY d.hash
  `).all() as { hash: string; body: string; path: string }[];
}

/**
 * Get all unique content hashes that need fragment-level embeddings.
 * Returns hashes that have no content_vectors row with fragment_type set.
 */
export function getHashesNeedingFragments(db: Database): { hash: string; body: string; path: string; title: string }[] {
  return db.prepare(`
    SELECT d.hash, c.doc as body, MIN(d.path) as path, MIN(d.title) as title
    FROM documents d
    JOIN content c ON d.hash = c.hash
    LEFT JOIN content_vectors v ON d.hash = v.hash AND v.fragment_type IS NOT NULL
    WHERE d.active = 1 AND v.hash IS NULL
    GROUP BY d.hash
  `).all() as { hash: string; body: string; path: string; title: string }[];
}

/**
 * Clear all embeddings from the database (force re-index).
 * Deletes all rows from content_vectors and drops the vectors_vec table.
 */
export function clearAllEmbeddings(db: Database): void {
  db.exec(`DELETE FROM content_vectors`);
  db.exec(`DROP TABLE IF EXISTS vectors_vec`);
}

/**
 * Insert a single embedding into both content_vectors and vectors_vec tables.
 * The hash_seq key is formatted as "hash_seq" for the vectors_vec table.
 */
export function insertEmbedding(
  db: Database,
  hash: string,
  seq: number,
  pos: number,
  embedding: Float32Array,
  model: string,
  embeddedAt: string,
  fragmentType?: string,
  fragmentLabel?: string
): void {
  const hashSeq = `${hash}_${seq}`;
  const insertVecStmt = db.prepare(`INSERT OR REPLACE INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)`);
  const insertContentVectorStmt = db.prepare(
    `INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embedded_at, fragment_type, fragment_label) VALUES (?, ?, ?, ?, ?, ?, ?)`
  );

  insertVecStmt.run(hashSeq, embedding);
  insertContentVectorStmt.run(hash, seq, pos, model, embeddedAt, fragmentType ?? null, fragmentLabel ?? null);
}

// =============================================================================
// Query expansion
// =============================================================================

export async function expandQuery(query: string, model: string = DEFAULT_QUERY_MODEL, db: Database): Promise<string[]> {
  // Check cache first
  const cacheKey = getCacheKey("expandQuery", { query, model });
  const cached = getCachedResult(db, cacheKey);
  if (cached) {
    const lines = cached.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    return [query, ...lines.slice(0, 2)];
  }

  const llm = getDefaultLlamaCpp();
  // Note: LlamaCpp uses hardcoded model, model parameter is ignored
  const results = await llm.expandQuery(query);
  const queryTexts = results.map(r => r.text);

  // Cache the expanded queries (excluding original)
  const expandedOnly = queryTexts.filter(t => t !== query);
  if (expandedOnly.length > 0) {
    setCachedResult(db, cacheKey, expandedOnly.join('\n'));
  }

  return Array.from(new Set([query, ...queryTexts]));
}

// =============================================================================
// Reranking
// =============================================================================

export async function rerank(query: string, documents: { file: string; text: string }[], model: string = DEFAULT_RERANK_MODEL, db: Database): Promise<{ file: string; score: number }[]> {
  const cachedResults: Map<string, number> = new Map();
  const uncachedDocs: RerankDocument[] = [];

  // Check cache for each document
  for (const doc of documents) {
    const cacheKey = getCacheKey("rerank", { query, file: doc.file, model });
    const cached = getCachedResult(db, cacheKey);
    if (cached !== null) {
      cachedResults.set(doc.file, parseFloat(cached));
    } else {
      uncachedDocs.push({ file: doc.file, text: doc.text });
    }
  }

  // Rerank uncached documents (remote GPU preferred, local node-llama-cpp fallback)
  if (uncachedDocs.length > 0) {
    const rerankUrl = Bun.env.CLAWMEM_RERANK_URL;
    let scored = false;

    // Try remote GPU reranker first
    // Truncate to ~400 chars per doc to fit within server's 512-token context
    // (query + document must fit in one pair; ~2 chars/token for mixed content)
    if (rerankUrl) {
      try {
        const resp = await fetch(`${rerankUrl}/v1/rerank`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            documents: uncachedDocs.map(d => d.text.slice(0, 400)),
          }),
        });
        if (resp.ok) {
          const data = await resp.json() as { results: { index: number; relevance_score: number }[] };
          for (const r of data.results) {
            const doc = uncachedDocs[r.index]!;
            const cacheKey = getCacheKey("rerank", { query, file: doc.file, model });
            setCachedResult(db, cacheKey, r.relevance_score.toString());
            cachedResults.set(doc.file, r.relevance_score);
          }
          scored = true;
        }
      } catch {
        // Remote failed, fall through to local
      }
    }

    // Fallback to local node-llama-cpp
    if (!scored) {
      const llm = getDefaultLlamaCpp();
      const rerankResult = await llm.rerank(query, uncachedDocs, { model });
      for (const result of rerankResult.results) {
        const cacheKey = getCacheKey("rerank", { query, file: result.file, model });
        setCachedResult(db, cacheKey, result.score.toString());
        cachedResults.set(result.file, result.score);
      }
    }
  }

  // Return all results sorted by score
  return documents
    .map(doc => ({ file: doc.file, score: cachedResults.get(doc.file) || 0 }))
    .sort((a, b) => b.score - a.score);
}

// =============================================================================
// Reciprocal Rank Fusion
// =============================================================================

export function reciprocalRankFusion(
  resultLists: RankedResult[][],
  weights: number[] = [],
  k: number = 60
): RankedResult[] {
  const scores = new Map<string, { result: RankedResult; rrfScore: number; topRank: number }>();

  for (let listIdx = 0; listIdx < resultLists.length; listIdx++) {
    const list = resultLists[listIdx];
    if (!list) continue;
    const weight = weights[listIdx] ?? 1.0;

    for (let rank = 0; rank < list.length; rank++) {
      const result = list[rank];
      if (!result) continue;
      const rrfContribution = weight / (k + rank + 1);
      const existing = scores.get(result.file);

      if (existing) {
        existing.rrfScore += rrfContribution;
        existing.topRank = Math.min(existing.topRank, rank);
      } else {
        scores.set(result.file, {
          result,
          rrfScore: rrfContribution,
          topRank: rank,
        });
      }
    }
  }

  // Top-rank bonus
  for (const entry of scores.values()) {
    if (entry.topRank === 0) {
      entry.rrfScore += 0.05;
    } else if (entry.topRank <= 2) {
      entry.rrfScore += 0.02;
    }
  }

  return Array.from(scores.values())
    .sort((a, b) => b.rrfScore - a.rrfScore)
    .map(e => ({ ...e.result, score: e.rrfScore }));
}

// =============================================================================
// Document retrieval
// =============================================================================

type DbDocRow = {
  virtual_path: string;
  display_path: string;
  title: string;
  hash: string;
  collection: string;
  path: string;
  modified_at: string;
  body_length: number;
  body?: string;
};

/**
 * Find a document by filename/path, docid (#hash), or with fuzzy matching.
 * Returns document metadata without body by default.
 *
 * Supports:
 * - Virtual paths: clawmem://collection/path/to/file.md
 * - Absolute paths: /path/to/file.md
 * - Relative paths: path/to/file.md
 * - Short docid: #abc123 (first 6 chars of hash)
 */
export function findDocument(db: Database, filename: string, options: { includeBody?: boolean } = {}): DocumentResult | DocumentNotFound {
  let filepath = filename;
  const colonMatch = filepath.match(/:(\d+)$/);
  if (colonMatch) {
    filepath = filepath.slice(0, -colonMatch[0].length);
  }

  // Check if this is a docid lookup (#hash or just 6-char hex)
  if (filepath.startsWith('#') || /^[a-f0-9]{6}$/i.test(filepath)) {
    const docidMatch = findDocumentByDocid(db, filepath);
    if (docidMatch) {
      filepath = docidMatch.filepath;
    } else {
      return { error: "not_found", query: filename, similarFiles: [] };
    }
  }

  if (filepath.startsWith('~/')) {
    filepath = homedir() + filepath.slice(1);
  }

  const bodyCol = options.includeBody ? `, content.doc as body` : ``;

  // Build computed columns
  // Note: absoluteFilepath is computed from YAML collections after query
  const selectCols = `
    'clawmem://' || d.collection || '/' || d.path as virtual_path,
    d.collection || '/' || d.path as display_path,
    d.title,
    d.hash,
    d.collection,
    d.modified_at,
    LENGTH(content.doc) as body_length
    ${bodyCol}
  `;

  // Try to match by virtual path first
  let doc = db.prepare(`
    SELECT ${selectCols}
    FROM documents d
    JOIN content ON content.hash = d.hash
    WHERE 'clawmem://' || d.collection || '/' || d.path = ? AND d.active = 1
  `).get(filepath) as DbDocRow | null;

  // Try fuzzy match by virtual path
  if (!doc) {
    doc = db.prepare(`
      SELECT ${selectCols}
      FROM documents d
      JOIN content ON content.hash = d.hash
      WHERE 'clawmem://' || d.collection || '/' || d.path LIKE ? AND d.active = 1
      LIMIT 1
    `).get(`%${filepath}`) as DbDocRow | null;
  }

  // Try to match by absolute path (requires looking up collection paths from YAML)
  if (!doc && !filepath.startsWith('clawmem://')) {
    const collections = collectionsListCollections();
    for (const coll of collections) {
      let relativePath: string | null = null;

      // If filepath is absolute and starts with collection path, extract relative part
      if (filepath.startsWith(coll.path + '/')) {
        relativePath = filepath.slice(coll.path.length + 1);
      }
      // Otherwise treat filepath as relative to collection
      else if (!filepath.startsWith('/')) {
        relativePath = filepath;
      }

      if (relativePath) {
        doc = db.prepare(`
          SELECT ${selectCols}
          FROM documents d
          JOIN content ON content.hash = d.hash
          WHERE d.collection = ? AND d.path = ? AND d.active = 1
        `).get(coll.name, relativePath) as DbDocRow | null;
        if (doc) break;
      }
    }
  }

  if (!doc) {
    const similar = findSimilarFiles(db, filepath, 5, 5);
    return { error: "not_found", query: filename, similarFiles: similar };
  }

  // Get context using virtual path
  const virtualPath = doc.virtual_path || `clawmem://${doc.collection}/${doc.display_path}`;
  const context = getContextForFile(db, virtualPath);

  return {
    filepath: virtualPath,
    displayPath: doc.display_path,
    title: doc.title,
    context,
    hash: doc.hash,
    docid: getDocid(doc.hash),
    collectionName: doc.collection,
    modifiedAt: doc.modified_at,
    bodyLength: doc.body_length,
    ...(options.includeBody && doc.body !== undefined && { body: doc.body }),
  };
}

/**
 * Get the body content for a document
 * Optionally slice by line range
 */
export function getDocumentBody(db: Database, doc: DocumentResult | { filepath: string }, fromLine?: number, maxLines?: number): string | null {
  const filepath = doc.filepath;

  // Try to resolve document by filepath (absolute or virtual)
  let row: { body: string } | null = null;

  // Try virtual path first
  if (filepath.startsWith('clawmem://')) {
    row = db.prepare(`
      SELECT content.doc as body
      FROM documents d
      JOIN content ON content.hash = d.hash
      WHERE 'clawmem://' || d.collection || '/' || d.path = ? AND d.active = 1
    `).get(filepath) as { body: string } | null;
  }

  // Try absolute path by looking up in YAML collections
  if (!row) {
    const collections = collectionsListCollections();
    for (const coll of collections) {
      if (filepath.startsWith(coll.path + '/')) {
        const relativePath = filepath.slice(coll.path.length + 1);
        row = db.prepare(`
          SELECT content.doc as body
          FROM documents d
          JOIN content ON content.hash = d.hash
          WHERE d.collection = ? AND d.path = ? AND d.active = 1
        `).get(coll.name, relativePath) as { body: string } | null;
        if (row) break;
      }
    }
  }

  // Try collection/path format (e.g., "_clawmem/decisions/foo.md")
  if (!row) {
    const slashIdx = filepath.indexOf('/');
    if (slashIdx > 0) {
      const collection = filepath.slice(0, slashIdx);
      const path = filepath.slice(slashIdx + 1);
      row = db.prepare(`
        SELECT content.doc as body
        FROM documents d
        JOIN content ON content.hash = d.hash
        WHERE d.collection = ? AND d.path = ? AND d.active = 1
      `).get(collection, path) as { body: string } | null;
    }
  }

  if (!row) return null;

  let body = row.body;
  if (fromLine !== undefined || maxLines !== undefined) {
    const lines = body.split('\n');
    const start = (fromLine || 1) - 1;
    const end = maxLines !== undefined ? start + maxLines : lines.length;
    body = lines.slice(start, end).join('\n');
  }

  return body;
}

/**
 * Find multiple documents by glob pattern or comma-separated list
 * Returns documents without body by default (use getDocumentBody to load)
 */
export function findDocuments(
  db: Database,
  pattern: string,
  options: { includeBody?: boolean; maxBytes?: number } = {}
): { docs: MultiGetResult[]; errors: string[] } {
  const isCommaSeparated = pattern.includes(',') && !pattern.includes('*') && !pattern.includes('?');
  const errors: string[] = [];
  const maxBytes = options.maxBytes ?? DEFAULT_MULTI_GET_MAX_BYTES;

  const bodyCol = options.includeBody ? `, content.doc as body` : ``;
  const selectCols = `
    'clawmem://' || d.collection || '/' || d.path as virtual_path,
    d.collection || '/' || d.path as display_path,
    d.title,
    d.hash,
    d.collection,
    d.modified_at,
    LENGTH(content.doc) as body_length
    ${bodyCol}
  `;

  let fileRows: DbDocRow[];

  if (isCommaSeparated) {
    const names = pattern.split(',').map(s => s.trim()).filter(Boolean);
    fileRows = [];
    for (const name of names) {
      let doc = db.prepare(`
        SELECT ${selectCols}
        FROM documents d
        JOIN content ON content.hash = d.hash
        WHERE 'clawmem://' || d.collection || '/' || d.path = ? AND d.active = 1
      `).get(name) as DbDocRow | null;
      if (!doc) {
        doc = db.prepare(`
          SELECT ${selectCols}
          FROM documents d
          JOIN content ON content.hash = d.hash
          WHERE 'clawmem://' || d.collection || '/' || d.path LIKE ? AND d.active = 1
          LIMIT 1
        `).get(`%${name}`) as DbDocRow | null;
      }
      if (doc) {
        fileRows.push(doc);
      } else {
        const similar = findSimilarFiles(db, name, 5, 3);
        let msg = `File not found: ${name}`;
        if (similar.length > 0) {
          msg += ` (did you mean: ${similar.join(', ')}?)`;
        }
        errors.push(msg);
      }
    }
  } else {
    // Glob pattern match
    const matched = matchFilesByGlob(db, pattern);
    if (matched.length === 0) {
      errors.push(`No files matched pattern: ${pattern}`);
      return { docs: [], errors };
    }
    const virtualPaths = matched.map(m => m.filepath);
    const placeholders = virtualPaths.map(() => '?').join(',');
    fileRows = db.prepare(`
      SELECT ${selectCols}
      FROM documents d
      JOIN content ON content.hash = d.hash
      WHERE 'clawmem://' || d.collection || '/' || d.path IN (${placeholders}) AND d.active = 1
    `).all(...virtualPaths) as DbDocRow[];
  }

  const results: MultiGetResult[] = [];

  for (const row of fileRows) {
    // Get context using virtual path
    const virtualPath = row.virtual_path || `clawmem://${row.collection}/${row.display_path}`;
    const context = getContextForFile(db, virtualPath);

    if (row.body_length > maxBytes) {
      results.push({
        doc: { filepath: virtualPath, displayPath: row.display_path },
        skipped: true,
        skipReason: `File too large (${Math.round(row.body_length / 1024)}KB > ${Math.round(maxBytes / 1024)}KB)`,
      });
      continue;
    }

    results.push({
      doc: {
        filepath: virtualPath,
        displayPath: row.display_path,
        title: row.title || row.display_path.split('/').pop() || row.display_path,
        context,
        hash: row.hash,
        docid: getDocid(row.hash),
        collectionName: row.collection,
        modifiedAt: row.modified_at,
        bodyLength: row.body_length,
        ...(options.includeBody && row.body !== undefined && { body: row.body }),
      },
      skipped: false,
    });
  }

  return { docs: results, errors };
}

// =============================================================================
// Status
// =============================================================================

export function getStatus(db: Database): IndexStatus {
  // Load collections from YAML
  const yamlCollections = collectionsListCollections();

  // Get document counts and last update times for each collection
  const collections = yamlCollections.map(col => {
    const stats = db.prepare(`
      SELECT
        COUNT(*) as active_count,
        MAX(modified_at) as last_doc_update
      FROM documents
      WHERE collection = ? AND active = 1
    `).get(col.name) as { active_count: number; last_doc_update: string | null };

    return {
      name: col.name,
      path: col.path,
      pattern: col.pattern,
      documents: stats.active_count,
      lastUpdated: stats.last_doc_update || new Date().toISOString(),
    };
  });

  // Sort by last update time (most recent first)
  collections.sort((a, b) => {
    if (!a.lastUpdated) return 1;
    if (!b.lastUpdated) return -1;
    return new Date(b.lastUpdated).getTime() - new Date(a.lastUpdated).getTime();
  });

  const totalDocs = (db.prepare(`SELECT COUNT(*) as c FROM documents WHERE active = 1`).get() as { c: number }).c;
  const needsEmbedding = getHashesNeedingEmbedding(db);
  const hasVectors = !!db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();

  return {
    totalDocuments: totalDocs,
    needsEmbedding,
    hasVectorIndex: hasVectors,
    collections,
  };
}

// =============================================================================
// Snippet extraction
// =============================================================================

export type SnippetResult = {
  line: number;           // 1-indexed line number of best match
  snippet: string;        // The snippet text with diff-style header
  linesBefore: number;    // Lines in document before snippet
  linesAfter: number;     // Lines in document after snippet
  snippetLines: number;   // Number of lines in snippet
};

export function extractSnippet(body: string, query: string, maxLen = 500, chunkPos?: number): SnippetResult {
  const totalLines = body.split('\n').length;
  let searchBody = body;
  let lineOffset = 0;

  if (chunkPos && chunkPos > 0) {
    const contextStart = Math.max(0, chunkPos - 100);
    const contextEnd = Math.min(body.length, chunkPos + maxLen + 100);
    searchBody = body.slice(contextStart, contextEnd);
    if (contextStart > 0) {
      lineOffset = body.slice(0, contextStart).split('\n').length - 1;
    }
  }

  const lines = searchBody.split('\n');
  const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 0);
  let bestLine = 0, bestScore = -1;

  for (let i = 0; i < lines.length; i++) {
    const lineLower = (lines[i] ?? "").toLowerCase();
    let score = 0;
    for (const term of queryTerms) {
      if (lineLower.includes(term)) score++;
    }
    if (score > bestScore) {
      bestScore = score;
      bestLine = i;
    }
  }

  const start = Math.max(0, bestLine - 1);
  const end = Math.min(lines.length, bestLine + 3);
  const snippetLines = lines.slice(start, end);
  let snippetText = snippetLines.join('\n');

  // If we focused on a chunk window and it produced an empty/whitespace-only snippet,
  // fall back to a full-document snippet so we always show something useful.
  if (chunkPos && chunkPos > 0 && snippetText.trim().length === 0) {
    return extractSnippet(body, query, maxLen, undefined);
  }

  if (snippetText.length > maxLen) snippetText = snippetText.substring(0, maxLen - 3) + "...";

  const absoluteStart = lineOffset + start + 1; // 1-indexed
  const snippetLineCount = snippetLines.length;
  const linesBefore = absoluteStart - 1;
  const linesAfter = totalLines - (absoluteStart + snippetLineCount - 1);

  // Format with diff-style header: @@ -start,count @@ (linesBefore before, linesAfter after)
  const header = `@@ -${absoluteStart},${snippetLineCount} @@ (${linesBefore} before, ${linesAfter} after)`;
  const snippet = `${header}\n${snippetText}`;

  return {
    line: lineOffset + bestLine + 1,
    snippet,
    linesBefore,
    linesAfter,
    snippetLines: snippetLineCount,
  };
}

// =============================================================================
// SAME: Session Tracking
// =============================================================================

function insertSessionFn(db: Database, sessionId: string, startedAt: string, machine?: string): void {
  db.prepare(`
    INSERT OR IGNORE INTO session_log (session_id, started_at, machine)
    VALUES (?, ?, ?)
  `).run(sessionId, startedAt, machine ?? null);
}

function updateSessionFn(db: Database, sessionId: string, updates: { endedAt?: string; handoffPath?: string; filesChanged?: string[]; summary?: string }): void {
  const sets: string[] = [];
  const vals: (string | null)[] = [];
  if (updates.endedAt !== undefined) { sets.push("ended_at = ?"); vals.push(updates.endedAt); }
  if (updates.handoffPath !== undefined) { sets.push("handoff_path = ?"); vals.push(updates.handoffPath); }
  if (updates.filesChanged !== undefined) { sets.push("files_changed = ?"); vals.push(JSON.stringify(updates.filesChanged)); }
  if (updates.summary !== undefined) { sets.push("summary = ?"); vals.push(updates.summary); }
  if (sets.length === 0) return;
  vals.push(sessionId);
  db.prepare(`UPDATE session_log SET ${sets.join(", ")} WHERE session_id = ?`).run(...vals);
}

function getSessionFn(db: Database, sessionId: string): SessionRecord | null {
  const row = db.prepare(`SELECT * FROM session_log WHERE session_id = ?`).get(sessionId) as any;
  if (!row) return null;
  return {
    sessionId: row.session_id,
    startedAt: row.started_at,
    endedAt: row.ended_at,
    handoffPath: row.handoff_path,
    machine: row.machine,
    filesChanged: row.files_changed ? JSON.parse(row.files_changed) : [],
    summary: row.summary,
  };
}

function getRecentSessionsFn(db: Database, limit: number): SessionRecord[] {
  const rows = db.prepare(`SELECT * FROM session_log ORDER BY started_at DESC LIMIT ?`).all(limit) as any[];
  return rows.map(row => ({
    sessionId: row.session_id,
    startedAt: row.started_at,
    endedAt: row.ended_at,
    handoffPath: row.handoff_path,
    machine: row.machine,
    filesChanged: row.files_changed ? JSON.parse(row.files_changed) : [],
    summary: row.summary,
  }));
}

// =============================================================================
// SAME: Context Usage Tracking
// =============================================================================

function insertUsageFn(db: Database, usage: UsageRecord): void {
  db.prepare(`
    INSERT INTO context_usage (session_id, timestamp, hook_name, injected_paths, estimated_tokens, was_referenced)
    VALUES (?, ?, ?, ?, ?, ?)
  `).run(usage.sessionId, usage.timestamp, usage.hookName, JSON.stringify(usage.injectedPaths), usage.estimatedTokens, usage.wasReferenced);
}

function getUsageForSessionFn(db: Database, sessionId: string): UsageRow[] {
  return db.prepare(`
    SELECT id, session_id AS sessionId, timestamp, hook_name AS hookName,
           injected_paths AS injectedPaths, estimated_tokens AS estimatedTokens,
           was_referenced AS wasReferenced
    FROM context_usage WHERE session_id = ? ORDER BY timestamp
  `).all(sessionId) as UsageRow[];
}

function markUsageReferencedFn(db: Database, id: number): void {
  db.prepare(`UPDATE context_usage SET was_referenced = 1 WHERE id = ?`).run(id);
}

// =============================================================================
// SAME: Document Metadata Operations
// =============================================================================

function updateDocumentMetaFn(db: Database, docId: number, meta: { domain?: string; workstream?: string; tags?: string; content_type?: string; review_by?: string; confidence?: number }): void {
  const sets: string[] = [];
  const vals: (string | number | null)[] = [];
  if (meta.domain !== undefined) { sets.push("domain = ?"); vals.push(meta.domain); }
  if (meta.workstream !== undefined) { sets.push("workstream = ?"); vals.push(meta.workstream); }
  if (meta.tags !== undefined) { sets.push("tags = ?"); vals.push(meta.tags); }
  if (meta.content_type !== undefined) { sets.push("content_type = ?"); vals.push(meta.content_type); }
  if (meta.review_by !== undefined) { sets.push("review_by = ?"); vals.push(meta.review_by); }
  if (meta.confidence !== undefined) { sets.push("confidence = ?"); vals.push(meta.confidence); }
  if (sets.length === 0) return;
  vals.push(docId);
  db.prepare(`UPDATE documents SET ${sets.join(", ")} WHERE id = ?`).run(...vals);
}

function incrementAccessCountFn(db: Database, paths: string[]): void {
  if (paths.length === 0) return;
  const placeholders = paths.map(() => "?").join(",");
  db.prepare(`
    UPDATE documents SET access_count = access_count + 1
    WHERE active = 1 AND (collection || '/' || path) IN (${placeholders})
  `).run(...paths);
}

function getDocumentsByTypeFn(db: Database, contentType: string, limit: number = 10): DocumentRow[] {
  return db.prepare(`
    SELECT d.id, d.collection, d.path, d.title, d.hash, d.modified_at as modifiedAt,
           d.domain, d.workstream, d.tags, d.content_type as contentType,
           d.review_by as reviewBy, d.confidence, d.access_count as accessCount,
           LENGTH(c.doc) as bodyLength
    FROM documents d
    JOIN content c ON c.hash = d.hash
    WHERE d.active = 1 AND d.content_type = ?
    ORDER BY d.modified_at DESC
    LIMIT ?
  `).all(contentType, limit) as DocumentRow[];
}

function updateObservationFieldsFn(
  db: Database,
  docPath: string,
  collectionName: string,
  fields: { observation_type?: string; facts?: string; narrative?: string; concepts?: string; files_read?: string; files_modified?: string }
): void {
  const sets: string[] = [];
  const vals: (string | null)[] = [];
  if (fields.observation_type !== undefined) { sets.push("observation_type = ?"); vals.push(fields.observation_type); }
  if (fields.facts !== undefined) { sets.push("facts = ?"); vals.push(fields.facts); }
  if (fields.narrative !== undefined) { sets.push("narrative = ?"); vals.push(fields.narrative); }
  if (fields.concepts !== undefined) { sets.push("concepts = ?"); vals.push(fields.concepts); }
  if (fields.files_read !== undefined) { sets.push("files_read = ?"); vals.push(fields.files_read); }
  if (fields.files_modified !== undefined) { sets.push("files_modified = ?"); vals.push(fields.files_modified); }
  if (sets.length === 0) return;
  vals.push(collectionName, docPath);
  db.prepare(`UPDATE documents SET ${sets.join(", ")} WHERE collection = ? AND path = ? AND active = 1`).run(...vals);
}

function getStaleDocumentsFn(db: Database, beforeDate: string): DocumentRow[] {
  return db.prepare(`
    SELECT d.id, d.collection, d.path, d.title, d.hash, d.modified_at as modifiedAt,
           d.domain, d.workstream, d.tags, d.content_type as contentType,
           d.review_by as reviewBy, d.confidence, d.access_count as accessCount,
           LENGTH(c.doc) as bodyLength
    FROM documents d
    JOIN content c ON c.hash = d.hash
    WHERE d.active = 1 AND d.review_by IS NOT NULL AND d.review_by != '' AND d.review_by <= ?
    ORDER BY d.review_by ASC
  `).all(beforeDate) as DocumentRow[];
}

// =============================================================================
// Beads Integration
// =============================================================================

/**
 * Sync Beads issues from .beads/beads.jsonl into ClawMem.
 * Returns count of synced and newly created issues.
 */
export async function syncBeadsIssues(
  db: Database,
  beadsJsonlPath: string
): Promise<{ synced: number; created: number; newDocIds: number[] }> {
  const issues = parseBeadsJsonl(beadsJsonlPath);
  let synced = 0;
  let created = 0;
  const newDocIds: number[] = [];

  for (const issue of issues) {
    // Create markdown document for issue
    const docPath = `_clawmem/beads/${issue.id}.md`;
    const docBody = formatBeadsIssueAsMarkdown(issue);
    const hash = await hashContent(docBody);

    // Check if document exists
    const existingDoc = findActiveDocument(db, 'beads', docPath);

    if (existingDoc) {
      // Update document content if it changed
      if (existingDoc.hash !== hash) {
        insertContent(db, hash, docBody, new Date().toISOString());
        db.prepare(`UPDATE documents SET hash = ?, modified_at = ? WHERE id = ?`)
          .run(hash, new Date().toISOString(), existingDoc.id);
      }

      // Update beads_issues table metadata
      db.prepare(`
        UPDATE beads_issues
        SET status = ?, priority = ?, assignee = ?, last_synced_at = ?
        WHERE beads_id = ?
      `).run(
        issue.status,
        issue.priority,
        issue.assignee || null,
        new Date().toISOString(),
        issue.id
      );
      synced++;
    } else {
      // Insert content first
      insertContent(db, hash, docBody, issue.created_at);

      // Insert document
      insertDocument(
        db,
        'beads',
        docPath,
        issue.title,
        hash,
        issue.created_at,
        issue.created_at
      );

      // Get the newly inserted document ID
      const newDoc = findActiveDocument(db, 'beads', docPath);
      if (!newDoc) {
        console.warn(`[beads] Failed to insert document for ${issue.id}`);
        continue;
      }

      // Insert beads metadata
      db.prepare(`
        INSERT INTO beads_issues (
          beads_id, doc_id, issue_type, status, priority, tags,
          assignee, parent_id, created_at, closed_at, last_synced_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        issue.id,
        newDoc.id,
        issue.type,
        issue.status,
        issue.priority,
        JSON.stringify(issue.tags || []),
        issue.assignee || null,
        issue.parent || null,
        issue.created_at,
        issue.closed_at || null,
        new Date().toISOString()
      );

      newDocIds.push(newDoc.id);
      created++;
    }
  }

  // Second pass: insert dependencies after all issues exist (avoids FK violations for forward references)
  for (const issue of issues) {
    if (issue.blocks && issue.blocks.length > 0) {
      for (const blockedId of issue.blocks) {
        db.prepare(`
          INSERT OR IGNORE INTO beads_dependencies (source_id, target_id, dep_type, created_at)
          VALUES (?, ?, 'blocks', ?)
        `).run(blockedId, issue.id, new Date().toISOString());
      }
    }
  }

  // Third pass: bridge beads_dependencies → memory_relations for MAGMA graph traversal
  // Mapping: blocks→causal, discovered-from→supporting, relates-to→semantic
  const depTypeMap: Record<string, string> = {
    'blocks': 'causal',
    'discovered-from': 'supporting',
    'relates-to': 'semantic',
    'waits-for': 'causal',
  };

  const allDeps = db.prepare(`SELECT source_id, target_id, dep_type FROM beads_dependencies`).all() as {
    source_id: string; target_id: string; dep_type: string;
  }[];

  for (const dep of allDeps) {
    const relationType = depTypeMap[dep.dep_type] || 'semantic';

    // Resolve beads_id → doc_id via beads_issues table
    const sourceRow = db.prepare(`SELECT doc_id FROM beads_issues WHERE beads_id = ?`).get(dep.source_id) as { doc_id: number } | undefined;
    const targetRow = db.prepare(`SELECT doc_id FROM beads_issues WHERE beads_id = ?`).get(dep.target_id) as { doc_id: number } | undefined;

    if (sourceRow && targetRow) {
      db.prepare(`
        INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type, weight, metadata, created_at)
        VALUES (?, ?, ?, 1.0, ?, ?)
      `).run(
        sourceRow.doc_id,
        targetRow.doc_id,
        relationType,
        JSON.stringify({ origin: 'beads', dep_type: dep.dep_type }),
        new Date().toISOString()
      );
    }
  }

  return { synced, created, newDocIds };
}

/**
 * Export for MCP tool registration.
 */
export { detectBeadsProject };

// =============================================================================
// MAGMA Graph Building
// =============================================================================

/**
 * Build temporal backbone - connect documents in chronological order.
 * Returns number of edges created.
 */
export function buildTemporalBackbone(db: Database): number {
  // Get all documents ordered by creation time
  const docs = db.prepare(`
    SELECT id, created_at, modified_at
    FROM documents
    WHERE active = 1
    ORDER BY created_at ASC
  `).all() as { id: number; created_at: string; modified_at: string }[];

  let edges = 0;

  // Create temporal edges between consecutive documents
  for (let i = 1; i < docs.length; i++) {
    const prev = docs[i - 1]!;
    const curr = docs[i]!;

    db.prepare(`
      INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type, weight, created_at)
      VALUES (?, ?, 'temporal', 1.0, ?)
    `).run(prev.id, curr.id, new Date().toISOString());

    edges++;
  }

  return edges;
}

/**
 * Build semantic graph from existing embeddings.
 * Connects documents with similarity > threshold.
 * Returns number of edges created.
 */
export async function buildSemanticGraph(
  db: Database,
  threshold: number = 0.7
): Promise<number> {
  // Query all documents with embeddings
  const docs = db.prepare(`
    SELECT DISTINCT d.id, d.hash
    FROM documents d
    JOIN content_vectors cv ON d.hash = cv.hash
    WHERE d.active = 1 AND cv.seq = 0
  `).all() as { id: number; hash: string }[];

  let edges = 0;

  // For each document, find similar neighbors
  for (let i = 0; i < docs.length; i++) {
    const doc1 = docs[i]!;

    // Find similar documents above threshold
    const similar = db.prepare(`
      SELECT
        d2.id as target_id,
        vec_distance_cosine(v1.embedding, v2.embedding) as distance
      FROM vectors_vec v1, vectors_vec v2
      JOIN documents d2 ON v2.hash_seq = d2.hash || '_0'
      WHERE v1.hash_seq = ? || '_0'
        AND d2.id != ?
        AND d2.active = 1
        AND vec_distance_cosine(v1.embedding, v2.embedding) < ?
      ORDER BY distance
      LIMIT 10
    `).all(doc1.hash, doc1.id, 1 - threshold) as { target_id: number; distance: number }[];

    for (const sim of similar) {
      const similarity = 1 - sim.distance;
      db.prepare(`
        INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation_type, weight, created_at)
        VALUES (?, ?, 'semantic', ?, ?)
      `).run(doc1.id, sim.target_id, similarity, new Date().toISOString());
      edges++;
    }
  }

  return edges;
}

// =============================================================================
// A-MEM: Causal Graph Traversal
// =============================================================================

export type CausalLink = {
  docId: number;
  title: string;
  filepath: string;
  depth: number;
  weight: number;
  reasoning: string | null;
};

export function findCausalLinks(
  db: Database,
  docId: number,
  direction: 'causes' | 'caused_by' | 'both' = 'both',
  maxDepth: number = 5
): CausalLink[] {
  if (maxDepth < 1) maxDepth = 1;
  if (maxDepth > 10) maxDepth = 10;

  let query: string;

  if (direction === 'causes') {
    // Outbound: documents this one causes
    query = `
      WITH RECURSIVE causal_chain(doc_id, depth, path) AS (
        -- Base case: immediate causal links outbound
        SELECT target_id, 1, json_array(?)
        FROM memory_relations
        WHERE source_id = ? AND relation_type = 'causal'

        UNION ALL

        -- Recursive case: follow the chain
        SELECT mr.target_id, cc.depth + 1, json_insert(cc.path, '$[#]', cc.doc_id)
        FROM memory_relations mr
        JOIN causal_chain cc ON mr.source_id = cc.doc_id
        WHERE cc.depth < ?
          AND mr.relation_type = 'causal'
          AND mr.target_id NOT IN (SELECT value FROM json_each(cc.path))
      )
      SELECT DISTINCT
        cc.doc_id as docId,
        d.title,
        d.collection || '/' || d.path as filepath,
        cc.depth,
        COALESCE(mr.weight, 1.0) as weight,
        json_extract(mr.metadata, '$.reasoning') as reasoning
      FROM causal_chain cc
      JOIN documents d ON d.id = cc.doc_id
      LEFT JOIN memory_relations mr ON (mr.source_id = ? AND mr.target_id = cc.doc_id AND mr.relation_type = 'causal')
      WHERE d.active = 1
      ORDER BY cc.depth, weight DESC
    `;
    return db.prepare(query).all(docId, docId, maxDepth, docId) as CausalLink[];
  } else if (direction === 'caused_by') {
    // Inbound: documents that cause this one
    query = `
      WITH RECURSIVE causal_chain(doc_id, depth, path) AS (
        -- Base case: immediate causal links inbound
        SELECT source_id, 1, json_array(?)
        FROM memory_relations
        WHERE target_id = ? AND relation_type = 'causal'

        UNION ALL

        -- Recursive case: follow the chain
        SELECT mr.source_id, cc.depth + 1, json_insert(cc.path, '$[#]', cc.doc_id)
        FROM memory_relations mr
        JOIN causal_chain cc ON mr.target_id = cc.doc_id
        WHERE cc.depth < ?
          AND mr.relation_type = 'causal'
          AND mr.source_id NOT IN (SELECT value FROM json_each(cc.path))
      )
      SELECT DISTINCT
        cc.doc_id as docId,
        d.title,
        d.collection || '/' || d.path as filepath,
        cc.depth,
        COALESCE(mr.weight, 1.0) as weight,
        json_extract(mr.metadata, '$.reasoning') as reasoning
      FROM causal_chain cc
      JOIN documents d ON d.id = cc.doc_id
      LEFT JOIN memory_relations mr ON (mr.target_id = ? AND mr.source_id = cc.doc_id AND mr.relation_type = 'causal')
      WHERE d.active = 1
      ORDER BY cc.depth, weight DESC
    `;
    return db.prepare(query).all(docId, docId, maxDepth, docId) as CausalLink[];
  } else {
    // Both directions
    const outbound = findCausalLinks(db, docId, 'causes', maxDepth);
    const inbound = findCausalLinks(db, docId, 'caused_by', maxDepth);

    // Merge and deduplicate
    const seen = new Set<number>();
    const merged: CausalLink[] = [];

    for (const link of [...outbound, ...inbound]) {
      if (!seen.has(link.docId)) {
        seen.add(link.docId);
        merged.push(link);
      }
    }

    return merged.sort((a, b) => a.depth - b.depth || b.weight - a.weight);
  }
}

// =============================================================================
// A-MEM: Memory Evolution Timeline
// =============================================================================

export type EvolutionEntry = {
  version: number;
  triggeredBy: {
    docId: number;
    title: string;
    filepath: string;
  };
  previousKeywords: string[] | null;
  newKeywords: string[] | null;
  previousContext: string | null;
  newContext: string | null;
  reasoning: string | null;
  createdAt: string;
};

export function getEvolutionTimeline(
  db: Database,
  docId: number,
  limit: number = 10
): EvolutionEntry[] {
  if (limit < 1) limit = 1;
  if (limit > 100) limit = 100;

  const query = `
    SELECT
      e.version,
      e.triggered_by,
      d.title as trigger_title,
      d.collection || '/' || d.path as trigger_filepath,
      e.previous_keywords,
      e.new_keywords,
      e.previous_context,
      e.new_context,
      e.reasoning,
      e.created_at
    FROM memory_evolution e
    JOIN documents d ON d.id = e.triggered_by
    WHERE e.memory_id = ?
      AND d.active = 1
    ORDER BY e.created_at DESC
    LIMIT ?
  `;

  const rows = db.prepare(query).all(docId, limit) as Array<{
    version: number;
    triggered_by: number;
    trigger_title: string;
    trigger_filepath: string;
    previous_keywords: string | null;
    new_keywords: string | null;
    previous_context: string | null;
    new_context: string | null;
    reasoning: string | null;
    created_at: string;
  }>;

  return rows.map(row => {
    // Parse JSON keywords if present
    let prevKeywords: string[] | null = null;
    let newKeywords: string[] | null = null;

    try {
      prevKeywords = row.previous_keywords ? JSON.parse(row.previous_keywords) : null;
    } catch (e) {
      console.error('[amem] Failed to parse previous_keywords:', e);
    }

    try {
      newKeywords = row.new_keywords ? JSON.parse(row.new_keywords) : null;
    } catch (e) {
      console.error('[amem] Failed to parse new_keywords:', e);
    }

    return {
      version: row.version,
      triggeredBy: {
        docId: row.triggered_by,
        title: row.trigger_title,
        filepath: row.trigger_filepath,
      },
      previousKeywords: prevKeywords,
      newKeywords: newKeywords,
      previousContext: row.previous_context,
      newContext: row.new_context,
      reasoning: row.reasoning,
      createdAt: row.created_at,
    };
  });
}
