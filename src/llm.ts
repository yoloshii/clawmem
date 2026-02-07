/**
 * llm.ts - LLM abstraction layer for QMD using node-llama-cpp
 *
 * Provides text generation and reranking using local GGUF models.
 * Embeddings are served by a remote GPU server (CLAWMEM_EMBED_URL).
 */

// node-llama-cpp is loaded lazily to avoid ~630ms import cost when all
// operations route to remote GPU servers. Only loaded if a local fallback
// is actually needed (GPU server down).
let _nodeLlamaCpp: typeof import("node-llama-cpp") | null = null;
async function getNodeLlamaCpp() {
  if (!_nodeLlamaCpp) {
    _nodeLlamaCpp = await import("node-llama-cpp");
  }
  return _nodeLlamaCpp;
}

// Re-export type aliases for internal use (structural, no runtime cost)
type Llama = any;
type LlamaModel = any;
type LlamaToken = any;

import { homedir } from "os";
import { join } from "path";
import { existsSync, mkdirSync } from "fs";

// =============================================================================
// Embedding Formatting Functions
// =============================================================================

/**
 * Format a query for embedding.
 * Uses task prefix format for embedding models.
 */
export function formatQueryForEmbedding(query: string): string {
  return `task: search result | query: ${query}`;
}

/**
 * Format a document for embedding.
 * Uses title + text format for embedding models.
 */
export function formatDocForEmbedding(text: string, title?: string): string {
  return `title: ${title || "none"} | text: ${text}`;
}

// =============================================================================
// Types
// =============================================================================

/**
 * Token with log probability
 */
export type TokenLogProb = {
  token: string;
  logprob: number;
};

/**
 * Embedding result
 */
export type EmbeddingResult = {
  embedding: number[];
  model: string;
};

/**
 * Generation result with optional logprobs
 */
export type GenerateResult = {
  text: string;
  model: string;
  logprobs?: TokenLogProb[];
  done: boolean;
};

/**
 * Rerank result for a single document
 */
export type RerankDocumentResult = {
  file: string;
  score: number;
  index: number;
};

/**
 * Batch rerank result
 */
export type RerankResult = {
  results: RerankDocumentResult[];
  model: string;
};

/**
 * Model info
 */
export type ModelInfo = {
  name: string;
  exists: boolean;
  path?: string;
};

/**
 * Options for embedding
 */
export type EmbedOptions = {
  model?: string;
  isQuery?: boolean;
  title?: string;
};

/**
 * Options for text generation
 */
export type GenerateOptions = {
  model?: string;
  maxTokens?: number;
  temperature?: number;
  signal?: AbortSignal;
};

/**
 * Options for reranking
 */
export type RerankOptions = {
  model?: string;
};

/**
 * Supported query types for different search backends
 */
export type QueryType = 'lex' | 'vec' | 'hyde';

/**
 * A single query and its target backend type
 */
export type Queryable = {
  type: QueryType;
  text: string;
};

/**
 * Document to rerank
 */
export type RerankDocument = {
  file: string;
  text: string;
  title?: string;
};

// =============================================================================
// Model Configuration
// =============================================================================

// HuggingFace model URIs for node-llama-cpp
// Format: hf:<user>/<repo>/<file>
const DEFAULT_RERANK_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
const DEFAULT_GENERATE_MODEL = "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";

// Local model cache directory
const MODEL_CACHE_DIR = join(homedir(), ".cache", "qmd", "models");

// =============================================================================
// LLM Interface
// =============================================================================

/**
 * Abstract LLM interface - implement this for different backends
 */
export interface LLM {
  /**
   * Get embeddings for text
   */
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;

  /**
   * Generate text completion
   */
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null>;

  /**
   * Check if a model exists/is available
   */
  modelExists(model: string): Promise<ModelInfo>;

  /**
   * Expand a search query into multiple variations for different backends.
   * Returns a list of Queryable objects.
   */
  expandQuery(query: string, options?: { context?: string, includeLexical?: boolean }): Promise<Queryable[]>;

  /**
   * Rerank documents by relevance to a query
   * Returns list of documents with relevance scores (higher = more relevant)
   */
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;

  /**
   * Dispose of resources
   */
  dispose(): Promise<void>;
}

// =============================================================================
// node-llama-cpp Implementation
// =============================================================================

export type LlamaCppConfig = {
  generateModel?: string;
  rerankModel?: string;
  modelCacheDir?: string;
  /**
   * Remote embedding server URL (e.g. "http://your-gpu-server:8088").
   * When set, embed() uses HTTP POST to /v1/embeddings instead of local node-llama-cpp.
   * Env: CLAWMEM_EMBED_URL
   */
  remoteEmbedUrl?: string;
  /**
   * Remote LLM server URL for text generation (e.g. http://localhost:8089).
   * When set, generate() calls /v1/chat/completions instead of local node-llama-cpp.
   */
  remoteLlmUrl?: string;
  /**
   * Inactivity timeout in ms before unloading contexts (default: 2 minutes, 0 to disable).
   *
   * Per node-llama-cpp lifecycle guidance, we prefer keeping models loaded and only disposing
   * contexts when idle, since contexts (and their sequences) are the heavy per-session objects.
   * @see https://node-llama-cpp.withcat.ai/guide/objects-lifecycle
   */
  inactivityTimeoutMs?: number;
  /**
   * Whether to dispose models on inactivity (default: false).
   *
   * Keeping models loaded avoids repeated VRAM thrash; set to true only if you need aggressive
   * memory reclaim.
   */
  disposeModelsOnInactivity?: boolean;
};

/**
 * LLM implementation using node-llama-cpp
 */
// Default inactivity timeout: 2 minutes
const DEFAULT_INACTIVITY_TIMEOUT_MS = 2 * 60 * 1000;

export class LlamaCpp implements LLM {
  private llama: Llama | null = null;
  private generateModel: LlamaModel | null = null;
  private rerankModel: LlamaModel | null = null;
  private rerankContext: Awaited<ReturnType<LlamaModel["createRankingContext"]>> | null = null;

  private generateModelUri: string;
  private rerankModelUri: string;
  private modelCacheDir: string;
  private remoteEmbedUrl: string | null;
  private remoteLlmUrl: string | null;

  // Ensure we don't load the same model concurrently (which can allocate duplicate VRAM).
  private generateModelLoadPromise: Promise<LlamaModel> | null = null;
  private rerankModelLoadPromise: Promise<LlamaModel> | null = null;

  // Inactivity timer for auto-unloading models
  private inactivityTimer: ReturnType<typeof setTimeout> | null = null;
  private inactivityTimeoutMs: number;
  private disposeModelsOnInactivity: boolean;

  // Track disposal state to prevent double-dispose
  private disposed = false;


  constructor(config: LlamaCppConfig = {}) {
    this.generateModelUri = config.generateModel || DEFAULT_GENERATE_MODEL;
    this.rerankModelUri = config.rerankModel || DEFAULT_RERANK_MODEL;
    this.modelCacheDir = config.modelCacheDir || MODEL_CACHE_DIR;
    this.remoteEmbedUrl = config.remoteEmbedUrl || null;
    this.remoteLlmUrl = config.remoteLlmUrl || null;
    this.inactivityTimeoutMs = config.inactivityTimeoutMs ?? DEFAULT_INACTIVITY_TIMEOUT_MS;
    this.disposeModelsOnInactivity = config.disposeModelsOnInactivity ?? false;
  }

  /**
   * Reset the inactivity timer. Called after each model operation.
   * When timer fires, models are unloaded to free memory.
   */
  private touchActivity(): void {
    // Clear existing timer
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }

    // Only set timer if we have disposable contexts and timeout is enabled
    if (this.inactivityTimeoutMs > 0 && this.hasLoadedContexts()) {
      this.inactivityTimer = setTimeout(() => {
        this.unloadIdleResources().catch(err => {
          console.error("Error unloading idle resources:", err);
        });
      }, this.inactivityTimeoutMs);
      // Don't keep process alive just for this timer
      this.inactivityTimer.unref();
    }
  }

  /**
   * Check if any contexts are currently loaded (and therefore worth unloading on inactivity).
   */
  private hasLoadedContexts(): boolean {
    return !!this.rerankContext;
  }

  /**
   * Unload idle resources but keep the instance alive for future use.
   *
   * By default, this disposes contexts (and their dependent sequences), while keeping models loaded.
   * This matches the intended lifecycle: model → context → sequence, where contexts are per-session.
   */
  async unloadIdleResources(): Promise<void> {
    // Don't unload if already disposed
    if (this.disposed) {
      return;
    }

    // Clear timer
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }

    // Dispose contexts first
    if (this.rerankContext) {
      await this.rerankContext.dispose();
      this.rerankContext = null;
    }

    // Optionally dispose models too (opt-in)
    if (this.disposeModelsOnInactivity) {
      if (this.generateModel) {
        await this.generateModel.dispose();
        this.generateModel = null;
      }
      if (this.rerankModel) {
        await this.rerankModel.dispose();
        this.rerankModel = null;
      }
      // Reset load promises so models can be reloaded later
      this.generateModelLoadPromise = null;
      this.rerankModelLoadPromise = null;
    }

    // Note: We keep llama instance alive - it's lightweight
  }

  /**
   * Ensure model cache directory exists
   */
  private ensureModelCacheDir(): void {
    if (!existsSync(this.modelCacheDir)) {
      mkdirSync(this.modelCacheDir, { recursive: true });
    }
  }

  /**
   * Initialize the llama instance (lazy)
   */
  private async ensureLlama(): Promise<Llama> {
    if (!this.llama) {
      const { getLlama, LlamaLogLevel } = await getNodeLlamaCpp();
      this.llama = await getLlama({ logLevel: LlamaLogLevel.error });
    }
    return this.llama;
  }

  /**
   * Resolve a model URI to a local path, downloading if needed
   */
  private async resolveModel(modelUri: string): Promise<string> {
    this.ensureModelCacheDir();
    const { resolveModelFile } = await getNodeLlamaCpp();
    return await resolveModelFile(modelUri, this.modelCacheDir);
  }

  /**
   * Load generation model (lazy) - context is created fresh per call
   */
  private async ensureGenerateModel(): Promise<LlamaModel> {
    if (!this.generateModel) {
      if (this.generateModelLoadPromise) {
        return await this.generateModelLoadPromise;
      }

      this.generateModelLoadPromise = (async () => {
        const llama = await this.ensureLlama();
        const modelPath = await this.resolveModel(this.generateModelUri);
        const model = await llama.loadModel({ modelPath });
        this.generateModel = model;
        return model;
      })();

      try {
        await this.generateModelLoadPromise;
      } finally {
        this.generateModelLoadPromise = null;
      }
    }
    this.touchActivity();
    if (!this.generateModel) {
      throw new Error("Generate model not loaded");
    }
    return this.generateModel;
  }

  /**
   * Load rerank model (lazy)
   */
  private async ensureRerankModel(): Promise<LlamaModel> {
    if (this.rerankModel) {
      return this.rerankModel;
    }
    if (this.rerankModelLoadPromise) {
      return await this.rerankModelLoadPromise;
    }

    this.rerankModelLoadPromise = (async () => {
      const llama = await this.ensureLlama();
      const modelPath = await this.resolveModel(this.rerankModelUri);
      const model = await llama.loadModel({ modelPath });
      this.rerankModel = model;
      return model;
    })();

    try {
      return await this.rerankModelLoadPromise;
    } finally {
      this.rerankModelLoadPromise = null;
    }
  }

  /**
   * Load rerank context (lazy). Context can be disposed and recreated without reloading the model.
   */
  private async ensureRerankContext(): Promise<Awaited<ReturnType<LlamaModel["createRankingContext"]>>> {
    if (!this.rerankContext) {
      const model = await this.ensureRerankModel();
      this.rerankContext = await model.createRankingContext();
    }
    this.touchActivity();
    return this.rerankContext;
  }

  // ==========================================================================
  // Tokenization
  // ==========================================================================

  /**
   * Tokenize text using the generate model's tokenizer
   * Returns tokenizer tokens (opaque type from node-llama-cpp)
   */
  async tokenize(text: string): Promise<readonly LlamaToken[]> {
    const model = await this.ensureGenerateModel();
    return model.tokenize(text);
  }

  /**
   * Count tokens in text using the generate model's tokenizer
   */
  async countTokens(text: string): Promise<number> {
    const tokens = await this.tokenize(text);
    return tokens.length;
  }

  /**
   * Detokenize token IDs back to text
   */
  async detokenize(tokens: readonly LlamaToken[]): Promise<string> {
    const model = await this.ensureGenerateModel();
    return model.detokenize(tokens);
  }

  // ==========================================================================
  // Core API methods
  // ==========================================================================

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    if (!this.remoteEmbedUrl) {
      console.error("[embed] CLAWMEM_EMBED_URL not set — remote GPU embedding server required");
      return null;
    }
    return this.embedRemote(text);
  }

  /**
   * Batch embed multiple texts efficiently via single HTTP request.
   */
  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];
    if (!this.remoteEmbedUrl) {
      console.error("[embed] CLAWMEM_EMBED_URL not set — remote GPU embedding server required");
      return texts.map(() => null);
    }
    return this.embedRemoteBatch(texts);
  }

  // ---------- Remote embedding (GPU server via /v1/embeddings) ----------

  // granite-embedding-278m: 512 token context. With -c 2048 server context,
  // the server can handle more tokens but quality degrades past model training length.
  // Empirically: failures at ~1200+ chars; 1100 is safe across all content types.
  private static readonly MAX_REMOTE_EMBED_CHARS = 1100;

  private async embedRemote(text: string): Promise<EmbeddingResult | null> {
    try {
      const input = text.length > LlamaCpp.MAX_REMOTE_EMBED_CHARS
        ? text.slice(0, LlamaCpp.MAX_REMOTE_EMBED_CHARS) : text;
      const resp = await fetch(`${this.remoteEmbedUrl}/v1/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input, model: "embedding" }),
      });
      if (!resp.ok) {
        console.error(`Remote embed HTTP ${resp.status}: ${await resp.text()}`);
        return null;
      }
      const data = await resp.json() as {
        data: { embedding: number[] }[];
        model?: string;
      };
      return {
        embedding: data.data[0]!.embedding,
        model: data.model || this.remoteEmbedUrl!,
      };
    } catch (error) {
      console.error("Remote embed error:", error);
      return null;
    }
  }

  private async embedRemoteBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    try {
      const truncated = texts.map(t =>
        t.length > LlamaCpp.MAX_REMOTE_EMBED_CHARS ? t.slice(0, LlamaCpp.MAX_REMOTE_EMBED_CHARS) : t
      );
      const resp = await fetch(`${this.remoteEmbedUrl}/v1/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: truncated, model: "embedding" }),
      });
      if (!resp.ok) {
        console.error(`Remote batch embed HTTP ${resp.status}: ${await resp.text()}`);
        return texts.map(() => null);
      }
      const data = await resp.json() as {
        data: { embedding: number[]; index: number }[];
        model?: string;
      };
      const modelName = data.model || this.remoteEmbedUrl!;
      // Server returns data sorted by index
      const results: (EmbeddingResult | null)[] = new Array(texts.length).fill(null);
      for (const item of data.data) {
        results[item.index] = { embedding: item.embedding, model: modelName };
      }
      return results;
    } catch (error) {
      console.error("Remote batch embed error:", error);
      return texts.map(() => null);
    }
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    const maxTokens = options.maxTokens ?? 150;
    const temperature = options.temperature ?? 0;

    // Remote LLM server (GPU) — preferred path
    if (this.remoteLlmUrl) {
      return this.generateRemote(prompt, maxTokens, temperature, options.signal);
    }

    // Local fallback via node-llama-cpp (CPU)
    await this.ensureGenerateModel();

    const context = await this.generateModel!.createContext();
    const sequence = context.getSequence();
    const { LlamaChatSession } = await getNodeLlamaCpp();
    const session = new LlamaChatSession({ contextSequence: sequence });

    let result = "";
    try {
      await session.prompt(prompt, {
        maxTokens,
        temperature,
        signal: options.signal,
        stopOnAbortSignal: true,
        onTextChunk: (text) => {
          result += text;
        },
      });

      return {
        text: result,
        model: this.generateModelUri,
        done: true,
      };
    } finally {
      await context.dispose();
    }
  }

  private async generateRemote(
    prompt: string,
    maxTokens: number,
    temperature: number,
    signal?: AbortSignal
  ): Promise<GenerateResult | null> {
    try {
      const resp = await fetch(`${this.remoteLlmUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "qwen3",
          messages: [{ role: "user", content: `${prompt} /no_think` }],
          max_tokens: maxTokens,
          temperature,
        }),
        signal,
      });

      if (!resp.ok) {
        console.error(`[generate] Remote LLM error: ${resp.status} ${resp.statusText}`);
        return null;
      }

      const data = await resp.json() as {
        choices: { message: { content: string } }[];
        model?: string;
      };

      return {
        text: data.choices[0]?.message?.content || "",
        model: data.model || this.remoteLlmUrl!,
        done: true,
      };
    } catch (error) {
      console.error("[generate] Remote LLM error:", error);
      return null;
    }
  }

  private async expandQueryRemote(query: string, includeLexical: boolean, context?: string): Promise<Queryable[]> {
    const prompt = `Rewrite this search query for better retrieval. Output lines in format "type: text" where type is lex, vec, or hyde.
- lex: keyword search terms (1-3 lines)
- vec: semantic search queries (1-3 lines)
- hyde: hypothetical document passage that answers the query (1 line)

Query: ${query}${context ? `\nContext: ${context}` : ""}

Output:`;

    const result = await this.generateRemote(prompt, 500, 0.7);
    if (!result?.text) {
      const fallback: Queryable[] = [{ type: 'vec', text: query }];
      if (includeLexical) fallback.unshift({ type: 'lex', text: query });
      return fallback;
    }

    const lines = result.text.trim().split("\n");
    const queryables: Queryable[] = lines.map(line => {
      const colonIdx = line.indexOf(":");
      if (colonIdx === -1) return null;
      const type = line.slice(0, colonIdx).trim();
      if (type !== 'lex' && type !== 'vec' && type !== 'hyde') return null;
      const text = line.slice(colonIdx + 1).trim();
      if (!text) return null;
      return { type: type as QueryType, text };
    }).filter((q): q is Queryable => q !== null);

    if (queryables.length === 0) {
      const fallback: Queryable[] = [{ type: 'vec', text: query }];
      if (includeLexical) fallback.unshift({ type: 'lex', text: query });
      return fallback;
    }

    if (!includeLexical) {
      return queryables.filter(q => q.type !== 'lex');
    }
    return queryables;
  }

  async modelExists(modelUri: string): Promise<ModelInfo> {
    // For HuggingFace URIs, we assume they exist
    // For local paths, check if file exists
    if (modelUri.startsWith("hf:")) {
      return { name: modelUri, exists: true };
    }

    const exists = existsSync(modelUri);
    return {
      name: modelUri,
      exists,
      path: exists ? modelUri : undefined,
    };
  }

  // ==========================================================================
  // High-level abstractions
  // ==========================================================================

  async expandQuery(query: string, options: { context?: string, includeLexical?: boolean } = {}): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;
    const context = options.context;

    // Remote LLM path — no grammar constraint, parse output instead
    if (this.remoteLlmUrl) {
      return this.expandQueryRemote(query, includeLexical, context);
    }

    const llama = await this.ensureLlama();
    await this.ensureGenerateModel();

    const grammar = await llama.createGrammar({
      grammar: `
        root ::= line+
        line ::= type ": " content "\\n"
        type ::= "lex" | "vec" | "hyde"
        content ::= [^\\n]+
      `
    });

    const prompt = `You are a search query optimization expert. Your task is to improve retrieval by rewriting queries and generating hypothetical documents.

Original Query: ${query}

${context ? `Additional Context, ONLY USE IF RELEVANT:\n\n<context>${context}</context>` : ""}

## Step 1: Query Analysis
Identify entities, search intent, and missing context.

## Step 2: Generate Hypothetical Document
Write a focused sentence passage that would answer the query. Include specific terminology and domain vocabulary.

## Step 3: Query Rewrites
Generate 2-3 alternative search queries that resolve ambiguities. Use terminology from the hypothetical document.

## Step 4: Final Retrieval Text
Output exactly 1-3 'lex' lines, 1-3 'vec' lines, and MAX ONE 'hyde' line.

<format>
lex: {single search term}
vec: {single vector query}
hyde: {complete hypothetical document passage from Step 2 on a SINGLE LINE}
</format>

<example>
Example (FOR FORMAT ONLY - DO NOT COPY THIS CONTENT):
lex: example keyword 1
lex: example keyword 2
vec: example semantic query
hyde: This is an example of a hypothetical document passage that would answer the example query. It contains multiple sentences and relevant vocabulary.
</example>

<rules>
- DO NOT repeat the same line.
- Each 'lex:' line MUST be a different keyword variation based on the ORIGINAL QUERY.
- Each 'vec:' line MUST be a different semantic variation based on the ORIGINAL QUERY.
- The 'hyde:' line MUST be the full sentence passage from Step 2, but all on one line.
- DO NOT use the example content above.
${!includeLexical ? "- Do NOT output any 'lex:' lines" : ""}
</rules>

Final Output:`;

    // Create fresh context for each call
    const genContext = await this.generateModel!.createContext();
    const sequence = genContext.getSequence();
    const { LlamaChatSession } = await getNodeLlamaCpp();
    const session = new LlamaChatSession({ contextSequence: sequence });

    try {
      const result = await session.prompt(prompt, {
        grammar,
        maxTokens: 1000,
        temperature: 1,
      });

      const lines = result.trim().split("\n");
      const queryables: Queryable[] = lines.map(line => {
        const colonIdx = line.indexOf(":");
        if (colonIdx === -1) return null;
        const type = line.slice(0, colonIdx).trim();
        if (type !== 'lex' && type !== 'vec' && type !== 'hyde') return null;
        const text = line.slice(colonIdx + 1).trim();
        return { type: type as QueryType, text };
      }).filter((q): q is Queryable => q !== null);

      // Filter out lex entries if not requested
      if (!includeLexical) {
        return queryables.filter(q => q.type !== 'lex');
      }
      return queryables;
    } catch (error) {
      console.error("Structured query expansion failed:", error);
      // Fallback to original query
      const fallback: Queryable[] = [{ type: 'vec', text: query }];
      if (includeLexical) fallback.unshift({ type: 'lex', text: query });
      return fallback;
    } finally {
      await genContext.dispose();
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    const context = await this.ensureRerankContext();

    // Build a map from document text to original indices (for lookup after sorting)
    const textToDoc = new Map<string, { file: string; index: number }>();
    documents.forEach((doc, index) => {
      textToDoc.set(doc.text, { file: doc.file, index });
    });

    // Extract just the text for ranking
    const texts = documents.map((doc) => doc.text);

    // Use the proper ranking API - returns [{document: string, score: number}] sorted by score
    const ranked = await context.rankAndSort(query, texts);

    // Map back to our result format using the text-to-doc map
    const results: RerankDocumentResult[] = ranked.map((item) => {
      const docInfo = textToDoc.get(item.document)!;
      return {
        file: docInfo.file,
        score: item.score,
        index: docInfo.index,
      };
    });

    return {
      results,
      model: this.rerankModelUri,
    };
  }

  async dispose(): Promise<void> {
    // Prevent double-dispose
    if (this.disposed) {
      return;
    }
    this.disposed = true;

    // Clear inactivity timer
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }

    // Disposing llama cascades to models and contexts automatically
    // See: https://node-llama-cpp.withcat.ai/guide/objects-lifecycle
    // Note: llama.dispose() can hang indefinitely, so we use a timeout
    if (this.llama) {
      const disposePromise = this.llama.dispose();
      const timeoutPromise = new Promise<void>((resolve) => setTimeout(resolve, 1000));
      await Promise.race([disposePromise, timeoutPromise]);
    }

    // Clear references
    this.rerankContext = null;
    this.generateModel = null;
    this.rerankModel = null;
    this.llama = null;

    // Clear any in-flight load promises
    this.generateModelLoadPromise = null;
    this.rerankModelLoadPromise = null;
  }
}

// =============================================================================
// Singleton for default LlamaCpp instance
// =============================================================================

let defaultLlamaCpp: LlamaCpp | null = null;

/**
 * Get the default LlamaCpp instance (creates one if needed).
 * Reads CLAWMEM_EMBED_URL env var for remote GPU embedding.
 */
export function getDefaultLlamaCpp(): LlamaCpp {
  if (!defaultLlamaCpp) {
    defaultLlamaCpp = new LlamaCpp({
      remoteEmbedUrl: process.env.CLAWMEM_EMBED_URL || undefined,
      remoteLlmUrl: process.env.CLAWMEM_LLM_URL || undefined,
    });
  }
  return defaultLlamaCpp;
}

/**
 * Set a custom default LlamaCpp instance (useful for testing)
 */
export function setDefaultLlamaCpp(llm: LlamaCpp | null): void {
  defaultLlamaCpp = llm;
}

/**
 * Dispose the default LlamaCpp instance if it exists.
 * Call this before process exit to prevent NAPI crashes.
 */
export async function disposeDefaultLlamaCpp(): Promise<void> {
  if (defaultLlamaCpp) {
    await defaultLlamaCpp.dispose();
    defaultLlamaCpp = null;
  }
}

