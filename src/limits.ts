/**
 * Centralized limits for input validation and resource bounding.
 */

// Search & query
export const MAX_QUERY_LENGTH = 10_000;
export const MAX_SEARCH_LIMIT = 100;

// LLM
export const MAX_LLM_INPUT_CHARS = 100_000;
export const MAX_LLM_GENERATE_TIMEOUT_MS = 120_000; // 2 minutes

// Transcripts & hooks
export const MAX_TRANSCRIPT_BYTES = 50 * 1024 * 1024; // 50 MB
export const MAX_FILES_EXTRACTED = 200;

// Document processing
export const MAX_FRAGMENTS_PER_DOC = 500;
export const MAX_SPLITTER_INPUT_CHARS = 500_000;
export const MAX_FILE_LINES_READ = 100_000;

// Profile
export const MAX_LEVENSHTEIN_LENGTH = 1_000;

// Paths
export const MAX_PATH_LENGTH = 1_000;
