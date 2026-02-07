/**
 * Standardized error types for ClawMem.
 */

export class ClawMemError extends Error {
  code: string;
  details?: Record<string, unknown>;

  constructor(code: string, message: string, details?: Record<string, unknown>, cause?: Error) {
    super(message);
    this.name = "ClawMemError";
    this.code = code;
    this.details = details;
    if (cause) this.cause = cause;
  }

  toJSON(): { ok: false; error: { code: string; message: string; details?: Record<string, unknown> } } {
    return {
      ok: false,
      error: {
        code: this.code,
        message: this.message,
        ...(this.details ? { details: this.details } : {}),
      },
    };
  }
}

/** Format any error into a user-facing message (no stack traces). */
export function toUserError(err: unknown): string {
  if (err instanceof ClawMemError) return `[${err.code}] ${err.message}`;
  if (err instanceof Error) return err.message;
  return String(err);
}

/** Format any error into structured JSON for hooks/MCP boundaries. */
export function toErrorResponse(err: unknown): { ok: false; error: { code: string; message: string } } {
  if (err instanceof ClawMemError) return err.toJSON();
  const message = err instanceof Error ? err.message : String(err);
  return { ok: false, error: { code: "INTERNAL_ERROR", message } };
}
