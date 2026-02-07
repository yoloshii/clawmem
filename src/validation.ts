/**
 * Lightweight input validation helpers for module boundaries.
 */

import { ClawMemError } from "./errors.ts";
import { MAX_PATH_LENGTH } from "./limits.ts";

export function assertNonEmptyString(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string" || value.length === 0) {
    throw new ClawMemError("INVALID_INPUT", `${name} must be a non-empty string`);
  }
}

export function assertMaxLength(value: string, max: number, name: string): void {
  if (value.length > max) {
    throw new ClawMemError("INPUT_TOO_LONG", `${name} exceeds max length ${max} (got ${value.length})`, {
      max,
      actual: value.length,
    });
  }
}

export function assertFiniteNumber(value: unknown, name: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ClawMemError("INVALID_NUMBER", `${name} must be a finite number (got ${value})`);
  }
}

export function assertBounds(value: number, min: number, max: number, name: string): void {
  if (!Number.isFinite(value) || value < min || value > max) {
    throw new ClawMemError("OUT_OF_BOUNDS", `${name} must be between ${min} and ${max} (got ${value})`, {
      min,
      max,
      actual: value,
    });
  }
}

export function assertArrayLengthMatch(a: unknown[], b: unknown[], nameA: string, nameB: string): void {
  if (a.length !== b.length) {
    throw new ClawMemError(
      "LENGTH_MISMATCH",
      `${nameA} length (${a.length}) must match ${nameB} length (${b.length})`
    );
  }
}

export function assertSafePath(filepath: string, name: string = "path"): void {
  if (filepath.length > MAX_PATH_LENGTH) {
    throw new ClawMemError("PATH_TOO_LONG", `${name} exceeds max length ${MAX_PATH_LENGTH}`);
  }
  if (filepath.includes("\0")) {
    throw new ClawMemError("INVALID_PATH", `${name} contains null bytes`);
  }
  // Normalize and check for traversal
  const normalized = filepath.replace(/\\/g, "/");
  const segments = normalized.split("/");
  if (segments.some((s) => s === "..")) {
    throw new ClawMemError("PATH_TRAVERSAL", `${name} contains path traversal (..)`);
  }
}

/** Clamp a number to bounds (for cases where clamping is the existing behavior). */
export function clampBounds(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, value));
}
