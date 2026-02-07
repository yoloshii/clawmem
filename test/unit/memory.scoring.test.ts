import { describe, test, expect } from "bun:test";
import { recencyScore, compositeScore, confidenceScore, DEFAULT_WEIGHTS } from "../../src/memory.ts";

describe("recencyScore", () => {
  test("returns 1.0 for current date", () => {
    const now = new Date();
    expect(recencyScore(now, "note", now)).toBe(1.0);
  });

  test("returns 1.0 for future date", () => {
    const now = new Date();
    const future = new Date(now.getTime() + 86400000);
    expect(recencyScore(future, "note", now)).toBe(1.0);
  });

  test("decays over time", () => {
    const now = new Date();
    const tenDaysAgo = new Date(now.getTime() - 10 * 86400000);
    const score = recencyScore(tenDaysAgo, "note", now);
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1.0);
  });

  test("returns 1.0 for infinite half-life types", () => {
    const now = new Date();
    const yearAgo = new Date(now.getTime() - 365 * 86400000);
    expect(recencyScore(yearAgo, "decision", now)).toBe(1.0);
    expect(recencyScore(yearAgo, "hub", now)).toBe(1.0);
  });

  test("handles string date input", () => {
    const now = new Date();
    const score = recencyScore(now.toISOString(), "note", now);
    expect(score).toBe(1.0);
  });

  test("handles invalid date string gracefully (returns 0.5)", () => {
    const score = recencyScore("not-a-date", "note");
    expect(score).toBe(0.5);
  });

  test("handles unknown content type with default half-life", () => {
    const now = new Date();
    const score = recencyScore(now, "unknown-type", now);
    expect(score).toBe(1.0);
  });
});

describe("compositeScore", () => {
  test("computes weighted sum", () => {
    const result = compositeScore(1.0, 1.0, 1.0, DEFAULT_WEIGHTS);
    expect(result).toBeCloseTo(1.0, 2);
  });

  test("respects weights", () => {
    const result = compositeScore(0.5, 0.5, 0.5, { search: 0.5, recency: 0.25, confidence: 0.25 });
    expect(result).toBeCloseTo(0.5, 2);
  });

  test("handles NaN searchScore gracefully (treats as 0)", () => {
    const result = compositeScore(NaN, 0.5, 0.5, DEFAULT_WEIGHTS);
    expect(Number.isFinite(result)).toBe(true);
  });

  test("handles NaN recency gracefully", () => {
    const result = compositeScore(0.5, NaN, 0.5, DEFAULT_WEIGHTS);
    expect(Number.isFinite(result)).toBe(true);
  });

  test("handles Infinity gracefully", () => {
    const result = compositeScore(Infinity, 0.5, 0.5, DEFAULT_WEIGHTS);
    expect(Number.isFinite(result)).toBe(true);
  });

  test("all NaN inputs return 0", () => {
    const result = compositeScore(NaN, NaN, NaN, DEFAULT_WEIGHTS);
    expect(result).toBe(0);
  });
});

describe("confidenceScore", () => {
  test("returns within 0-1 range", () => {
    const now = new Date();
    const score = confidenceScore("note", now, 5, now);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1.0);
  });

  test("access count boost is bounded", () => {
    const now = new Date();
    const highAccess = confidenceScore("note", now, 1000000, now);
    expect(highAccess).toBeLessThanOrEqual(1.0);
  });

  test("handles NaN accessCount gracefully", () => {
    const now = new Date();
    const score = confidenceScore("note", now, NaN, now);
    expect(Number.isFinite(score)).toBe(true);
    expect(score).toBeGreaterThanOrEqual(0);
  });

  test("handles negative accessCount gracefully", () => {
    const now = new Date();
    const score = confidenceScore("note", now, -5, now);
    expect(Number.isFinite(score)).toBe(true);
    expect(score).toBeGreaterThanOrEqual(0);
  });

  test("handles Infinity accessCount gracefully", () => {
    const now = new Date();
    const score = confidenceScore("note", now, Infinity, now);
    expect(Number.isFinite(score)).toBe(true);
    expect(score).toBeLessThanOrEqual(1.0);
  });
});
