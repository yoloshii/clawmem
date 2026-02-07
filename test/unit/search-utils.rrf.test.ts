import { describe, test, expect } from "bun:test";
import { reciprocalRankFusion, type RankedResult } from "../../src/search-utils.ts";

function makeResult(file: string, score: number): RankedResult {
  return { file, displayPath: file, title: file, body: "", score };
}

describe("reciprocalRankFusion", () => {
  test("merges two ranked lists", () => {
    const list1 = [makeResult("a.md", 1.0), makeResult("b.md", 0.8)];
    const list2 = [makeResult("b.md", 1.0), makeResult("c.md", 0.6)];

    const result = reciprocalRankFusion([list1, list2], [1, 1]);
    expect(result.length).toBe(3);
    // b.md appears in both → highest score
    expect(result[0]!.file).toBe("b.md");
  });

  test("respects weights", () => {
    const list1 = [makeResult("a.md", 1.0)];
    const list2 = [makeResult("b.md", 1.0)];

    const result = reciprocalRankFusion([list1, list2], [10, 1]);
    // a.md has weight 10, b.md has weight 1
    expect(result[0]!.file).toBe("a.md");
  });

  test("throws on weight/list length mismatch", () => {
    const list1 = [makeResult("a.md", 1.0)];
    expect(() => reciprocalRankFusion([list1], [1, 2])).toThrow("must match");
  });

  test("handles empty lists", () => {
    const result = reciprocalRankFusion([], []);
    expect(result).toHaveLength(0);
  });

  test("handles empty weights (defaults to 1)", () => {
    const list1 = [makeResult("a.md", 1.0)];
    const result = reciprocalRankFusion([list1], []);
    expect(result).toHaveLength(1);
  });

  test("single list passthrough", () => {
    const list1 = [makeResult("a.md", 1.0), makeResult("b.md", 0.5)];
    const result = reciprocalRankFusion([list1], [1]);
    expect(result).toHaveLength(2);
    expect(result[0]!.file).toBe("a.md");
  });

  test("sanitizes NaN weights to 1", () => {
    const list1 = [makeResult("a.md", 1.0)];
    const list2 = [makeResult("b.md", 1.0)];
    const result = reciprocalRankFusion([list1, list2], [NaN, 1]);
    expect(result).toHaveLength(2);
    // NaN weight becomes 1, so both lists have equal weight
    expect(result.every(r => Number.isFinite(r.score))).toBe(true);
  });

  test("sanitizes negative weights to 1", () => {
    const list1 = [makeResult("a.md", 1.0)];
    const result = reciprocalRankFusion([list1], [-5]);
    expect(result).toHaveLength(1);
    expect(result[0]!.score).toBeGreaterThan(0);
  });

  test("skips zero-weight lists", () => {
    const list1 = [makeResult("a.md", 1.0)];
    const list2 = [makeResult("b.md", 1.0)];
    const result = reciprocalRankFusion([list1, list2], [1, 0]);
    // b.md from zero-weight list should not appear
    expect(result).toHaveLength(1);
    expect(result[0]!.file).toBe("a.md");
  });

  test("sanitizes invalid k to default 60", () => {
    const list1 = [makeResult("a.md", 1.0)];
    const result = reciprocalRankFusion([list1], [1], NaN);
    expect(result).toHaveLength(1);
    expect(Number.isFinite(result[0]!.score)).toBe(true);
  });
});
