import { describe, test, expect } from "bun:test";

describe("smoke tests", () => {
  test("bun test runner works", () => {
    expect(1 + 1).toBe(2);
  });

  test("imports resolve correctly", async () => {
    const { ClawMemError } = await import("../src/errors.ts");
    expect(new ClawMemError("TEST", "test")).toBeTruthy();
  });

  test("limits module loads", async () => {
    const limits = await import("../src/limits.ts");
    expect(limits.MAX_QUERY_LENGTH).toBeGreaterThan(0);
    expect(limits.MAX_LLM_INPUT_CHARS).toBeGreaterThan(0);
  });

  test("validation module loads", async () => {
    const { assertNonEmptyString } = await import("../src/validation.ts");
    expect(() => assertNonEmptyString("test", "val")).not.toThrow();
  });
});
