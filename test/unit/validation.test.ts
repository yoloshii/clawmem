import { describe, test, expect } from "bun:test";
import {
  assertNonEmptyString,
  assertMaxLength,
  assertFiniteNumber,
  assertBounds,
  assertArrayLengthMatch,
  assertSafePath,
  clampBounds,
} from "../../src/validation.ts";

describe("assertNonEmptyString", () => {
  test("accepts valid string", () => {
    expect(() => assertNonEmptyString("hello", "test")).not.toThrow();
  });

  test("rejects empty string", () => {
    expect(() => assertNonEmptyString("", "test")).toThrow("non-empty string");
  });

  test("rejects non-string", () => {
    expect(() => assertNonEmptyString(123 as any, "test")).toThrow("non-empty string");
  });

  test("rejects null", () => {
    expect(() => assertNonEmptyString(null as any, "test")).toThrow();
  });
});

describe("assertMaxLength", () => {
  test("accepts within limit", () => {
    expect(() => assertMaxLength("short", 100, "test")).not.toThrow();
  });

  test("rejects over limit", () => {
    expect(() => assertMaxLength("x".repeat(200), 100, "test")).toThrow("exceeds max length");
  });

  test("accepts exact limit", () => {
    expect(() => assertMaxLength("x".repeat(100), 100, "test")).not.toThrow();
  });
});

describe("assertFiniteNumber", () => {
  test("accepts finite number", () => {
    expect(() => assertFiniteNumber(42, "test")).not.toThrow();
  });

  test("rejects NaN", () => {
    expect(() => assertFiniteNumber(NaN, "test")).toThrow("finite number");
  });

  test("rejects Infinity", () => {
    expect(() => assertFiniteNumber(Infinity, "test")).toThrow("finite number");
  });

  test("rejects string", () => {
    expect(() => assertFiniteNumber("42" as any, "test")).toThrow("finite number");
  });
});

describe("assertBounds", () => {
  test("accepts within bounds", () => {
    expect(() => assertBounds(5, 0, 10, "test")).not.toThrow();
  });

  test("accepts min boundary", () => {
    expect(() => assertBounds(0, 0, 10, "test")).not.toThrow();
  });

  test("accepts max boundary", () => {
    expect(() => assertBounds(10, 0, 10, "test")).not.toThrow();
  });

  test("rejects below min", () => {
    expect(() => assertBounds(-1, 0, 10, "test")).toThrow("between");
  });

  test("rejects above max", () => {
    expect(() => assertBounds(11, 0, 10, "test")).toThrow("between");
  });

  test("rejects NaN", () => {
    expect(() => assertBounds(NaN, 0, 10, "test")).toThrow("between");
  });

  test("rejects Infinity", () => {
    expect(() => assertBounds(Infinity, 0, 10, "test")).toThrow("between");
  });
});

describe("assertArrayLengthMatch", () => {
  test("accepts matching lengths", () => {
    expect(() => assertArrayLengthMatch([1, 2], [3, 4], "a", "b")).not.toThrow();
  });

  test("rejects mismatched lengths", () => {
    expect(() => assertArrayLengthMatch([1], [1, 2], "a", "b")).toThrow("must match");
  });
});

describe("assertSafePath", () => {
  test("accepts valid paths", () => {
    expect(() => assertSafePath("src/store.ts")).not.toThrow();
    expect(() => assertSafePath("deeply/nested/path/to/file.md")).not.toThrow();
  });

  test("rejects path traversal", () => {
    expect(() => assertSafePath("../etc/passwd")).toThrow("traversal");
    expect(() => assertSafePath("sub/../../escape.md")).toThrow("traversal");
  });

  test("rejects null bytes", () => {
    expect(() => assertSafePath("file\0.md")).toThrow("null bytes");
  });

  test("rejects very long paths", () => {
    expect(() => assertSafePath("a".repeat(2000))).toThrow("max length");
  });
});

describe("clampBounds", () => {
  test("clamps below min", () => {
    expect(clampBounds(-5, 0, 100)).toBe(0);
  });

  test("clamps above max", () => {
    expect(clampBounds(200, 0, 100)).toBe(100);
  });

  test("passes through valid value", () => {
    expect(clampBounds(50, 0, 100)).toBe(50);
  });

  test("returns min for NaN", () => {
    expect(clampBounds(NaN, 0, 100)).toBe(0);
  });

  test("returns min for Infinity", () => {
    expect(clampBounds(Infinity, 0, 100)).toBe(0);
  });

  test("returns min for -Infinity", () => {
    expect(clampBounds(-Infinity, 0, 100)).toBe(0);
  });
});
