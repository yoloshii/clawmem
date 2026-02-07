import { describe, test, expect } from "bun:test";
import { ClawMemError, toUserError, toErrorResponse } from "../../src/errors.ts";

describe("ClawMemError", () => {
  test("creates with code and message", () => {
    const err = new ClawMemError("INVALID_INPUT", "bad input");
    expect(err.code).toBe("INVALID_INPUT");
    expect(err.message).toBe("bad input");
    expect(err.name).toBe("ClawMemError");
  });

  test("includes optional details", () => {
    const err = new ClawMemError("TOO_LONG", "too long", { max: 100, actual: 200 });
    expect(err.details?.max).toBe(100);
  });

  test("includes optional cause", () => {
    const cause = new Error("root cause");
    const err = new ClawMemError("INTERNAL", "failed", undefined, cause);
    expect(err.cause).toBe(cause);
  });

  test("serializes to JSON with consistent shape", () => {
    const err = new ClawMemError("TEST", "test error", { key: "val" });
    const json = err.toJSON();
    expect(json.ok).toBe(false);
    expect(json.error.code).toBe("TEST");
    expect(json.error.message).toBe("test error");
    expect(json.error.details?.key).toBe("val");
  });

  test("serializes without details when not provided", () => {
    const err = new ClawMemError("TEST", "simple");
    const json = err.toJSON();
    expect(json.error.details).toBeUndefined();
  });
});

describe("toUserError", () => {
  test("formats ClawMemError", () => {
    const err = new ClawMemError("BAD", "bad thing");
    expect(toUserError(err)).toBe("[BAD] bad thing");
  });

  test("formats regular Error", () => {
    expect(toUserError(new Error("oops"))).toBe("oops");
  });

  test("formats non-Error", () => {
    expect(toUserError("string error")).toBe("string error");
    expect(toUserError(42)).toBe("42");
  });
});

describe("toErrorResponse", () => {
  test("formats ClawMemError to JSON", () => {
    const err = new ClawMemError("INVALID", "bad");
    const resp = toErrorResponse(err);
    expect(resp.ok).toBe(false);
    expect(resp.error.code).toBe("INVALID");
  });

  test("formats regular Error", () => {
    const resp = toErrorResponse(new Error("generic"));
    expect(resp.ok).toBe(false);
    expect(resp.error.code).toBe("INTERNAL_ERROR");
    expect(resp.error.message).toBe("generic");
  });

  test("formats non-Error", () => {
    const resp = toErrorResponse("string");
    expect(resp.ok).toBe(false);
    expect(resp.error.code).toBe("INTERNAL_ERROR");
  });
});
