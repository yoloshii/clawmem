import { describe, test, expect } from "bun:test";
import { splitDocument, splitObservation } from "../../src/splitter.ts";
import { MAX_FRAGMENTS_PER_DOC, MAX_SPLITTER_INPUT_CHARS } from "../../src/limits.ts";

describe("splitDocument fragment limits", () => {
  test("short document returns only full fragment", () => {
    const result = splitDocument("Short text");
    expect(result).toHaveLength(1);
    expect(result[0]!.type).toBe("full");
  });

  test("normal document returns bounded fragments", () => {
    const sections = Array.from({ length: 20 }, (_, i) =>
      `## Section ${i}\n\nThis is section ${i} with enough content to be a valid fragment that exceeds the minimum character threshold.`
    ).join("\n\n");

    const result = splitDocument(sections);
    expect(result.length).toBeGreaterThan(1);
    expect(result.length).toBeLessThanOrEqual(MAX_FRAGMENTS_PER_DOC);
  });

  test("very large document input is bounded", () => {
    const huge = "## Heading\n\n" + "x".repeat(MAX_SPLITTER_INPUT_CHARS + 10000);
    const result = splitDocument(huge);
    // Full fragment content should be bounded
    expect(result[0]!.content.length).toBeLessThanOrEqual(MAX_SPLITTER_INPUT_CHARS);
  });

  test("fragment count never exceeds MAX_FRAGMENTS_PER_DOC", () => {
    // Generate a document with many sections
    const sections = Array.from({ length: 600 }, (_, i) =>
      `## Section ${i}\n\nContent for section ${i} that is long enough to be a valid fragment over fifty chars.`
    ).join("\n\n");

    const result = splitDocument(sections);
    expect(result.length).toBeLessThanOrEqual(MAX_FRAGMENTS_PER_DOC);
  });

  test("includes multiple fragment types", () => {
    const doc = `## Introduction

This is the introduction section with enough text to pass the minimum threshold.

## Code Example

\`\`\`typescript
function hello() {
  console.log("world");
  // more lines to pass min chars threshold
  return true;
}
\`\`\`

## Features

- Feature one with details about it
- Feature two with more details here
- Feature three to make a list
`;

    const result = splitDocument(doc);
    const types = new Set(result.map(f => f.type));
    expect(types.has("full")).toBe(true);
    expect(types.has("section")).toBe(true);
  });
});

describe("splitObservation fragment limits", () => {
  test("returns full fragment for body", () => {
    const result = splitObservation("Some observation body text", {});
    expect(result).toHaveLength(1);
    expect(result[0]!.type).toBe("full");
  });

  test("extracts facts as fragments", () => {
    const facts = JSON.stringify([
      "This is a fact that is long enough to pass the minimum character threshold of fifty characters.",
      "Short",
      "Another fact that also passes the minimum character threshold for fragment extraction to work properly.",
    ]);
    const result = splitObservation("Body text", { facts });
    // full + 2 facts (one too short)
    expect(result.length).toBe(3);
    expect(result.filter(f => f.type === "fact")).toHaveLength(2);
  });

  test("bounds large input", () => {
    const huge = "x".repeat(MAX_SPLITTER_INPUT_CHARS + 10000);
    const result = splitObservation(huge, {});
    expect(result[0]!.content.length).toBeLessThanOrEqual(MAX_SPLITTER_INPUT_CHARS);
  });

  test("caps fragment count", () => {
    const manyFacts = Array.from({ length: 600 }, (_, i) =>
      `Fact number ${i} with enough content to pass the minimum character threshold for fragment extraction.`
    );
    const result = splitObservation("Body", { facts: JSON.stringify(manyFacts) });
    expect(result.length).toBeLessThanOrEqual(MAX_FRAGMENTS_PER_DOC);
  });
});
