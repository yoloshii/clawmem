import { describe, test, expect } from "bun:test";
import { parseObservationXml, parseSummaryXml } from "../../src/observer.ts";

describe("parseObservationXml", () => {
  test("parses valid observation", () => {
    const xml = `
      <type>decision</type>
      <title>Use SQLite for storage</title>
      <facts>
        <fact>SQLite is embedded and requires no server</fact>
        <fact>Better-sqlite3 is synchronous and fast</fact>
      </facts>
      <narrative>Chose SQLite because it needs no external dependencies.</narrative>
      <concepts>
        <concept>trade-off</concept>
      </concepts>
      <files_read><file>src/store.ts</file></files_read>
      <files_modified><file>src/store.ts</file></files_modified>
    `;
    const obs = parseObservationXml(xml);
    expect(obs).not.toBeNull();
    expect(obs!.type).toBe("decision");
    expect(obs!.title).toBe("Use SQLite for storage");
    expect(obs!.facts).toHaveLength(2);
    expect(obs!.narrative).toContain("SQLite");
    expect(obs!.concepts).toContain("trade-off");
    expect(obs!.filesRead).toContain("src/store.ts");
    expect(obs!.filesModified).toContain("src/store.ts");
  });

  test("returns null for missing type", () => {
    const xml = `<title>No type</title>`;
    expect(parseObservationXml(xml)).toBeNull();
  });

  test("returns null for invalid type", () => {
    const xml = `<type>invalid_type</type><title>Test</title>`;
    expect(parseObservationXml(xml)).toBeNull();
  });

  test("returns null for missing title", () => {
    const xml = `<type>decision</type>`;
    expect(parseObservationXml(xml)).toBeNull();
  });

  test("truncates long titles to 80 chars", () => {
    const longTitle = "A".repeat(200);
    const xml = `<type>decision</type><title>${longTitle}</title>`;
    const obs = parseObservationXml(xml);
    expect(obs).not.toBeNull();
    expect(obs!.title.length).toBeLessThanOrEqual(80);
  });

  test("filters short facts", () => {
    const xml = `
      <type>decision</type>
      <title>Test</title>
      <facts><fact>abc</fact><fact>This is a longer valid fact string</fact></facts>
    `;
    const obs = parseObservationXml(xml);
    expect(obs!.facts).toHaveLength(1);
    expect(obs!.facts[0]).toContain("longer");
  });

  test("filters invalid concepts", () => {
    const xml = `
      <type>decision</type>
      <title>Test</title>
      <concepts><concept>trade-off</concept><concept>not-a-concept</concept></concepts>
    `;
    const obs = parseObservationXml(xml);
    expect(obs!.concepts).toEqual(["trade-off"]);
  });

  test("handles empty/malformed XML gracefully", () => {
    expect(parseObservationXml("")).toBeNull();
    expect(parseObservationXml("not xml at all")).toBeNull();
    expect(parseObservationXml("<unclosed>")).toBeNull();
  });
});

describe("parseSummaryXml", () => {
  test("parses valid summary", () => {
    const xml = `
      <request>User asked to implement auth</request>
      <investigated>Looked at JWT libraries</investigated>
      <learned>bcrypt is preferred over sha256</learned>
      <completed>Implemented login endpoint</completed>
      <next_steps>Add refresh token support</next_steps>
    `;
    const summary = parseSummaryXml(xml);
    expect(summary).not.toBeNull();
    expect(summary!.request).toContain("auth");
    expect(summary!.completed).toContain("login");
    expect(summary!.nextSteps).toContain("refresh");
  });

  test("returns null when both request and completed missing", () => {
    const xml = `<investigated>Some investigation</investigated>`;
    expect(parseSummaryXml(xml)).toBeNull();
  });

  test("fills defaults for missing fields", () => {
    const xml = `<request>Do something</request>`;
    const summary = parseSummaryXml(xml);
    expect(summary).not.toBeNull();
    expect(summary!.investigated).toBe("None");
    expect(summary!.learned).toBe("None");
    expect(summary!.completed).toBe("None");
    expect(summary!.nextSteps).toBe("None");
  });

  test("handles empty XML", () => {
    expect(parseSummaryXml("")).toBeNull();
  });
});
