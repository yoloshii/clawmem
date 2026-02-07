/**
 * Ephemeral SQLite database for testing.
 */

import Database from "bun:sqlite";
import { mkdtempSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

export interface TempDb {
  db: Database;
  path: string;
  cleanup: () => void;
}

export function createTempDb(): TempDb {
  const dir = mkdtempSync(join(tmpdir(), "clawmem-test-"));
  const dbPath = join(dir, "test.db");
  const db = new Database(dbPath);

  return {
    db,
    path: dbPath,
    cleanup: () => {
      try {
        db.close();
      } catch { /* already closed */ }
      try {
        rmSync(dir, { recursive: true, force: true });
      } catch { /* best effort */ }
    },
  };
}
