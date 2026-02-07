/**
 * ClawMem File Watcher - fs.watch with debounce for incremental reindex
 */

import { watch, type WatchEventType } from "fs";
import { shouldExclude } from "./indexer.ts";

export type WatcherOptions = {
  debounceMs?: number;
  onChanged: (path: string, event: WatchEventType) => Promise<void>;
  onError?: (error: Error) => void;
};

export function startWatcher(
  directories: string[],
  options: WatcherOptions
): { close: () => void } {
  const { debounceMs = 2000, onChanged, onError } = options;
  const pending = new Map<string, ReturnType<typeof setTimeout>>();
  const watchers: ReturnType<typeof watch>[] = [];

  for (const dir of directories) {
    try {
      const watcher = watch(dir, { recursive: true }, (event, filename) => {
        if (!filename) return;
        if (!filename.endsWith(".md") && !filename.endsWith(".jsonl")) return;
        if (shouldExclude(filename)) return;

        const fullPath = `${dir}/${filename}`;
        const existing = pending.get(fullPath);
        if (existing) clearTimeout(existing);

        pending.set(fullPath, setTimeout(async () => {
          pending.delete(fullPath);
          try {
            await onChanged(fullPath, event);
          } catch (err) {
            onError?.(err instanceof Error ? err : new Error(String(err)));
          }
        }, debounceMs));
      });
      watchers.push(watcher);
    } catch (err) {
      onError?.(err instanceof Error ? err : new Error(`Failed to watch ${dir}: ${err}`));
    }
  }

  return {
    close: () => {
      for (const w of watchers) w.close();
      for (const t of pending.values()) clearTimeout(t);
      pending.clear();
    },
  };
}
