/**
 * Stdin/stdout test harness for hooks and CLI.
 */

import { $ } from "bun";

export interface RunResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

/** Run a hook handler via the CLI with stdin input. */
export async function runHook(hookName: string, input: string | object, cwd?: string): Promise<RunResult> {
  const stdin = typeof input === "string" ? input : JSON.stringify(input);
  const proc = Bun.spawn(["bun", "src/clawmem.ts", "hook", hookName], {
    cwd: cwd || process.cwd(),
    stdin: new Response(stdin),
    stdout: "pipe",
    stderr: "pipe",
  });

  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  const exitCode = await proc.exited;

  return { stdout, stderr, exitCode };
}

/** Run a CLI subcommand. */
export async function runCli(args: string[], cwd?: string): Promise<RunResult> {
  const proc = Bun.spawn(["bun", "src/clawmem.ts", ...args], {
    cwd: cwd || process.cwd(),
    stdout: "pipe",
    stderr: "pipe",
  });

  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  const exitCode = await proc.exited;

  return { stdout, stderr, exitCode };
}
