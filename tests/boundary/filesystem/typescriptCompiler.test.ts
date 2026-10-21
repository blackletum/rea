import { spawnSync } from "node:child_process";
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { expect, it } from "vitest";
import { z } from "zod";

import { selectTypeScriptCompiler } from "../../../scripts/run-typescript.mjs";
import { createTestTempDirectory } from "../../fixtures/temporaryDirectory.js";

const repository = fileURLToPath(new URL("../../../", import.meta.url));
const launcher = join(repository, "scripts/run-typescript.mjs");
const require = createRequire(import.meta.url);
const rustPackage = z
  .object({ optionalDependencies: z.record(z.string(), z.string()) })
  .parse(
    JSON.parse(await readFile(require.resolve("tsrs/package.json"), "utf8")),
  );
const execute = (script: string, args: string[]) =>
  spawnSync(process.execPath, [script, ...args], {
    encoding: "utf8",
    timeout: 15_000,
  });

it.each([
  ["linux", "x64", "tsrs"],
  ["linux", "arm64", "tsrs"],
  ["darwin", "arm64", "tsrs"],
  ["win32", "x64", "typescript"],
  ["win32", "arm64", "typescript"],
  ["darwin", "x64", "typescript"],
  ["android", "arm64", "typescript"],
  ["linux", "riscv64", "typescript"],
])(
  "selects %s/%s compiler %s from the pinned inventory",
  (platform, arch, compiler) => {
    expect(
      selectTypeScriptCompiler(
        platform,
        arch,
        rustPackage.optionalDependencies,
      ),
    ).toBe(compiler);
  },
);

it("always retains the official compiler on Windows", () => {
  expect(
    selectTypeScriptCompiler("win32", "x64", { "@tsrs/win32-x64": "future" }),
  ).toBe("typescript");
});

it("runs the selected installed compiler", () => {
  const result = execute(launcher, ["--version"]);
  expect(result.error).toBeUndefined();
  expect(result.status, result.stderr).toBe(0);
  const compiler = selectTypeScriptCompiler(
    process.platform,
    process.arch,
    rustPackage.optionalDependencies,
  );
  expect(result.stdout).toContain(
    compiler === "tsrs" ? "tsrs 0.18.0" : "Version 7.0.2",
  );
});

it("forwards build arguments and preserves real type errors and exit status", async () => {
  const root = await createTestTempDirectory("rea-typescript-compiler-");
  const input = join(root, "input.ts");
  await writeFile(
    join(root, "tsconfig.json"),
    JSON.stringify({
      compilerOptions: {
        strict: true,
        target: "ES2023",
        types: [],
        outDir: "output",
      },
      files: ["input.ts"],
    }),
  );
  await writeFile(input, "export const value: number = 1;\n");
  const valid = execute(launcher, ["-p", root, "--pretty", "false"]);
  expect(valid.error).toBeUndefined();
  expect(valid.status, valid.stdout + valid.stderr).toBe(0);
  expect(await readFile(join(root, "output/input.js"), "utf8")).toContain(
    "value = 1",
  );
  await writeFile(input, 'export const value: number = "invalid";\n');
  const invalid = execute(launcher, [
    "-p",
    root,
    "--noEmit",
    "--pretty",
    "false",
  ]);
  expect(invalid.error).toBeUndefined();
  expect(invalid.status, invalid.stdout + invalid.stderr).toBe(2);
  expect(invalid.stdout + invalid.stderr).toContain("TS2322");
  expect(await readFile(join(root, "output/input.js"), "utf8")).toContain(
    "value = 1",
  );
});

it.skipIf(process.platform === "win32")(
  "does not retry failed supported compilers with the official compiler",
  async () => {
    const root = await createTestTempDirectory("rea-typescript-failure-");
    const script = join(root, "run-typescript.mjs");
    await copyFile(launcher, script);
    for (const name of ["tsrs", "typescript"]) {
      const directory = join(root, "node_modules", name);
      await mkdir(directory, { recursive: true });
      await writeFile(
        join(directory, "package.json"),
        JSON.stringify({
          name,
          type: "module",
          bin: { [name === "tsrs" ? "tsrs" : "tsc"]: "cli.mjs" },
          optionalDependencies: {
            [`@tsrs/${process.platform}-${process.arch}`]: "fixture",
          },
        }),
      );
      await writeFile(
        join(directory, "cli.mjs"),
        name === "tsrs"
          ? 'console.error("supported compiler failed"); process.exitCode = 42;\n'
          : 'console.log("unexpected fallback");\n',
      );
    }
    const result = execute(script, ["--noEmit"]);
    expect(result.error).toBeUndefined();
    expect(result.status, result.stdout + result.stderr).toBe(42);
    expect(result.stderr).toContain("supported compiler failed");
    expect(result.stdout).not.toContain("unexpected fallback");
    // The same selection must expose a missing installed launcher as a failure.
    await writeFile(
      join(root, "node_modules/tsrs/package.json"),
      JSON.stringify({
        name: "tsrs",
        bin: { tsrs: "missing.mjs" },
        optionalDependencies: {
          [`@tsrs/${process.platform}-${process.arch}`]: "fixture",
        },
      }),
    );
    const missing = execute(script, ["--noEmit"]);
    expect(missing.status, missing.stdout + missing.stderr).toBe(1);
    expect(missing.stderr).toContain("ERR_MODULE_NOT_FOUND");
    expect(missing.stdout).not.toContain("unexpected fallback");
  },
);
