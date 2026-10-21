#!/usr/bin/env node

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { availableParallelism } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

/** Use the pinned package's platform inventory; let its launcher select libc. */
export const selectTypeScriptCompiler = (
  platform,
  arch,
  optionalDependencies,
) => {
  if (platform === "win32") return "typescript";
  const prefix = `@tsrs/${platform}-${arch}`;
  return Object.keys(optionalDependencies).some(
    (name) => name === prefix || name.startsWith(`${prefix}-`),
  )
    ? "tsrs"
    : "typescript";
};

const main = async () => {
  const require = createRequire(import.meta.url);
  const rustPackagePath = require.resolve("tsrs/package.json");
  const rustPackage = JSON.parse(readFileSync(rustPackagePath, "utf8"));
  const compiler = selectTypeScriptCompiler(
    process.platform,
    process.arch,
    rustPackage.optionalDependencies,
  );
  const packagePath = require.resolve(`${compiler}/package.json`);
  const metadata = JSON.parse(readFileSync(packagePath, "utf8"));
  const command = compiler === "tsrs" ? "tsrs" : "tsc";
  if (compiler === "tsrs")
    process.argv.splice(
      2,
      0,
      "--checkers",
      String(Math.min(2, availableParallelism())),
      "--maxMemory",
      "3G",
    );
  // Delegate execution, diagnostics, status and signals to the package's launcher.
  // An unavailable binary on a supported host must fail rather than hide drift.
  await import(
    pathToFileURL(resolve(dirname(packagePath), metadata.bin[command])).href
  );
};

if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
