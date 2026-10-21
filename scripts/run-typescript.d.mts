export function selectTypeScriptCompiler(
  platform: string,
  arch: string,
  optionalDependencies: Readonly<Record<string, string>>,
): "tsrs" | "typescript";
