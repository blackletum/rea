import { expect, it } from "vitest";
import { writeFile, unlink } from "node:fs/promises";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { toolContract } from "../../src/contracts/toolContracts.js";
import { gpuCodeObjectsSchema } from "../../src/domain/gpu/gpuCodeObjects.js";
import {
  gpuCodeObjectsPageViewResultSchema,
  gpuCodeObjectsSummaryViewResultSchema,
} from "../../src/domain/analysisView/analysisView.js";
import { parseEvidence, createEvidence } from "../../src/domain/evidence.js";
import { createTestTempDirectory } from "../fixtures/temporaryDirectory.js";
import { connectLocalToolsMcp } from "../fixtures/localToolsMcp.js";
import { parseMcpToolError } from "../fixtures/mcpToolError.js";
import { cliTest } from "../support/cli/cliFixture.js";
import {
  cudaFixture,
  ccobFixture,
  ptxFixture,
} from "../fixtures/gpu/codeObjects.js";

cliTest(
  "reads CUDA/PTX and compressed HIP through the public CLI",
  async ({ cli }) => {
    const root = await createTestTempDirectory("rea-gpu-cli-");
    for (const [name, bytes] of [
      ["cuda", cudaFixture("zstd")],
      ["hip", ccobFixture()],
    ] as const) {
      const path = join(root, name);
      await writeFile(path, bytes);
      const response = await cli.run({
        arguments: [
          "inspect-gpu-code-objects",
          path,
          "--analysis",
          "kernels",
          "--detail",
          "complete",
          "--json",
        ],
      });
      expect(response.exitCode, response.stderr).toBe(0);
      const evidence = parseEvidence(
        toolContract("inspect_gpu_code_objects").outputSchema.parse(
          response.json,
        ),
      );
      const report = gpuCodeObjectsSchema.parse(evidence.normalized_result);
      expect(report.artifact.sha256).toBe(
        createHash("sha256").update(bytes).digest("hex"),
      );
      expect(report.kernels).toHaveLength(name === "cuda" ? 2 : 1);
      expect(report.kernels[0]!.parameter_bytes).toBe(12);
      expect(report.coverage.payloads).toBe("complete");
    }
  },
);

it("retains selected kernel facts and complete PTX for historical MCP pages after source removal", async () => {
  const root = await createTestTempDirectory("rea-gpu-mcp-"),
    path = join(root, "application");
  await writeFile(path, cudaFixture("zstd"));
  const { call } = await connectLocalToolsMcp();
  const inventoryReply = await call("inspect_gpu_code_objects", {
    path,
    view: { collection: "objects", fields: ["kind", "target"] },
  });
  expect(inventoryReply.isError, JSON.stringify(inventoryReply)).not.toBe(true);
  const inventory = gpuCodeObjectsPageViewResultSchema.parse(
    parseEvidence(inventoryReply.structuredContent).normalized_result,
  );
  expect(inventory.items).toHaveLength(2);
  expect(inventory.items[0]).not.toHaveProperty("sha256");
  expect(inventory.items[0]).toHaveProperty("metadata_location");
  const targetsReply = await call("inspect_analysis_view", {
    source: {
      kind: "retained-evidence",
      evidence_id: inventory.parent_evidence_id,
    },
    view: { kind: "gpu-code-objects", collection: "targets" },
  });
  expect(targetsReply.isError, JSON.stringify(targetsReply)).not.toBe(true);
  const targets = gpuCodeObjectsPageViewResultSchema.parse(
    parseEvidence(targetsReply.structuredContent).normalized_result,
  );
  expect(targets.items).toHaveLength(2);
  expect(targets.items).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        target: "sm_80",
        object_count: 1,
        kernel_count: 0,
        kernel_coverage: "not-requested",
      }),
      expect.objectContaining({
        target: "sm_90",
        object_count: 1,
        kernel_count: 0,
        kernel_coverage: "not-requested",
      }),
    ]),
  );
  const inspectedReply = await call("inspect_gpu_code_objects", {
    path,
    analysis: "kernels",
    view: {
      collection: "kernels",
      names: ["add_one"],
      fields: ["parameter_count", "registers"],
    },
  });
  expect(inspectedReply.isError, JSON.stringify(inspectedReply)).not.toBe(true);
  const inspected = gpuCodeObjectsPageViewResultSchema.parse(
    parseEvidence(inspectedReply.structuredContent).normalized_result,
  );
  expect(inspected.items).toHaveLength(2);
  expect(inspected.items[0]).not.toHaveProperty("parameters");
  expect(inspected.items[0]).toHaveProperty("sources");
  const objectId =
    "object_id" in inspected.items[0]!
      ? inspected.items[0]!.object_id
      : undefined;
  expect(objectId).toBeTypeOf("string");
  await unlink(path);
  const source = {
    kind: "retained-evidence",
    evidence_id: inspected.parent_evidence_id,
  };
  const textReply = await call("inspect_analysis_view", {
    source,
    view: {
      kind: "gpu-code-objects",
      collection: "ptx",
      object_id: objectId,
      offset: 0,
      limit: 4096,
    },
  });
  expect(textReply.isError, JSON.stringify(textReply)).not.toBe(true);
  const text = gpuCodeObjectsPageViewResultSchema.parse(
    parseEvidence(textReply.structuredContent).normalized_result,
  );
  expect(text.ptx!.text).toBe(ptxFixture);
  expect(text.ptx!.sha256).toBe(
    createHash("sha256").update(ptxFixture).digest("hex"),
  );
  const summaryReply = await call("inspect_analysis_view", {
    source,
    view: { kind: "summary" },
  });
  expect(summaryReply.isError, JSON.stringify(summaryReply)).not.toBe(true);
  const summary = gpuCodeObjectsSummaryViewResultSchema.parse(
    parseEvidence(summaryReply.structuredContent).normalized_result,
  );
  expect(summary.summary.counts).toMatchObject({
    objects: 2,
    kernels: 2,
    ptx_artifacts: 1,
  });
  expect(summary.summary).not.toHaveProperty("ptx_artifacts");
});

it("rejects forged duplicate object IDs and mismatched Evidence subjects through MCP", async () => {
  const root = await createTestTempDirectory("rea-gpu-import-"),
    path = join(root, "cuda");
  await writeFile(path, cudaFixture());
  const { call } = await connectLocalToolsMcp();
  const reply = await call("inspect_gpu_code_objects", {
    path,
    analysis: "kernels",
    detail: "complete",
  });
  expect(reply.isError, JSON.stringify(reply)).not.toBe(true);
  const parent = parseEvidence(reply.structuredContent),
    report = gpuCodeObjectsSchema.parse(parent.normalized_result);
  report.objects[1]!.id = report.objects[0]!.id;
  const forged = createEvidence(undefined, parent.provider, {
    operation: "inspect_gpu_code_objects",
    parameters: parent.parameters,
    result: report,
  });
  const failure = parseMcpToolError(
    await call("inspect_analysis_view", {
      source: { kind: "inline", evidence: forged },
      view: { kind: "gpu-code-objects", collection: "objects" },
    }),
  );
  expect(failure.error).toMatchObject({ code: "invalid_request" });
  const mismatched = createEvidence(
    { path: join(root, "other-file"), sha256: "f".repeat(64), format: "file" },
    parent.provider,
    {
      operation: parent.operation,
      parameters: parent.parameters,
      result: parent.normalized_result,
    },
  );
  const identityFailure = parseMcpToolError(
    await call("inspect_analysis_view", {
      source: { kind: "inline", evidence: mismatched },
      view: { kind: "summary" },
    }),
  );
  expect(identityFailure.error).toMatchObject({ code: "invalid_request" });
});
