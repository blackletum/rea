import { expect, it } from "vitest";
import { GpuInventory } from "./GpuInventory.js";
import { GpuBufferSource } from "./GpuSource.js";
it("observes cancellation while walking flat empty bundles in a decoded buffer", async () => {
  const empty = Buffer.alloc(32);
  empty.write("__CLANG_OFFLOAD_BUNDLE__");
  const bytes = Buffer.concat(Array.from({ length: 4096 }, () => empty));
  const controller = new AbortController(),
    reason = { selected: "cancel-buffer-walk" };
  const inventory = new GpuInventory(
    { path: "/fixture", analysis: "inventory" },
    controller.signal,
  );
  const result = inventory.inspect(new GpuBufferSource(bytes, "fixture"));
  setImmediate(() => controller.abort(reason));
  await expect(result).rejects.toMatchObject({
    reason: "cancelled",
    cause: reason,
  });
});
