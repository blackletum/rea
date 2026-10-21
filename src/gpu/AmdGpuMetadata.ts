import { decode } from "@msgpack/msgpack";
import { BinaryResourceFailure } from "../native/metadata/BinaryContainer.js";
import { rejectGpu } from "./GpuSource.js";

/** Validate allocation and nesting costs before the standard decoder allocates containers. */
const preflight = (bytes: Buffer): void => {
  let at = 0,
    values = 0;
  const take = (size: number): number => {
    const start = at;
    if (size > bytes.length - at)
      rejectGpu("Truncated AMDGPU MessagePack metadata");
    at += size;
    return start;
  };
  const value = (depth: number): void => {
    if (depth > 32 || ++values > 262144)
      throw new BinaryResourceFailure(
        "structure",
        262144,
        "AMDGPU metadata exceeds depth 32 or 262144 decoded values",
      );
    const code = bytes[take(1)]!;
    let count = 0,
      payload = 0;
    if (
      code <= 0x7f ||
      code >= 0xe0 ||
      code === 0xc0 ||
      code === 0xc2 ||
      code === 0xc3
    )
      return;
    if ((code & 0xe0) === 0xa0) payload = code & 31;
    else if ((code & 0xf0) === 0x90) count = code & 15;
    else if ((code & 0xf0) === 0x80) count = (code & 15) * 2;
    else if (code === 0xdc || code === 0xde)
      count = bytes.readUInt16BE(take(2)) * (code === 0xde ? 2 : 1);
    else if (code === 0xdd || code === 0xdf)
      count = bytes.readUInt32BE(take(4)) * (code === 0xdf ? 2 : 1);
    else if (code === 0xd9 || code === 0xc4) payload = bytes[take(1)]!;
    else if (code === 0xda || code === 0xc5)
      payload = bytes.readUInt16BE(take(2));
    else if (code === 0xdb || code === 0xc6)
      payload = bytes.readUInt32BE(take(4));
    else {
      const widths: Readonly<Record<number, number>> = {
        0xca: 4,
        0xcb: 8,
        0xcc: 1,
        0xcd: 2,
        0xce: 4,
        0xcf: 8,
        0xd0: 1,
        0xd1: 2,
        0xd2: 4,
        0xd3: 8,
      };
      const width = widths[code];
      if (width === undefined)
        rejectGpu("Unsupported AMDGPU MessagePack extension or token");
      payload = width!;
    }
    if (count > 262144 - values || payload > 65536)
      throw new BinaryResourceFailure(
        "structure",
        262144,
        "AMDGPU metadata container/string exceeds its allocation budget",
      );
    take(payload);
    for (let index = 0; index < count; index++) value(depth + 1);
  };
  value(0);
  if (at !== bytes.length) rejectGpu("Trailing AMDGPU MessagePack values");
};
export const readAmdMetadata = (bytes: Buffer): unknown => {
  if (bytes.length > 8 * 1024 * 1024)
    throw new BinaryResourceFailure(
      "structure",
      8 * 1024 * 1024,
      "AMDGPU metadata exceeds 8 MiB",
    );
  preflight(bytes);
  try {
    return decode(bytes, {
      useBigInt64: true,
      maxStrLength: 65536,
      maxBinLength: 65536,
      maxArrayLength: 65536,
      maxMapLength: 65536,
      maxExtLength: 0,
    });
  } catch {
    return rejectGpu("Malformed AMDGPU MessagePack metadata");
  }
};
export const amdRecord = (value: unknown): Record<string, unknown> | null =>
  typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
export const amdCount = (value: unknown): number | null => {
  if (typeof value === "bigint")
    return value >= 0n && value <= BigInt(Number.MAX_SAFE_INTEGER)
      ? Number(value)
      : null;
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? value
    : null;
};
export const amdString = (value: unknown): string | null =>
  typeof value === "string" ? value : null;
