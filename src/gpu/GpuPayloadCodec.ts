import { constants, inflate, zstdDecompress } from "node:zlib";
import { promisify } from "node:util";
import { assertArtifactReadActive } from "../artifacts/ArtifactFileOperation.js";
import { decodeLz4Block } from "./Lz4Block.js";
import { rejectGpu } from "./GpuSource.js";
const unzip = promisify(inflate);
const unzstd = promisify(zstdDecompress);
export const decodeGpuPayload = async (
  bytes: Buffer,
  compression: "zlib" | "zstd" | "lz4",
  expected: number,
  signal?: AbortSignal,
): Promise<Buffer> => {
  assertArtifactReadActive(signal);
  if (compression === "lz4") return decodeLz4Block(bytes, expected, signal);
  let decoded: Buffer;
  try {
    const maxOutputLength = Math.max(1, expected);
    decoded = await (compression === "zstd"
      ? unzstd(bytes, {
          maxOutputLength,
          params: { [constants.ZSTD_d_windowLogMax]: 27 },
        })
      : unzip(bytes, { maxOutputLength }));
  } catch {
    assertArtifactReadActive(signal);
    return rejectGpu(
      "GPU payload decompression failed or exceeded its declared length",
    );
  }
  assertArtifactReadActive(signal);
  if (decoded.length !== expected)
    rejectGpu("GPU decompressed length disagrees with its declaration");
  return decoded;
};
