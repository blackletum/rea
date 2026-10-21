import {
  BinaryReader,
  BinaryResourceFailure,
} from "../native/metadata/BinaryContainer.js";
import { rejectGpu } from "./GpuSource.js";
export interface CudaInfoRecord {
  readonly format: number;
  readonly attribute: number;
  readonly value: number;
  readonly offset: number;
  readonly length: number;
  readonly data: Buffer;
}
/** CUDA nv.info is a numeric record format, not a string-keyed metadata map. */
export function* readCudaInfo(
  bytes: Buffer,
  little: boolean,
): Generator<CudaInfoRecord> {
  const reader = new BinaryReader(bytes, little);
  let records = 0;
  for (let offset = 0; offset < bytes.length;) {
    if (++records > 262144)
      throw new BinaryResourceFailure(
        "structure",
        262144,
        "CUDA nv.info record count exceeds the metadata allocation budget",
      );
    const head = reader.range(offset, 4, "CUDA nv.info record");
    const format = head[0]!,
      attribute = head[1]!,
      value = reader.u16(offset + 2);
    if (format < 1 || format > 4)
      rejectGpu("Unknown CUDA nv.info record format");
    const length = format === 4 ? value : 0;
    yield {
      format,
      attribute,
      value,
      offset,
      length: 4 + length,
      data: reader.range(offset + 4, length, "CUDA nv.info value"),
    };
    offset += 4 + length;
  }
}

/** Kernel argument interpretation owns duplicate and range semantics at the numeric-record boundary. */
export const readCudaParameters = (bytes: Buffer, little: boolean) => {
  const parameters: import("../domain/gpu/gpuCodeObjects.js").GpuKernel["parameters"] =
    [];
  const byteDeclarations = new Set<number>();
  const limitations: string[] = [];
  let complete = true;
  for (const record of readCudaInfo(bytes, little)) {
    if (record.format === 3 && record.attribute === 0x19)
      byteDeclarations.add(record.value);
    if (record.format !== 4 || record.attribute !== 0x17) continue;
    if (record.data.length !== 12) {
      complete = false;
      continue;
    }
    const r = new BinaryReader(record.data, little);
    if (r.u32(0) !== 0) complete = false;
    parameters.push({
      ordinal: r.u16(4),
      name: null,
      offset: r.u16(6),
      size: r.u32(8) >>> 18,
      type: null,
      hidden: null,
    });
  }
  parameters.sort((a, b) => a.ordinal - b.ordinal);
  if (parameters.some((row, index) => row.ordinal !== index)) complete = false;
  const parameter_bytes =
    byteDeclarations.size === 1 ? [...byteDeclarations][0]! : null;
  if (byteDeclarations.size > 1)
    limitations.push(
      `Conflicting CUDA argument-area byte declarations: ${[...byteDeclarations].join(", ")}; parameter_bytes remains unknown`,
    );
  if (
    parameter_bytes !== null &&
    parameters.some(
      (row) =>
        row.offset! > parameter_bytes ||
        row.size! > parameter_bytes - row.offset!,
    )
  )
    limitations.push(
      "CUDA parameter ranges disagree with the declared argument-area length; declarations do not establish a valid ABI layout",
    );
  if (!complete)
    limitations.push(
      "CUDA parameter records have unsupported indices/sizes, duplicate ordinals or gaps; parameter count is unknown",
    );
  return {
    parameter_count:
      complete && (parameters.length > 0 || parameter_bytes === 0)
        ? parameters.length
        : null,
    parameter_bytes,
    parameters,
    limitations,
  };
};
