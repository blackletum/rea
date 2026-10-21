import { BinaryReader } from "../native/metadata/BinaryContainer.js";
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
export const readCudaInfo = (
  bytes: Buffer,
  little: boolean,
): CudaInfoRecord[] => {
  const reader = new BinaryReader(bytes, little),
    records: CudaInfoRecord[] = [];
  for (let offset = 0; offset < bytes.length;) {
    const head = reader.range(offset, 4, "CUDA nv.info record");
    const format = head[0]!,
      attribute = head[1]!,
      value = reader.u16(offset + 2);
    if (format < 1 || format > 4)
      rejectGpu("Unknown CUDA nv.info record format");
    const length = format === 4 ? value : 0;
    records.push({
      format,
      attribute,
      value,
      offset,
      length: 4 + length,
      data: reader.range(offset + 4, length, "CUDA nv.info value"),
    });
    offset += 4 + length;
  }
  return records;
};
