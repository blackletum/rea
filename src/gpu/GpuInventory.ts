import { gpuObject, gpuLocation } from "./GpuObject.js";
import { readCudaFatbinary } from "./CudaFatbinary.js";
import { decodeGpuPayload } from "./GpuPayloadCodec.js";
import { scheduler } from "node:timers/promises";
import { createHash } from "node:crypto";
import type {
  GpuCodeObjectsInput,
  GpuCodeObjects,
  GpuObject,
  GpuKernel,
} from "../domain/gpu/gpuCodeObjects.js";
import {
  BinaryFormatFailure,
  BinaryResourceFailure,
} from "../native/metadata/BinaryContainer.js";
import {
  readElfSectionDirectory,
  readPeSectionDirectory,
} from "../native/metadata/NativeSectionDirectory.js";
import { assertArtifactReadActive } from "../artifacts/ArtifactFileOperation.js";
import { jsonValueByteLength } from "../domain/jsonValueByteLength.js";
import {
  GpuBufferSource,
  GPU_OBJECT_BYTES,
  GPU_REPORT_BYTES,
  gpuCount,
  gpuSha256,
  readGpuNativeDirectory,
  rejectGpu,
  type GpuSource,
} from "./GpuSource.js";
import { readGpuElfKernels } from "./GpuElfKernels.js";
import { readPtxKernels, readPtxTarget } from "./PtxKernels.js";
const CLANG_MAGIC = Buffer.from("__CLANG_OFFLOAD_BUNDLE__");
const selected = (row: GpuObject, input: GpuCodeObjectsInput): boolean =>
  (input.object_ids === undefined || input.object_ids.includes(row.id)) &&
  (input.targets === undefined ||
    (row.target !== null && input.targets.includes(row.target)));

/** File declarations only. Every child range stays in its named parent coordinate space. */
export class GpuInventory {
  readonly objects: GpuObject[] = [];
  readonly kernels: GpuKernel[] = [];
  readonly ptx_artifacts: GpuCodeObjects["ptx_artifacts"] = [];
  readonly limitations = [
    "Static declarations only; no GPU execution, SASS disassembly, host launch correlation or runtime compatibility was verified.",
    "Object IDs are positional identities scoped to artifact.sha256. Locations inside decoded objects name their containing object; they are not file offsets.",
  ];
  discovery: "complete" | "partial" = "complete";
  examined_sections = 0;
  unknown_section_names = 0;
  #factBytes = 0;
  #decodedBytes = 0;
  constructor(
    readonly input: GpuCodeObjectsInput,
    readonly signal?: AbortSignal,
  ) {}
  #retain(
    value: GpuObject | GpuKernel | GpuCodeObjects["ptx_artifacts"][number],
  ): void {
    const bytes = jsonValueByteLength(value, GPU_REPORT_BYTES);
    if (bytes > GPU_REPORT_BYTES - this.#factBytes)
      throw new BinaryResourceFailure(
        "structure",
        GPU_REPORT_BYTES,
        "GPU facts exceed the 16 MiB retained report budget",
      );
    this.#factBytes += bytes;
  }
  #add(row: GpuObject): void {
    this.#retain(row);
    this.objects.push(row);
  }
  #budgetDecoded(size: number): void {
    if (
      size > GPU_OBJECT_BYTES ||
      size > 512 * 1024 * 1024 - this.#decodedBytes
    )
      throw new BinaryResourceFailure(
        "structure",
        GPU_OBJECT_BYTES,
        "GPU decoding exceeds 128 MiB per object or 512 MiB per inspection",
      );
    this.#decodedBytes += size;
  }
  async inspect(
    source: GpuSource,
    depth = 0,
  ): Promise<GpuCodeObjects["format"]> {
    assertArtifactReadActive(this.signal);
    if (depth > 8)
      throw new BinaryResourceFailure(
        "structure",
        8,
        "GPU container nesting exceeds eight levels",
      );
    const prefix = await source.read(0, Math.min(32, source.size));
    if (prefix.subarray(0, 4).equals(Buffer.from([0x50, 0xed, 0x55, 0xba]))) {
      await this.#cuda(source);
      return "cuda-fatbin";
    }
    if (prefix.subarray(0, CLANG_MAGIC.length).equals(CLANG_MAGIC)) {
      await this.#hip(source, depth);
      return "hip-bundle";
    }
    if (prefix.subarray(0, 4).toString("ascii") === "CCOB") {
      await this.#compressedHip(source, depth);
      return "hip-bundle";
    }
    if (
      prefix.subarray(0, 4).equals(Buffer.from([0x7f, 0x45, 0x4c, 0x46])) ||
      prefix.subarray(0, 2).toString("ascii") === "MZ"
    ) {
      const elf = prefix[0] === 0x7f;
      const directory = await readGpuNativeDirectory(
        source,
        elf
          ? readElfSectionDirectory(source.size)
          : readPeSectionDirectory(source.size),
      );
      if (elf && (directory.machine === 190 || directory.machine === 224)) {
        const row = gpuObject(source, {
          offset: 0,
          size: source.size,
          kind: directory.machine === 190 ? "cubin" : "amdgpu",
        });
        if (directory.machine === 190)
          row.target = `sm_${directory.flags & 0xff}`;
        this.#add(row);
        await this.#payload(source, row);
      } else {
        for (const section of directory.sections) {
          this.examined_sections++;
          if (section.name === null) {
            this.unknown_section_names++;
            this.discovery = "partial";
            continue;
          }
          if (!isGpuSection(section.name)) continue;
          if (
            !section.file_backed ||
            (directory.format === "elf" && (section.flags & 0x800n) !== 0n)
          ) {
            this.discovery = "partial";
            this.limitations.push(
              `GPU section ${section.index} has no supported uncompressed file range`,
            );
            continue;
          }
          const offset = gpuCount(section.offset),
            size = gpuCount(section.size);
          const borrowed: GpuSource = {
            size,
            containerId: source.containerId,
            read: (start, length) => source.read(offset + start, length),
          };
          // A section is a coordinate space with an explicit physical backing, including PE short names.
          const row = gpuObject(source, {
            offset: offset,
            size: size,
            kind: section.name.includes("hip") ? "hip-bundle" : "cuda-fatbin",
            metadata: {
              offset: section.header_offset,
              size:
                directory.format === "elf"
                  ? directory.bits === 64
                    ? 64
                    : 40
                  : 40,
            },
          });
          row.location.section_index = section.index;
          row.location.section_name = section.name;
          this.#add(row);
          const sectionSource: GpuSource = { ...borrowed, containerId: row.id };
          try {
            await this.inspect(sectionSource, depth + 1);
            row.inspection = "complete";
          } catch (cause) {
            this.#retainFailure(row, cause);
            this.discovery = "partial";
          }
        }
      }
      return directory.format;
    }
    // PTX is positively identified by its directives, never by extension or arbitrary embedded text.
    const probe = await source.read(0, Math.min(source.size, 64 * 1024));
    const target = readPtxTarget(probe.toString("utf8"));
    if (target !== null) {
      const row = gpuObject(source, {
        offset: 0,
        size: source.size,
        kind: "ptx",
      });
      row.target = target;
      this.#add(row);
      await this.#payload(source, row);
      return "ptx";
    }
    throw new BinaryFormatFailure(
      "unsupported",
      "No supported CUDA/HIP container or PTX declarations were identified",
    );
  }
  #retainFailure(row: GpuObject, cause: unknown): void {
    if (
      !(cause instanceof BinaryFormatFailure) &&
      !(cause instanceof BinaryResourceFailure)
    )
      throw cause;
    row.inspection =
      cause instanceof BinaryResourceFailure
        ? "capacity"
        : cause.kind === "unsupported"
          ? "unsupported"
          : "malformed";
    row.limitations.push(cause.message);
  }
  async #payload(source: GpuSource, row: GpuObject): Promise<void> {
    if (this.input.analysis !== "kernels") return;
    if (!selected(row, this.input)) {
      row.inspection = "not-selected";
      return;
    }
    if (row.compression === "unsupported") {
      row.inspection = "unsupported";
      return;
    }
    try {
      if (
        row.kind === "ptx" &&
        (row.decoded_size === null || row.decoded_size > GPU_REPORT_BYTES)
      )
        throw new BinaryResourceFailure(
          "structure",
          GPU_REPORT_BYTES,
          "PTX text exceeds the 16 MiB retained report budget",
        );
      let bytes = await source.read(row.location.offset, row.location.size);
      if (
        row.compression === "zstd" ||
        row.compression === "zlib" ||
        row.compression === "lz4"
      ) {
        if (row.decoded_size === null)
          rejectGpu("Compressed GPU object has no decoded length");
        try {
          this.#budgetDecoded(row.decoded_size!);
          bytes = await decodeGpuPayload(
            bytes,
            row.compression,
            row.decoded_size!,
            this.signal,
          );
        } catch (cause) {
          if (cause instanceof BinaryResourceFailure) throw cause;
          assertArtifactReadActive(this.signal);
          throw new BinaryFormatFailure(
            "malformed",
            "GPU payload decompression failed",
          );
        }
      } else this.#budgetDecoded(bytes.length);
      row.sha256 = gpuSha256(bytes);
      row.decoded_size = bytes.length;
      const decoded = new GpuBufferSource(bytes, row.id);
      if (row.kind === "ptx") {
        // NVIDIA appends NUL alignment padding to an uncompressed PTX entry.
        let length = bytes.length;
        if (row.raw_flags !== null)
          while (length > 0 && bytes[length - 1] === 0) length--;
        const content = bytes.subarray(0, length);
        let text: string;
        try {
          text = new TextDecoder("utf-8", {
            fatal: true,
            ignoreBOM: true,
          }).decode(content);
        } catch {
          return rejectGpu("PTX is not valid UTF-8 text");
        }
        const parsed = readPtxKernels(text, row.id);
        const artifact = {
          id: `${row.id}/ptx`,
          object_id: row.id,
          sha256: gpuSha256(content),
          media_type: "text/x-ptx" as const,
          encoding: "utf-8" as const,
          byte_length: content.length,
          text,
          location: gpuLocation(decoded, 0, content.length),
        };
        this.#retain(artifact);
        this.ptx_artifacts.push(artifact);
        for (const kernel of parsed) {
          this.#retain(kernel);
          this.kernels.push(kernel);
        }
      } else if (row.kind === "cubin" || row.kind === "amdgpu") {
        const parsed = await readGpuElfKernels(decoded, row.kind, (kernel) => {
          this.#retain(kernel);
          this.kernels.push(kernel);
        });
        if (row.target === null) row.target = parsed.target;
        row.limitations.push(...parsed.limitations);
      } else {
        row.inspection = "unsupported";
        row.limitations.push(
          "Payload kind is outside kernel metadata coverage",
        );
        return;
      }
      row.inspection = "complete";
    } catch (cause) {
      this.#retainFailure(row, cause);
    }
  }
  async #cuda(source: GpuSource): Promise<void> {
    for await (const row of readCudaFatbinary(source, this.signal)) {
      this.#add(row);
      await this.#payload(source, row);
    }
  }
  async #tail(
    source: GpuSource,
    offset: number,
    kind: "cuda-fatbin" | "hip-bundle",
    depth: number,
  ): Promise<void> {
    const size = source.size - offset;
    let nonzero = false;
    for (
      let position = offset;
      position < source.size && !nonzero;
      position += 65536
    )
      nonzero = (
        await source.read(position, Math.min(65536, source.size - position))
      ).some((byte) => byte !== 0);
    if (!nonzero) return;
    const row = gpuObject(source, { offset: offset, size: size, kind: kind });
    this.#add(row);
    const tail: GpuSource = {
      size,
      containerId: row.id,
      read: (start, length) => source.read(offset + start, length),
    };
    try {
      await this.inspect(tail, depth + 1);
      row.inspection = "complete";
    } catch (cause) {
      this.#retainFailure(row, cause);
      this.discovery = "partial";
    }
  }

  async #hip(source: GpuSource, depth: number): Promise<void> {
    const header = await source.read(0, 32);
    const count = gpuCount(header.readBigUInt64LE(24));
    if (count > 65536)
      throw new BinaryResourceFailure(
        "structure",
        65536,
        "HIP bundle entry count exceeds the inventory budget",
      );
    let position = 32;
    const rows: GpuObject[] = [];
    for (let index = 0; index < count; index++) {
      if (index % 256 === 0) {
        await scheduler.yield();
        assertArtifactReadActive(this.signal);
      }
      const h = await source.read(position, 24);
      const offset = gpuCount(h.readBigUInt64LE(0)),
        size = gpuCount(h.readBigUInt64LE(8)),
        nameSize = gpuCount(h.readBigUInt64LE(16));
      if (nameSize > 65536)
        throw new BinaryResourceFailure(
          "structure",
          65536,
          "HIP bundle ID exceeds the string budget",
        );
      const name = new TextDecoder("utf-8", {
        fatal: true,
        ignoreBOM: true,
      }).decode(await source.read(position + 24, nameSize));
      if (offset > source.size || size > source.size - offset)
        rejectGpu("HIP entry range is outside its bundle");
      const row = gpuObject(source, {
        offset: offset,
        size: size,
        kind: name.startsWith("host-")
          ? "host"
          : /^(?:hip|hipv4)-amdgcn-/u.test(name)
            ? "amdgpu"
            : "unknown",
        metadata: { offset: position, size: 24 + nameSize },
      });
      row.bundle_id = name;
      row.target = /--(gfx[a-zA-Z0-9]+(?::[^\s]+)?)$/u.exec(name)?.[1] ?? null;
      this.#add(row);
      rows.push(row);
      position += 24 + nameSize;
    }
    for (const row of rows) {
      if (row.location.offset < position && row.encoded_size > 0)
        rejectGpu("HIP entry overlaps its bundle directory");
      if (row.kind === "host") {
        row.inspection = "complete";
        continue;
      }
      await this.#payload(source, row);
    }
    void depth;
  }
  async #compressedHip(source: GpuSource, depth: number): Promise<void> {
    const h = await source.read(0, Math.min(32, source.size));
    if (h.length < 20) rejectGpu("CCOB header is truncated");
    const version = h.readUInt16LE(4),
      method = h.readUInt16LE(6);
    let width: number, total: number, decoded: number, hash: bigint;
    if (version === 1) {
      width = 20;
      total = source.size;
      decoded = h.readUInt32LE(8);
      hash = h.readBigUInt64LE(12);
    } else if (version === 2) {
      width = 24;
      if (h.length < width) rejectGpu("CCOB v2 header is truncated");
      total = h.readUInt32LE(8);
      decoded = h.readUInt32LE(12);
      hash = h.readBigUInt64LE(16);
    } else if (version === 3) {
      width = 32;
      if (h.length < width) rejectGpu("CCOB v3 header is truncated");
      total = gpuCount(h.readBigUInt64LE(8));
      decoded = gpuCount(h.readBigUInt64LE(16));
      hash = h.readBigUInt64LE(24);
    } else
      throw new BinaryFormatFailure(
        "unsupported",
        `Unsupported CCOB version ${version}`,
      );
    if (total < width || total > source.size)
      rejectGpu("CCOB declared extent is invalid");
    const row = gpuObject(source, {
      offset: width,
      size: total - width,
      kind: "hip-bundle",
      metadata: { offset: 0, size: width },
    });
    row.compression =
      method === 0 ? "zlib" : method === 1 ? "zstd" : "unsupported";
    row.compression_id = `ccob:${version}:${method}`;
    row.decoded_size = decoded;
    this.#add(row);
    if (row.compression === "unsupported") {
      row.inspection = "unsupported";
      row.limitations.push("Unsupported CCOB compression method");
      this.discovery = "partial";
      return;
    }
    try {
      this.#budgetDecoded(decoded);
      const data = await decodeGpuPayload(
        await source.read(width, total - width),
        row.compression,
        decoded,
        this.signal,
      );
      if (createHash("md5").update(data).digest().readBigUInt64LE(0) !== hash)
        rejectGpu("CCOB truncated MD5 does not match decoded bytes");
      row.sha256 = gpuSha256(data);
      await this.inspect(new GpuBufferSource(data, row.id), depth + 1);
      row.inspection = "complete";
    } catch (cause) {
      this.#retainFailure(row, cause);
      this.discovery = "partial";
    }
    if (total < source.size)
      await this.#tail(source, total, "hip-bundle", depth);
  }
}
const isGpuSection = (name: string): boolean =>
  [
    ".nv_fatbin",
    ".nv_fatb",
    ".hip_fatbin",
    ".hip_fat",
    ".llvm.offloading",
  ].includes(name);
