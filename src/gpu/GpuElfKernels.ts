import { readCudaInfo, readCudaParameters } from "./CudaInfo.js";
import type { GpuKernel, GpuLocation } from "../domain/gpu/gpuCodeObjects.js";
import {
  BinaryReader,
  BinaryFormatFailure,
} from "../native/metadata/BinaryContainer.js";
import {
  readElfSectionDirectory,
  type NativeSection,
  type NativeSectionDirectory,
} from "../native/metadata/NativeSectionDirectory.js";
import { readElfString, readElfSymbol } from "../native/metadata/ElfSymbol.js";
import {
  gpuCount,
  readGpuNativeDirectory,
  rejectGpu,
  type GpuSource,
} from "./GpuSource.js";
import {
  amdCount,
  amdRecord,
  amdString,
  readAmdMetadata,
} from "./AmdGpuMetadata.js";
const sectionBytes = (
  source: GpuSource,
  section: NativeSection,
): Promise<Buffer> => {
  if (!section.file_backed || (section.flags & 0x800n) !== 0n)
    rejectGpu("GPU metadata section has no supported uncompressed file range");
  return source.read(gpuCount(section.offset), gpuCount(section.size));
};
const site = (
  source: GpuSource,
  section: NativeSection,
  offset = 0,
  size = gpuCount(section.size),
): GpuLocation => ({
  container_id: source.containerId,
  offset: gpuCount(section.offset) + offset,
  size,
  section_index: section.index,
  section_name: section.name,
});
const blankKernel = (
  source: GpuSource,
  id: string,
  name: string,
  location: GpuLocation,
): GpuKernel => ({
  id: `${source.containerId!}/${id}`,
  object_id: source.containerId!,
  representation: "cuda-elf-metadata",
  name,
  symbol: name,
  parameter_count: null,
  parameter_bytes: null,
  parameters: [],
  registers: {
    physical: null,
    sgpr: null,
    vgpr: null,
    agpr: null,
    virtual: [],
  },
  shared_bytes: null,
  private_bytes: null,
  location,
  sources: { parameters: null, registers: null, shared: null, private: null },
  limitations: [],
});
export const readGpuElfKernels = async (
  source: GpuSource,
  kind: "cubin" | "amdgpu",
  onKernel: (kernel: GpuKernel) => void,
): Promise<{ target: string | null; limitations: string[] }> => {
  const directory = await readGpuNativeDirectory(
    source,
    readElfSectionDirectory(source.size),
  );
  if (directory.machine !== (kind === "cubin" ? 190 : 224))
    rejectGpu("GPU entry kind disagrees with ELF machine");
  if (kind === "cubin") {
    await cudaKernels(source, directory, onKernel);
    return {
      target: `sm_${directory.flags & 0xff}`,
      limitations: directory.sections.some((s) => s.name === null)
        ? [
            "Some ELF section names are unknown; named metadata coverage is partial",
          ]
        : [],
    };
  }
  return amdKernels(source, directory, onKernel);
};
const indexCudaMetadata = async (
  source: GpuSource,
  directory: NativeSectionDirectory,
) => {
  const infosByText = new Map<number, NativeSection[]>(),
    sharedByText = new Map<number, NativeSection[]>();
  const registerFacts = new Map<
    string,
    { value: number; location: GpuLocation }[]
  >();
  for (const row of directory.sections) {
    const map =
      row.type === 8 && row.name?.startsWith(".nv.shared.") === true
        ? sharedByText
        : row.type === 0x70000000 && row.name?.startsWith(".nv.info.") === true
          ? infosByText
          : undefined;
    if (map !== undefined) {
      const values = map.get(row.info) ?? [];
      values.push(row);
      map.set(row.info, values);
    }
    if (row.type === 0x70000000 && row.name === ".nv.info") {
      for (const record of readCudaInfo(
        await sectionBytes(source, row),
        directory.little,
      )) {
        if (record.format !== 4 || record.attribute !== 0x2f) continue;
        if (record.data.length !== 8)
          rejectGpu("CUDA register count record has an unsupported extent");
        const r = new BinaryReader(record.data, directory.little),
          key = `${row.link}:${r.u32(0)}`;
        const values = registerFacts.get(key) ?? [];
        values.push({
          value: r.u32(4),
          location: site(source, row, record.offset, record.length),
        });
        registerFacts.set(key, values);
      }
    }
  }
  const allNamesKnown = directory.sections.every((row) => row.name !== null);
  return {
    source,
    directory,
    infosByText,
    sharedByText,
    registerFacts,
    allNamesKnown,
  };
};
type CudaMetadataIndex = Awaited<ReturnType<typeof indexCudaMetadata>>;

const attachCudaResources = (
  kernel: GpuKernel,
  text: NativeSection,
  index: CudaMetadataIndex,
  symbolKey: string,
): void => {
  const { source, directory } = index;
  const registerRows = index.registerFacts.get(symbolKey) ?? [];
  const registerValues = new Set(registerRows.map((row) => row.value));
  const sectionRegisters = text.info >>> 24;
  kernel.registers.physical =
    registerValues.size === 1
      ? [...registerValues][0]!
      : registerValues.size === 0 && sectionRegisters !== 0
        ? sectionRegisters
        : null;
  if (registerValues.size === 1)
    kernel.sources.registers = registerRows[0]!.location;
  else if (sectionRegisters !== 0)
    kernel.sources.registers = {
      container_id: source.containerId,
      offset: text.header_offset,
      size: directory.bits === 64 ? 64 : 40,
      section_index: text.index,
      section_name: text.name,
    };
  if (
    registerValues.size > 1 ||
    (kernel.registers.physical !== null &&
      sectionRegisters !== 0 &&
      kernel.registers.physical !== sectionRegisters)
  ) {
    kernel.registers.physical = null;
    kernel.limitations.push(
      "Conflicting CUDA register declarations; physical count remains unknown",
    );
  }
  const shared = index.sharedByText.get(text.index) ?? [];
  kernel.shared_bytes =
    shared.length === 1
      ? gpuCount(shared[0]!.size)
      : shared.length === 0 && index.allNamesKnown
        ? 0
        : null;
  if (shared.length === 1)
    kernel.sources.shared = {
      container_id: source.containerId,
      offset: shared[0]!.header_offset,
      size: directory.bits === 64 ? 64 : 40,
      section_index: shared[0]!.index,
      section_name: shared[0]!.name,
    };
};

const cudaKernels = async (
  source: GpuSource,
  directory: NativeSectionDirectory,
  onKernel: (kernel: GpuKernel) => void,
): Promise<void> => {
  const index = await indexCudaMetadata(source, directory);
  for (const section of directory.sections) {
    if (section.type !== 2 && section.type !== 11) continue;
    const width = gpuCount(section.entry_size),
      minimum = directory.bits === 64 ? 24 : 16;
    if (width < minimum || section.size % BigInt(width) !== 0n)
      rejectGpu("CUBIN symbol table width is invalid");
    const namesSection = directory.sections[section.link];
    if (namesSection === undefined || namesSection.type !== 3)
      rejectGpu("CUBIN symbol string table is invalid");
    const names = await sectionBytes(source, namesSection!);
    const bytes = await sectionBytes(source, section);
    for (let offset = 0; offset < bytes.length; offset += width) {
      const symbol = readElfSymbol(
        bytes.subarray(offset, offset + width),
        directory.bits,
        directory.little,
      );
      // STO_CUDA_ENTRY distinguishes launchable kernels from ordinary device functions.
      if (symbol.type !== 2 || (symbol.other & 0x10) === 0) continue;
      const text = directory.sections[symbol.section_index];
      if (text === undefined || text.type !== 1 || (text.flags & 4n) === 0n)
        rejectGpu("CUBIN kernel symbol has no executable section");
      const kernel = blankKernel(
        source,
        `sym:${section.index}:${offset / width}`,
        readElfString(names, symbol.name_offset),
        site(source, section, offset, width),
      );
      attachCudaResources(
        kernel,
        text!,
        index,
        `${section.index}:${offset / width}`,
      );
      const infos = index.infosByText.get(text!.index) ?? [];
      if (infos.length === 1) {
        const decoded = readCudaParameters(
          await sectionBytes(source, infos[0]!),
          directory.little,
        );
        kernel.sources.parameters = site(source, infos[0]!);
        kernel.parameter_bytes = decoded.parameter_bytes;
        kernel.parameter_count = decoded.parameter_count;
        kernel.parameters = decoded.parameters;
        kernel.limitations.push(...decoded.limitations);
      } else
        kernel.limitations.push(
          "Exactly one matching CUDA nv.info parameter section was not available",
        );
      kernel.limitations.push(
        "CUDA metadata fields are declarations from ELF symbols, section headers and nv.info; SASS and runtime behavior were not inspected",
      );
      onKernel(kernel);
    }
  }
};
const amdKernels = async (
  source: GpuSource,
  directory: NativeSectionDirectory,
  onKernel: (kernel: GpuKernel) => void,
): Promise<{ target: string | null; limitations: string[] }> => {
  const targets = new Set<string>(),
    limitations: string[] = [];
  let foundMetadata = false;
  for (const section of directory.sections) {
    if (section.type !== 7) continue;
    const bytes = await sectionBytes(source, section),
      r = new BinaryReader(bytes, directory.little);
    for (let p = 0; p < bytes.length;) {
      if (bytes.length - p < 12) rejectGpu("Truncated ELF note header");
      const namesize = r.u32(p),
        size = r.u32(p + 4),
        type = r.u32(p + 8);
      const nameOffset = p + 12,
        desc = nameOffset + Math.ceil(namesize / 4) * 4,
        end = desc + Math.ceil(size / 4) * 4;
      r.range(nameOffset, namesize, "ELF note vendor");
      r.range(desc, size, "ELF note descriptor");
      if (end > bytes.length)
        rejectGpu("ELF note alignment extends outside its section");
      const vendor = bytes
        .subarray(nameOffset, nameOffset + namesize)
        .toString("latin1")
        .replace(/\0$/u, "");
      if (vendor === "AMDGPU" && type === 32) {
        foundMetadata = true;
        const metadata = amdRecord(
          readAmdMetadata(bytes.subarray(desc, desc + size)),
        );
        if (metadata === null || !Array.isArray(metadata["amdhsa.kernels"]))
          rejectGpu("AMDGPU metadata has no valid kernel array");
        const target = amdString(metadata!["amdhsa.target"]);
        if (target !== null) targets.add(target);
        const rows = metadata!["amdhsa.kernels"] as unknown[];
        for (let index = 0; index < rows.length; index++) {
          const value = amdRecord(rows[index]);
          const name = value === null ? null : amdString(value[".name"]);
          if (value === null || name === null)
            rejectGpu("AMDGPU kernel metadata has no name");
          const kernel = blankKernel(
            source,
            `note:${section.index}:${p}:${index}`,
            name!,
            site(source, section, desc, size),
          );
          kernel.representation = "amdgpu-metadata";
          kernel.sources = {
            parameters: kernel.location,
            registers: kernel.location,
            shared: kernel.location,
            private: kernel.location,
          };
          kernel.symbol = amdString(value![".symbol"]);
          kernel.parameter_bytes = amdCount(value![".kernarg_segment_size"]);
          kernel.shared_bytes = amdCount(value![".group_segment_fixed_size"]);
          kernel.private_bytes = amdCount(
            value![".private_segment_fixed_size"],
          );
          kernel.registers.sgpr = amdCount(value![".sgpr_count"]);
          kernel.registers.vgpr = amdCount(value![".vgpr_count"]);
          kernel.registers.agpr = amdCount(value![".agpr_count"]);
          const args = value![".args"];
          if (Array.isArray(args)) {
            kernel.parameter_count = args.length;
            kernel.parameters = args.map((argument, ordinal) => {
              const row = amdRecord(argument);
              if (row === null) rejectGpu("Malformed AMDGPU argument metadata");
              const kind = amdString(row![".value_kind"]);
              return {
                ordinal,
                name: amdString(row![".name"]),
                offset: amdCount(row![".offset"]),
                size: amdCount(row![".size"]),
                type: amdString(row![".type_name"]),
                hidden: kind === null ? null : kind.startsWith("hidden_"),
              };
            });
          }
          kernel.limitations.push(
            "AMDGPU parameter count includes hidden ABI arguments when present; missing fields remain unknown. Kernel metadata is a declaration, not runtime verification.",
          );
          onKernel(kernel);
        }
      } else if (vendor === "AMD" && (type === 10 || type === 11))
        limitations.push(
          "Legacy AMDGPU YAML metadata is outside MessagePack coverage",
        );
      p = end;
    }
  }
  if (!foundMetadata)
    throw new BinaryFormatFailure(
      "unsupported",
      "No AMDGPU v3+ MessagePack kernel metadata was available",
    );
  return { target: targets.size === 1 ? [...targets][0]! : null, limitations };
};
