# Static CUDA and HIP code objects

`inspect_gpu_code_objects` answers which GPU targets and kernel declarations a
local artifact carries. It reads ELF/PE sections, standalone CUDA fatbins,
CUBIN/PTX and Clang HIP bundles, including CCOB v1/v2/v3. It neither executes the
artifact nor invokes CUDA/ROCm tools. GPU execution, SASS disassembly and host
launch correlation remain outside coverage.

Start with a target inventory:

```json
{ "path": "/artifacts/library.so", "view": { "collection": "targets" } }
```

Target rows aggregate inventoried payloads and distinguish an unrequested kernel
analysis from a complete or partial one. `objects` pages expose exact object
IDs, compression declarations and encoded/header locations. Fields, exact
filters, batch IDs and stable pagination select the facts delivered to the
caller. IDs are positional within the recorded artifact SHA-256. Selecting `object_ids`
in a new inspection requires `expected_sha256` from that inventory; a changed
file is rejected before parsing selected payloads.

Decode selected targets and return a kernel page in the same call:

```json
{
  "path": "/artifacts/library.so",
  "analysis": "kernels",
  "targets": ["sm_100"],
  "view": {
    "collection": "kernels",
    "names": ["kernel_name"],
    "fields": [
      "name",
      "parameter_count",
      "parameter_bytes",
      "parameters",
      "registers",
      "shared_bytes"
    ]
  }
}
```

Only selected payloads are decoded. Other inventory entries remain present with
`not-selected` states, and overall payload coverage remains partial. An empty
kernel page proves no matching retained fact; it does not establish that an
unexamined payload contains no kernel. Missing fields remain null.

Kernel representations have different meanings:

- PTX reports parameter declarations and virtual registers. Shared bytes cover
  supported static declarations inside that entry; they do not establish module
  shared memory, dynamic allocation or physical device resources.
- CUBIN identifies CUDA entry symbols and reads numeric `nv.info` records and
  native section declarations. Global register declarations also cover compiler
  versions that leave the older text-section register field empty.
- AMDGPU reads v3+ MessagePack notes. Argument counts include hidden ABI arguments
  when present; SGPR/VGPR counts and fixed group/private segment sizes retain
  their declared meanings. Legacy YAML metadata remains unsupported.

Each selected PTX payload becomes a retained UTF-8 text artifact with a SHA-256,
byte length and source coordinates. It is omitted from compact replies. Read it
with `inspect_analysis_view` and the parent Evidence ID returned by the compact
receipt:

```json
{
  "source": { "kind": "retained-evidence", "evidence_id": "ev_..." },
  "view": {
    "kind": "gpu-code-objects",
    "collection": "ptx",
    "object_id": "...",
    "offset": 0,
    "limit": 4096
  }
}
```

Text offsets use UTF-16 code units and must fall on Unicode scalar boundaries.
Complete Evidence retains the selected text; `export_evidence_bundle` provides
its portable source record. Historical pages never reopen the input file.

A location with `container_id: null` uses original file offsets. Other locations
use that inventoried object's decoded byte coordinates. Follow its parent,
encoded location and digest to trace decompressed metadata back to a native
section and the original file. Decoded offsets are never presented as physical
file offsets.

The CLI uses the same service and contracts:

```sh
rea inspect-gpu-code-objects /artifacts/library.so --analysis kernels \
  --targets '["sm_100"]' --detail summary \
  --view '{"collection":"kernels","names":["kernel_name"]}' \
  --evidence_output /analysis/library-gpu.evidence.json --json
```

A full-file SHA-256 always streams every byte. Inventory then reads bounded
native directories and bundle headers; CCOB requires decompression to discover
its inner directory. Kernel analysis reads selected payloads. `bytes_read`
includes hashing and subsequent descriptor reads, so this is not a header-only
hashing workflow.

CUDA raw LZ4 blocks and Zstd, and CCOB zlib/Zstd are supported. Unknown versions
or codecs retain declarations with explicit limitations. Lengths are checked
before decoding, and CCOB's truncated MD5 is verified against decoded bytes.
Resource limits are one active inspection, 128 MiB per decoded object, 512 MiB
of decoded work per inspection, and 16 MiB retained facts including selected PTX.
Native tables and decoded section names each have a 16 MiB structural budget;
AMDGPU notes have bounded depth and container/string allocations. Pages have a
1 MiB response budget. Resource refusals and interrupted observations preserve
known facts and explicit uncertainty.
