# Static GPU code-object verification

The [receipts](gpu-code-objects-2026-10-21.json) pin the measured runtime tree,
publisher URLs/digests, actual commands, completed answers and resource costs.
Downloaded binaries, toolchains and complete Evidence bundles remain outside Git.

## Correctness

NVIDIA CUDA 12.9.1 redistribution archives were verified against their publisher
manifest. nvcc 12.9.86 compiled a small owned kernel for sm_80 and sm_90a with
uncompressed, size-compressed and speed-compressed fatbins. cuobjdump 12.9.82
independently confirmed two parameters at offsets 0/8 with sizes 8/4, 12 parameter
bytes, 10 physical registers and architecture-dependent shared memory. Separate
PTX/CUBIN and a host ELF object exercised the same parser boundaries.

Clang 14 produced a gfx900 HIP bundle and AMDGPU ELF object; llvm-readelf supplied
the MessagePack metadata oracle. An LLVM 19 offload bundler produced a real CCOB
v2 Zstd bundle. All nine producer artifacts were reread on the final runtime
source with complete payload coverage. CCOB v1/v3 and zlib have portable boundary
fixtures, rather than real producer verification.

Released Linux and Windows nvJPEG 12.4.0.76 artifacts each contain 165 CUBIN and
10 PTX payloads, with 3,968 kernel declarations across 15 target declarations.
Independent extraction matched all 165 decoded CUBIN hashes and all 10 PTX byte
sequences; cuobjdump resource output matched all 3,720 native kernel
name/register/shared-memory tuples. Pre-review MCP workflows rechecked the Linux
sm_100 subset's 248 complete kernel facts against those prior oracle facts.
The pre-review public CLI analyzed the entire Windows DLL, exported complete
Evidence, and reproduced all previously verified facts exactly.

Windows validation ran on Linux: LLVM independently identified the PE section
coordinates, and cuobjdump consumed its extracted fatbin. cuobjdump did not
recognize the DLL directly. This does not establish Windows runtime behavior,
GPU execution, host launch correlation or legacy AMDGPU YAML support.

## Initial completed MCP workflows

Both fresh stdio MCP sessions analyzed only `sm_100` in the same Linux library,
retained 176 inventory objects and 248 kernel facts, queried the same named
kernel, exported Evidence, and repeated the historical query three times.
The complete variant returned the selected-target report; the selected variant
returned a same-call kernel page. Both preserved the exact answer: one parameter
of 168 bytes, 31 physical registers and 9,216 shared bytes.

| Cost                                                            |   Complete report |      Selected page |
| --------------------------------------------------------------- | ----------------: | -----------------: |
| Used input declarations + three primary calls, estimated tokens |           406,862 |             26,106 |
| All input declarations + primary calls, estimated tokens        |           635,381 |            254,625 |
| Full advertised catalog + all six calls, estimated tokens       |         1,475,258 |          1,094,502 |
| Process-family elapsed seconds                                  |            15.492 |             15.443 |
| Process-family CPU seconds                                      |            18.220 |             18.074 |
| Sampled peak family RSS bytes                                   |     1,139,601,408 |      1,321,566,208 |
| First inspection milliseconds                                   |             8,572 |              8,810 |
| Three warm retained-page calls, milliseconds                    | 10.7 / 10.6 / 9.8 | 12.0 / 11.4 / 10.6 |

Tokenx estimates include advertised schemas, arguments and full SDK replies;
they are not actual model usage. Text and structured reply representations are
both counted. Initialization and JSON-RPC envelopes are unmeasured. Complete
catalog discovery substantially reduces the overall percentage saving.

These are workflow variants on the same implementation, not a before/after
parser speed benchmark. Node compile caching was disabled, but OS caches and
unrelated host load were uncontrolled. No general latency, CPU or memory
improvement is established. The selected workflow used more peak memory in this
pair. An earlier all-target complete reply exceeded the existing transport guard;
that incomplete attempt is excluded from the completed comparison.

The full Windows DLL CLI task took 4.064 seconds / 4.909 CPU seconds /
503,459,840 peak family RSS bytes. All profiled completed tasks stayed within
the 8 GiB and 420 second guards, using two CPU affinities. Peak RSS sums payload
descendants every 20 ms and is approximate; the observer is excluded.

## Regression checks

- The initial focused run passed 265 tests in 19 files: GPU domain/codec/parser
  cases, public CLI/MCP, retained views, output profiles, cancellation and cleanup,
  plus shared ELF/PE Go, Rust and IL2CPP consumers.
- Full static checks passed before the catalog/docs-only follow-up; its receipt
  records 6,866,296,832 peak family RSS bytes. The final CI run qualifies the PR.
- Documentation/catalog/license checks passed after fixing missing catalog
  ownership, translated README links and the new MessagePack dependency notices.
- An earlier combined schema run hit two 30 second timeouts on two CPUs. The
  isolated canonical schema suite passed all 15 cases without increasing limits.
  This is recorded as a successful isolation, not a proven baseline failure.

Retention caches are scoped to immutable Evidence through a WeakMap; historical
pages do not reopen files. Live inspections always stream a fresh full-file hash
and revalidate the descriptor/path. No live-state freshness cache is claimed.

## Review follow-up

At `36b87ed`, the public CLI reproduced both review findings: selecting
`amdgcn-amd-amdhsa--gfx900` in the real Clang-produced standalone object returned
zero kernels with an unknown target, and malformed `--targets` JSON was reported
at the `view` issue path.

At `09c6062`, AMD ELF note parsing owns target discovery and prepared kernel
declarations. Inventory reads those bounded notes before selection; selected
analysis reuses them without decoding MessagePack twice. Aggregate encoded note
bytes are limited to 8 MiB before reads, with existing per-value/depth limits;
the parsed JavaScript representation expands beyond that source-byte count.
The prepared declarations live only for the current payload inspection. Named
CLI option parsing also supplies the actual parsed input to schema diagnostics.

The real public CLI now inventories the gfx900 target, returns the producer's
one kernel/two parameters/12-byte parameter area/6 SGPR/2 VGPR when selected,
and preserves an unselected object without kernel facts for a different target.
Those three complete CLI tasks together took 5.974 seconds / 6.951 CPU seconds /
383,623,168 sampled peak family RSS bytes, including the Python task driver.
All nine producer artifacts were reread with complete payload coverage.

The review regression run passed 35 cases in four files, including public CLI,
MCP, metadata capacity and cleanup. A further 10 product-catalog cases passed
after adding the missing CLI `analysis` option description identified by CI.
The earlier paired nvJPEG measurements remain pinned to their original runtime;
these follow-ups do not establish a new CUDA performance result.

## Bounded metadata and partial-fact review

The latest qualification at `ad1e17a` rebases onto main `5241597`, preserving
both Rust and GPU retained views. All 70 focused cases in nine files passed,
including both public CLI/MCP workflows; the separate SDK-converted canonical
schema suite passed all 15 cases.

CUDA parsing shares an aggregate 262,144 record/symbol visit budget, 128 MiB
section-read budget and 16 MiB serialized-size accounting budget for index facts.
Equal scoped byte ranges and duplicate register values are coalesced before
index growth; conflicting values survive. Parsed parameter metadata is reused.
The public CLI completed an 861,896-byte CUBIN containing 1,024 headers aliasing
65,536 records and retained the correct physical count. The old 67-million-entry
expansion is source-derived; it was not run to exhaustion.

Decoder coverage now reaches object, report and target pages: absent CUBIN
symbol tables and unknown section names remain partial. AMD note parsing uses
a tagged Result that retains earlier parsed notes on later malformed, capacity,
unsupported or interrupted reads; retained kernel facts are emitted before the
original failure is propagated. Standalone PTX recognition continues beyond the
64 KiB fast path up to its actual 16 MiB text budget and reuses complete probe
bytes. Unsupported/overflowing register counts stay null instead of low totals.

The public CLI verified an AMD valid-note/truncated-note input and a PTX file
with a 70 KiB leading comment, in addition to the alias case. Those three tasks
together took 5.132 seconds / 6.222 CPU seconds / 405,835,776 peak family RSS
bytes, including the Python driver. Regression cases additionally cover invalid
MessagePack and unsupported extensions in standalone and bundled AMD objects.

The revised Windows DLL workflow preserved all 165 decoded CUBIN hashes, all
10 PTX byte identities and all 3,720 native kernel facts. Independently counting
register declarations in cuobjdump-extracted PTX verified all 248 PTX kernels;
20 kernel totals now correctly include previously missed declarations. This
is a virtual-declaration oracle, not physical allocation or GPU execution.
The Linux sm_100 workflows still match all 248 independently checked native
kernel facts. All nine real producer artifacts retain complete payload coverage.

The latest complete/selected MCP pair performed the same six calls and answered
the same question as the initial pair. Receipts retain both measurement stages.

| Cost                                                      | Complete report | Selected page |
| --------------------------------------------------------- | --------------: | ------------: |
| Used input declarations + primary calls, estimated tokens |         407,368 |        26,612 |
| All input declarations + primary calls, estimated tokens  |         636,495 |       255,739 |
| Full catalog + all six calls, estimated tokens            |       1,480,698 |     1,099,942 |
| Process-family elapsed seconds                            |          15.487 |        15.923 |
| Process-family CPU seconds                                |          18.404 |        18.760 |
| Peak family RSS bytes                                     |   1,251,254,272 | 1,201,774,592 |

These remain tokenizer estimates and same-code workflow variants. The single
latest pair establishes no general latency/CPU/memory improvement; OS caches
and host contention remain uncontrolled. The Windows CLI rerun completed in
4.078 seconds / 4.880 CPU seconds / 498,380,800 peak family RSS bytes.

Main CI independently reproduced the browser termination and WASM optional-input
assertion failures without GPU changes. Cancelled fail-fast shards are not
counted as passing. Those baseline failures are outside this feature's fixes.
