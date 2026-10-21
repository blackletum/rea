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
name/register/shared-memory tuples. Final MCP workflows rechecked the Linux
sm_100 subset's 248 complete kernel facts against those prior oracle facts.
The final public CLI analyzed the entire Windows DLL, exported complete
Evidence, and reproduced all previously verified facts exactly.

Windows validation ran on Linux: LLVM independently identified the PE section
coordinates, and cuobjdump consumed its extracted fatbin. cuobjdump did not
recognize the DLL directly. This does not establish Windows runtime behavior,
GPU execution, host launch correlation or legacy AMDGPU YAML support.

## Equivalent completed MCP workflows

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

- The final focused run passed 265 tests in 19 files: GPU domain/codec/parser
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
