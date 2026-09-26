# PCS Auto-Annotation v1 — Multimodal Baseline

## Goal

Prove one trustworthy end-to-end annotation loop on a real DAT trace:

```
DAT
 ├─ LiDAR point cloud
 ├─ camera frames
 └─ ADMA
      ↓
synchronized sample
      ↓
camera proposal + Innov3 DSVT proposal
      ↓
camera↔LiDAR association
      ↓
3D annotation candidate + provenance
      ↓
PCS inspection / correction
      ↓
accepted / modified / rejected annotation
```

The v1 success criterion is not maximum model accuracy. It is whether the pipeline materially reduces the human effort required to create trustworthy 3D annotations while preserving the original machine evidence.

## Architectural rules

1. Keep this repository independent from Point Cloud Studio and lidar-perception-lab.
2. Reuse the existing VLM-AutoYOLO application shell: React UI, FastAPI backend, persistence, model lifecycle, batch/job infrastructure, and existing camera AI.
3. DAT is a first-class source. Do not require manual conversion before ingestion.
4. `point_cloud_studio_native` is the sole DAT decoding dependency. This repository must not copy PCS parsers, import PCS GUI/application modules, or reimplement IFSCAN/camera/ADMA binary semantics.
5. Consume the versioned **headless PCS native wheel** and verify its `pcs-native-manifest.json` (PCS commit, module version, Python ABI/platform, enabled features, wheel filename and SHA-256), following the same external-consumer model used by LiDAR Lab.
6. LiDAR DAT access uses the native `transport.NativeDatReader` plus native codec surfaces; camera DAT access uses `transport.NativeDatImageStreamSource` / the equivalent active-reader image methods.
7. ADMA must also come through a supported native-module API. If the current external wheel does not expose typed ADMA samples, extend `point_cloud_studio_native` in PCS first; do **not** add an ADMA parser here.
8. Introduce a generic synchronized sample contract so future PCAP/MCAP/OpenX sources can be added without changing downstream model code.
9. Innov3 DSVT is the frozen v1 LiDAR baseline. No fine-tuning in this PR series.
10. Keep camera and LiDAR model outputs as independent evidence. Fusion must preserve per-provider confidence and provenance.
11. PCS remains the authoritative LiDAR inspection/correction environment; this tool orchestrates annotation and review but does not duplicate PCS visualization or decoding.
12. Human decisions never overwrite the original machine proposal.
13. Avoid introducing additional VLMs or learned fusion until the baseline is measured.

## Core contracts

### Synchronized sample

```text
Sample
├─ sample_id
├─ timestamp
├─ camera
├─ point_cloud
├─ adma_pose
├─ calibration
└─ source_metadata
```

### Annotation proposal

```text
ObjectProposal
├─ proposal_id
├─ sample_id
├─ class
├─ bbox_3d
├─ bbox_2d            optional
├─ mask               optional
├─ provider_evidence[]
├─ association_score  optional
├─ fused_confidence   optional
└─ provenance
```

### Human disposition

```text
proposed → accepted | modified | rejected
```

The original proposal and provider evidence remain immutable after human review.

## v1 work packages

### WP1 — PCS native dependency, source and sample foundation
- Add the versioned `point_cloud_studio_native` headless wheel as the DAT runtime dependency and verify its manifest/checksum.
- Add a thin DAT dataset adapter over native APIs; no copied PCS decoding code.
- Use native LiDAR and camera stream APIs to expose synchronized point cloud and camera samples.
- Verify whether the native wheel already exposes the required ADMA stream contract. If not, open/implement the missing API in PCS native first and consume it here afterward.
- Expose synchronized camera, point cloud, ADMA and metadata as `Sample` only after all three are backed by native authority.
- Define calibration attachment and timestamp semantics.
- Add deterministic sample identifiers.
- Add contract tests plus real-DAT qualification.

### WP2 — Provider abstraction
- Generalize the current detection strategy into provider-oriented annotation evidence.
- Preserve existing LocateAnything/SAM behavior.
- Add a provider interface capable of 2D and 3D outputs.
- Add model/provenance metadata to persisted proposals.

### WP3 — Innov3 DSVT provider
- Integrate the existing Innov3 DSVT checkpoint as a frozen LiDAR provider.
- Record exact model/checkpoint identity, preprocessing contract, class mapping and checksum.
- No retraining or model conversion unless required for inference compatibility.

### WP4 — Camera↔LiDAR association
- Use camera intrinsics, distortion/rectification state and LiDAR→camera extrinsics.
- Project 3D candidates to camera space.
- Associate DSVT proposals with camera detections using deterministic geometric and semantic rules.
- Keep all source scores visible.

### WP5 — PCS review loop
- Export/open a sample plus candidate 3D annotations in PCS.
- Reuse PCS manual annotation/inspection capabilities.
- Persist accept / modify / reject decisions back into the annotation dataset.
- Preserve original proposals and correction deltas.

### WP6 — Baseline qualification
Measure on representative DAT data:
- DSVT-only detections
- camera-only detections
- fused candidates
- false positives / misses
- class corrections
- 3D box corrections
- percentage accepted unchanged
- annotation time per frame

## Explicit non-goals for v1

- No Qwen/SmolVLM model comparison.
- No learned fusion network.
- No DSVT fine-tuning.
- No automatic training loop.
- No full temporal tracker.
- No production packaging.
- No replacement of PCS.
- No migration of lidar-perception-lab experiments into this repository.

## PR1 scope

PR1 establishes only the architectural foundation:

- project v1 contract and documentation;
- generic `Sample` and annotation/provenance domain contracts;
- provider interface compatible with existing camera detection paths;
- PCS-native dependency/manifest contract and a thin DAT adapter over `point_cloud_studio_native`;
- native capability check for LiDAR, camera and ADMA, with ADMA explicitly blocked rather than locally reimplemented if the wheel lacks the API;
- test scaffolding for contracts and source/provider boundaries;
- no DSVT inference yet;
- no fusion yet;
- no PCS bridge yet.

## PR1 acceptance criteria

- Existing VLM-AutoYOLO behavior remains functional.
- Existing camera detection strategies can be adapted behind the provider abstraction without changing observable behavior.
- Domain contracts support both 2D and 3D evidence.
- DAT source boundary is explicit and testable.
- The auto-annotation repository contains no DAT/IFSCAN/camera/ADMA binary parser.
- Native wheel identity/checksum is recorded and validated.
- Missing ADMA native support fails explicitly and points to a PCS-native capability gap; there is no Python fallback parser.
- No hard-coded echo count, sensor topology, camera resolution or model-specific fields in generic contracts.
- Provenance is mandatory for machine-generated proposals.
- Architecture leaves PCS integration and Innov3 DSVT as additive providers/consumers rather than special cases.
- Tests and linting pass.
