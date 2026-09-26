# PCS Auto-Annotation Companion — Architecture and Baseline

## Product north star

PCS Auto-Annotation exists to reduce the manual effort required to create
trustworthy automotive 3D annotations.

It is a **companion to Point Cloud Studio (PCS)**, not a replacement for PCS and
not an independent sensor tool. The production/default service refuses to start
when the PCS native dependency is unavailable. CI may explicitly disable this
startup gate because the private/native artifact is not present there.

PCS remains the central component and owns:

- DAT and sensor-protocol interpretation;
- point-cloud decoding semantics;
- camera and ADMA source semantics;
- calibration authority;
- detailed LiDAR visualization and inspection;
- authoritative human review/correction workflows.

PCS Auto-Annotation owns:

- annotation model orchestration;
- camera and LiDAR proposal generation;
- cross-modal association;
- proposal provenance and confidence;
- annotation review queues and dataset curation.

The current product scope ends at **auto-annotation**. Real2Sim is not part of
the current roadmap. Data contracts should remain reusable enough that validated
annotations could be consumed by Real2Sim later without coupling this tool to a
Real2Sim pipeline.

## Target workflow

```text
DAT
  │
  ▼
point_cloud_studio_native
  ├── LiDAR
  ├── Camera
  └── ADMA
        │
        ▼
synchronized sensor sample
        │
        ├── Innov3 DSVT ───────────────► 3D proposals
        │
        └── LocateAnything + SAM2 ─────► 2D proposals / masks
                                           │
                         camera↔LiDAR association
                                           │
                                           ▼
                              annotation proposal
                               + provenance
                               + confidence
                                           │
                                           ▼
                                          PCS
                                  inspect / correct
                                           │
                                           ▼
                           accepted / modified / rejected
```

## PCS dependency and ownership contract

`pcs-autoannotation` must not contain its own DAT, IFSCAN, camera-stream or
ADMA binary parsers.

The supported external dependency is the versioned PCS headless PyO3 wheel:

```text
point_cloud_studio_native
```

The artifact provenance contract is `pcs-native-manifest.json` with schema
`pcs-native-artifact/v1`. It records the PCS repository/commit, native module
version, Python ABI/platform, enabled features, wheel name and SHA-256.

Install the dependency from a PCS-generated wheel and manifest:

```bash
cd backend
PYTHONPATH=. python scripts/install_pcs_native.py \
  --wheel /path/to/point_cloud_studio_native-*.whl \
  --manifest /path/to/pcs-native-manifest.json
```

The installer verifies the wheel SHA-256 before installation and validates the
installed module against the manifest afterward.

A missing native capability is a PCS-native integration gap. It must not be
worked around by introducing a second protocol implementation in this repository.

## PCS native DAT ingestion

The companion consumes these PCS-owned surfaces:

- `transport.NativeDatReader` for LiDAR DAT access;
- `codec.decode_ifscan_payload` for structured point-cloud decoding;
- `transport.NativeDatImageStreamSource` for timestamped camera access;
- `transport.NativeDatAdmaStreamSource` for structurally classified,
  timestamp-indexed ADMA access.

The companion may adapt native outputs into its own annotation-domain objects,
but must not infer or duplicate sensor binary layouts.

### Current implementation

`backend/app/autoannotation/pcs_native_dat_source.py` provides a thin
`PcsNativeDatSource`.

It can currently:

- inspect PCS-native DAT streams;
- identify LiDAR, image and ADMA streams from PCS structural classification;
- decode a LiDAR frame through PCS native;
- retrieve the camera frame nearest to the LiDAR timestamp;
- retrieve the ADMA sample through `NativeDatAdmaStreamSource`;
- fail closed when ADMA is outside the qualified 20 ms synchronization
  tolerance;
- construct a deterministic sample identifier from the actual selected streams.

The companion does not identify ADMA by names such as `ADMA` or
`ADMA_NET_3330`, and does not decode ADMAnet payloads itself.

The implementation intentionally takes echo count, slots, layers, IFSCAN version
and structure metadata from the native decoded frame. Those values are not
hard-coded in the companion.

## Synchronized sensor sample contract

The annotation core works with a sensor-neutral orchestration contract while
keeping each sensor payload strongly typed:

```text
SynchronizedSample
├── sample_id
├── timestamp_ns
├── LidarFrame
│   ├── points
│   ├── attributes
│   └── PCS-native metadata
├── CameraFrame
│   ├── encoded image bytes
│   ├── width / height / encoding
│   └── synchronization metadata
├── AdmaSample
└── source_metadata
```

ADMA is part of the baseline synchronized sample. The default V1 path requires
LiDAR, camera and ADMA to be supplied by the PCS native wheel.

The PCS native ADMA contract was qualified on the shared `highway-v1` DAT with
6,001 ADMA samples and 235 LiDAR frames. Native nearest-time association produced
median 2.321 ms, p95 4.762 ms, p99 6.916 ms and max 17.938 ms, with zero samples
outside the 20 ms qualification tolerance. The companion uses bounded
`get_sample_at(..., tolerance_ns)` rather than unbounded nearest lookup.

## Annotation provider contract

Model implementations must consume synchronized samples rather than source
formats.

The generic provider boundary is:

```text
AnnotationProvider
  identity
  infer(SynchronizedSample) -> ObjectProposal[]
```

Every machine proposal records provider identity and evidence. Model code must
not know how DAT or IFSCAN is encoded.

## Camera annotation baseline

The first camera baseline remains the functionality inherited from
VLM-AutoYOLO:

- NVIDIA LocateAnything-3B for open-vocabulary visual grounding;
- SAM2 for mask/boundary refinement.

This baseline is experimental. LocateAnything's model license must be reviewed
before any commercial/product deployment.

The existing camera engine is now adapted behind the generic annotation
provider contract without changing the inherited image/video workflow. The
companion adapter accepts PCS-native JPEG/PNG and supported 8-bit raw camera
encodings in memory, runs LocateAnything, optionally refines detections with
SAM2, and emits `ObjectProposal` records. It does not introduce a second camera
reader.

The inherited LocateAnything confidence parser was also corrected so a
`<conf>` token applies to the immediately preceding bounding box, matching the
model output contract.

## Innov3 DSVT LiDAR proposals

Innov3 DSVT is the first LiDAR proposal model because it provides continuity
with the already-used internal model and gives the baseline a known reference.

The provider must consume only the normalized PCS-native point-cloud contract:

```text
PCS native decoded point cloud
        ↓
Innov3 DSVT provider
        ↓
3D boxes + class + confidence + model provenance
```

The provider must not contain DAT/IFSCAN handling.

The exact checkpoint, preprocessing contract, class mapping and checkpoint
SHA-256 must be recorded. No fine-tuning is part of the initial baseline.

The companion now contains:

- `innov3_dsvt.py` — the frozen internal model-input contract;
- `innov3_openpcdet_runtime.py` — lazy OpenPCDet/DSVT execution;
- `innov3_provider.py` — conversion from model output back to PCS annotation
  coordinates;
- `providers/factory.py` — explicit runtime configuration.

The trusted model configuration already exists in the internal reference at
`internal-dsvt-reference/model_data/internal_config.pkl`. The checkpoint remains
an external staged asset and is verified against SHA-256
`bdb7779c879094b1479ed3169025ee8eb1a391ec4c9d3a6c7fd2e06b8eff6c1f`.

Runtime configuration is explicit:

```text
INNOV3_DSVT_ROOT
INNOV3_CONFIG_PATH
INNOV3_CHECKPOINT_PATH
INNOV3_DEVICE
INNOV3_HOUSING_MERGED
```

There are no repository-specific workstation paths in the implementation. The
`/api/v1/autoannotation/dat/innov3-proposals` endpoint runs the current DAT
sample through PCS-native decode and then Innov3. A real workstation execution
is still required before this branch is considered runtime-qualified.

## Camera–LiDAR association

The first association implementation should be deterministic rather than learned.

Inputs:

- DSVT 3D proposals;
- LocateAnything/SAM2 2D proposals and masks;
- camera intrinsics;
- LiDAR-to-camera extrinsics;
- timestamp synchronization metadata.

Outputs preserve the source evidence and add an association score. Individual
provider confidence values must remain visible; a fused score must not erase
their provenance.

## PCS review and correction loop

PCS remains the authoritative environment for detailed LiDAR inspection and 3D
correction.

The companion should hand PCS:

- source DAT/sample identity;
- timestamp/frame identity;
- proposed 3D boxes/classes;
- camera evidence;
- provider provenance.

Human decisions are stored separately as:

```text
proposed → accepted | modified | rejected
```

The original machine proposal remains immutable so model quality and correction
effort can be measured later.

## Auto-annotation baseline qualification

The first qualification dataset must use representative real DAT recordings.

Measure:

- LiDAR-only proposal precision/recall;
- camera-only proposal precision/recall;
- associated proposal precision/recall;
- false positives and missed objects;
- class corrections;
- 3D box corrections;
- percentage accepted without modification;
- annotation time per frame;
- performance by distance and relevant scene conditions.

The main product metric is reduction in human annotation effort without reducing
annotation trustworthiness.

## Explicitly outside the current scope

- Real2Sim pipeline implementation;
- learned sensor-fusion networks;
- Innov3 fine-tuning;
- broad VLM model comparison;
- automatic model-training flywheels;
- full temporal tracking;
- replacement of PCS visualization or decoding;
- migration of LiDAR Perception Lab into this repository.

## Capability status

- **PCS Native DAT Ingestion — implemented and PCS-qualified.** Native runtime
  probing, structural stream inspection, LiDAR decode, nearest camera retrieval,
  native ADMA lookup and native-wheel verification/install support are present.
  PCS PR #147 head `aec3a119e1017be8dcf10924284b5334bd6038dc`
  passed PCS CI and the headless native artifact workflow.
- **Synchronized Sensor Sample — implemented foundation.** The V1 sample now
  contains PCS-native LiDAR, camera and ADMA evidence. ADMA association is
  bounded by the qualified 20 ms tolerance and preserves native decoded fields
  without companion-side pose interpretation.
- **Innov3 DSVT LiDAR Proposals — real Highway execution passed.** The frozen
  preprocessing contract, checkpoint verification, OpenPCDet runtime, provider
  and DAT-to-proposal path are present. Self-hosted run 36252474205 decoded a
  real Highway sample through PCS native and executed Innov3 on the RTX A2000.
  The smoke produced one raw model proposal (label 2, score 0.1568). This proves
  execution only; proposal quality and PCS visual validation remain unqualified.
- **LocateAnything/SAM2 Camera Provider — adapter implemented, runtime pending.**
  The real Highway smoke successfully extracted the selected Genicam2 frame
  through PCS native, but the workstation currently has no LocateAnything/SAM2
  Python runtime or cached LocateAnything-3B model. Camera-model execution is
  therefore still pending.
- **Camera–LiDAR Association — pending.**
- **PCS Review and Correction — pending.**
- **Auto-Annotation Baseline Qualification — pending.**

These explicit capability names are the tracking vocabulary for this repository.
Scheduling changes must update status under the same names rather than inventing
new numbered labels.


## Real Highway smoke evidence

Self-hosted workflow run `36252474205` at companion commit
`d752022db23345edc53654c5a1339d78b18ac04d` passed on `linux-ws`.

Verified on the shared `highway-v1` DAT:

- point-cloud stream: `ScaLa 3-PointCloud`, 235 frames;
- selected camera stream: `Genicam2`, 4096×960, 2,401 frames;
- ADMA stream: 6,001 samples;
- real first/middle/last synchronized sample construction passed;
- ADMA stayed inside the 20 ms synchronization gate;
- native IFSCAN10 decoding used PCS `export` + `math` surfaces;
- Innov3 executed on the real prepared frame and returned one raw proposal;
- evidence artifact: `pcs-autoannotation-highway-smoke-36252474205`.

The smoke also found that the runner's base Python environment does not yet
contain `torch`, `transformers`, `sam2`, or a cached
`nvidia/LocateAnything-3B` model. The camera provider code is present, but its
real model execution remains a separate qualification gate.


## v1 execution boundaries

The v1 path now has explicit runtime boundaries rather than assuming one Python
environment can host every model:

- the FastAPI/camera runtime stays on Python 3.12;
- PCS native is installed as the Qt-free headless wheel and remains the
  authority for DAT/IFSCAN/camera/ADMA interpretation;
- Innov3 may run through `Innov3SubprocessRuntime` in its separately qualified
  Python 3.11 + CUDA/OpenPCDet environment;
- LocateAnything and SAM2 remain camera evidence providers. On constrained GPUs
  the provider can release the VLM before loading SAM2 instead of requiring
  both model allocations to coexist.

Camera/LiDAR association is deterministic and calibration-gated. The companion
accepts PCS calibration-input sidecars or PCS overlay-session calibration,
projects canonical PCS LiDAR boxes with the PCS LiDAR-to-camera transform, and
uses class-compatible 2D IoU for one-to-one association. Missing, unverified,
distorted, or image-size-incompatible calibration fails closed.

The review boundary is the PCS-owned `pcs.scene_objects` schema version 1.
The companion emits immutable 3D hypotheses in the canonical PCS box
convention; PCS owns accept/correct/reject and manual-object editing. Reviewed
documents can be read back by the companion to compute baseline review and
human-effort metrics without changing the original machine proposal.

`backend/scripts/qualify_autoannotation_v1.py` is the common qualification
entry point for workstation CI and later deployment automation. The
self-hosted v1 workflow provisions a persistent camera-model environment,
reuses the exact PCS native contract, invokes Innov3 through its qualified
runtime, discovers a verified real PCS calibration, and writes a proposed PCS
scene-object document plus evidence artifacts.
