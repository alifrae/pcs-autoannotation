// Using auto-imported centralized request helper

export async function detectImage(
  file: File,
  categories: string[],
  useSam2?: boolean,
  sam2ScoreThreshold?: number,
  useSam3?: boolean,
  sam3Text?: string,
  useSam3Seg?: boolean,
  sam3Threshold?: number,
  sam3MaskThreshold?: number,
  signal?: AbortSignal,
): Promise<DetectResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("categories", JSON.stringify(categories));
  if (useSam3) {
    form.append("use_sam3", "true");
    if (sam3Text) form.append("sam3_text", sam3Text);
    if (useSam3Seg === false) form.append("use_sam3_seg", "false");
    if (sam3Threshold != null) form.append("sam3_threshold", String(sam3Threshold));
    if (sam3MaskThreshold != null) form.append("sam3_mask_threshold", String(sam3MaskThreshold));
  } else if (useSam2) {
    form.append("use_sam2", "true");
    if (sam2ScoreThreshold != null) form.append("sam2_score_threshold", String(sam2ScoreThreshold));
  }
  const { data } = await request.post<{ data: DetectResponse }>("/detect", form, {
    signal,
    timeout: DETECT_TIMEOUT,
  });
  return data.data;
}

export async function listDetections(
  page = 1,
  pageSize = 50,
): Promise<{ items: Detection[]; total: number }> {
  const { data } = await request.get<ListResponse<Detection>>("/detections", {
    params: { page, pageSize },
  });
  return { items: data.data, total: data.total };
}

export async function getDetection(id: string): Promise<Detection> {
  const { data } = await request.get<{ data: Detection }>(`/detections/${id}`);
  return data.data;
}

export async function deleteDetection(id: string): Promise<void> {
  await request.post(`/detections/${id}/delete`);
}

export async function deleteBox(detectionId: string, boxId: string): Promise<void> {
  await request.post(`/detections/${detectionId}/boxes/${boxId}/delete`);
}

export async function addBox(
  detectionId: string,
  box: { className: string; x1: number; y1: number; x2: number; y2: number },
): Promise<void> {
  await request.post(`/detections/${detectionId}/boxes`, box);
}

export function exportSingleUrl(id: string): string {
  return `${API_BASE}/detections/${id}/export`;
}

export async function exportBatch(ids: string[], format = "yolo"): Promise<Blob> {
  const { data } = await request.post(
    "/detections/export-batch",
    { detectionIds: ids, format },
    { responseType: "blob" },
  );
  return data;
}

export async function exportAll(format = "yolo-seg"): Promise<Blob> {
  const { data } = await request.post(
    "/detections/export-all",
    { format },
    { responseType: "blob" },
  );
  return data;
}

// ── Training ────────────────────────────────────

export type YoloSeries = Record<string, { label: string; variants: Record<string, string> }>;

export async function fetchYoloSeries(): Promise<YoloSeries> {
  const { data } = await request.get<{ data: YoloSeries }>("/train/variants");
  return data.data;
}

export async function startTraining(params: {
  detectionIds: string[];
  modelVariant: string;
  epochs: number;
  imgsz: number;
  batch: number;
  trainRatio: number;
  valRatio: number;
  taskType: string;
}): Promise<TrainingJob> {
  const { data } = await request.post<{ data: TrainingJob }>("/train/jobs", params);
  return data.data;
}

export async function fetchTrainingJobs(
  page = 1,
  pageSize = 30,
): Promise<{ items: TrainingJob[]; total: number }> {
  const { data } = await request.get<ListResponse<TrainingJob>>("/train/jobs", {
    params: { page, pageSize },
  });
  return { items: data.data ?? [], total: data.total };
}

export async function cancelTrainingJob(id: string): Promise<TrainingJob> {
  const { data } = await request.post<{ data: TrainingJob }>(`/train/jobs/${id}/cancel`);
  return data.data;
}

export async function renameTrainingJob(id: string, name: string): Promise<TrainingJob> {
  const { data } = await request.post<{ data: TrainingJob }>(`/train/jobs/${id}/rename`, { name });
  return data.data;
}

export async function deleteTrainingJob(id: string): Promise<void> {
  await request.post(`/train/jobs/${id}/delete`);
}

// ── Utils ───────────────────────────────────────

export async function saveFilterSettings(
  detectionId: string,
  filterMode: string,
  filterNmsIou: number | null,
): Promise<void> {
  await request.put(`/detections/${detectionId}/filter-settings`, {
    filterMode,
    filterNmsIou,
  });
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Video ───────────────────────────────────────

export async function uploadVideo(file: File): Promise<VideoInfo> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await request.post<{ data: VideoInfo }>("/videos/upload", form, {
    timeout: UPLOAD_TIMEOUT,
  });
  return data.data;
}

export async function listVideos(
  page = 1,
  pageSize = 30,
): Promise<{ items: VideoInfo[]; total: number }> {
  const { data } = await request.get<ListResponse<VideoInfo>>("/videos", {
    params: { page, pageSize },
  });
  return { items: data.data, total: data.total };
}

export async function getVideo(id: string): Promise<VideoInfo> {
  const { data } = await request.get<{ data: VideoInfo }>(`/videos/${id}`);
  return data.data;
}

export async function extractKeyframes(
  videoId: string,
  params: {
    method: string;
    threshold?: number;
    intervalSeconds?: number;
    maxFrames?: number;
    ssimThreshold?: number;
  },
): Promise<VideoInfo> {
  const { data } = await request.post<{ data: VideoInfo }>(
    `/videos/${videoId}/extract-keyframes`,
    params,
  );
  return data.data;
}

export async function deleteVideo(id: string): Promise<void> {
  await request.post(`/videos/${id}/delete`);
}

export async function deleteAllVideos(): Promise<void> {
  await request.post("/videos/delete-bulk");
}

export function keyframeImageUrl(videoId: string, keyframeId: string): string {
  return `${API_BASE}/videos/${videoId}/keyframes/${keyframeId}/image`;
}

export function downloadModelUrl(jobId: string): string {
  return `${API_BASE}/train/jobs/${jobId}/download`;
}

export function chartUrl(jobId: string): string {
  return `${API_BASE}/train/jobs/${jobId}/charts/results.png`;
}

export function downloadOnnxUrl(jobId: string): string {
  return `${API_BASE}/train/jobs/${jobId}/export-onnx`;
}

export function downloadDatasetUrl(jobId: string): string {
  return `${API_BASE}/train/jobs/${jobId}/dataset`;
}

// ── Model ────────────────────────────────────────

export interface ModelStatus {
  loaded: boolean;
  state: "unloaded" | "downloading" | "loading" | "loaded" | "error";
  stage: string;
  progress: number;
  error: string;
}

export async function getModelStatus(): Promise<ModelStatus> {
  const { data } = await request.get<{ data: ModelStatus }>("/model/status");
  return data.data;
}

export async function unloadModel(): Promise<void> {
  await request.post("/model/unload");
}

export async function getSam2Status(): Promise<ModelStatus> {
  const { data } = await request.get<{ data: ModelStatus }>("/model/sam2/status");
  return data.data;
}

export async function unloadSam2(): Promise<void> {
  await request.post("/model/sam2/unload");
}

export interface Sam3Status {
  loaded: boolean;
  status: string; // "starting" | "loading" | "loaded" | "unloaded"
}

export async function checkSam3Health(): Promise<Sam3Status> {
  try {
    const resp = await request.get("/model/sam3/status");
    const inner = resp.data?.data;
    return {
      loaded: inner?.loaded === true,
      status: inner?.status || "unloaded",
    };
  } catch {
    return { loaded: false, status: "unloaded" };
  }
}

export async function unloadSam3(): Promise<void> {
  await request.post("/model/sam3/unload");
}

// ── Dataset Import ────────────────────────────────

export interface ImportResult {
  importId: string;
  status: string;
}

export interface ImportProgress {
  total: number;
  completed: number;
  status: string;
  detectionIds?: string[];
  error?: string | null;
}

export interface ChunkInitResult {
  uploadId: string;
  totalChunks: number;
  uploadedChunks: number[];
}

export async function importChunkInit(
  fileName: string,
  totalSize: number,
  format: string,
): Promise<ChunkInitResult> {
  const { data } = await request.post<{ data: ChunkInitResult }>(
    "/datasets/import/chunk/init",
    { fileName, totalSize, chunkSize: 5 * 1024 * 1024, format },
  );
  return data.data;
}

export async function importChunkComplete(uploadId: string, format: string): Promise<ImportResult> {
  const { data } = await request.post<{ data: ImportResult }>(
    `/datasets/import/chunk/${uploadId}/complete`,
    null,
    { params: { format } },
  );
  return data.data;
}

export async function importChunkCancel(uploadId: string): Promise<void> {
  await request.post(`/datasets/import/chunk/${uploadId}/cancel`);
}

export async function importDataset(
  file: File,
  format: string,
): Promise<ImportResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("format", format);
  const { data } = await request.post<{ data: ImportResult }>("/datasets/import", form, {
    timeout: UPLOAD_TIMEOUT,
  });
  return data.data;
}

export async function fetchImportProgress(importId: string): Promise<ImportProgress> {
  const { data } = await request.get<{ data: ImportProgress }>(
    `/datasets/import/${importId}/progress`,
  );
  return data.data;
}

export async function cancelImport(importId: string): Promise<void> {
  await request.post(`/datasets/import/${importId}/cancel`);
}


// ── PCS DAT Companion ────────────────────────────

interface PcsDatRequest {
  path: string;
  lidarIndex: number;
  lidarStreamName?: string | null;
  cameraStreamName?: string | null;
  admaStreamName?: string | null;
}

export async function inspectPcsDat(path: string): Promise<PcsDatInspection> {
  const { data } = await request.post("/autoannotation/dat/inspect", { path });
  return {
    path: data.path,
    streams: (data.streams ?? []).map((stream: Record<string, unknown>) => ({
      name: String(stream.name ?? ""),
      streamId: Number(stream.stream_id ?? 0),
      kind: String(stream.kind ?? "unknown"),
      frameCount: Number(stream.frame_count ?? 0),
      width: stream.width == null ? null : Number(stream.width),
      height: stream.height == null ? null : Number(stream.height),
      fps: stream.fps == null ? null : Number(stream.fps),
    })),
    lidarStreams: data.lidar_streams ?? [],
    cameraStreams: data.camera_streams ?? [],
    admaStreams: data.adma_streams ?? [],
    admaNativeApi: data.adma_native_api === true,
  };
}

export async function fetchPcsDatSample(
  params: PcsDatRequest,
): Promise<PcsDatSampleSummary> {
  const { data } = await request.post(
    "/autoannotation/dat/sample-summary",
    {
      path: params.path,
      lidar_index: params.lidarIndex,
      lidar_stream_name: params.lidarStreamName ?? null,
      camera_stream_name: params.cameraStreamName ?? null,
      adma_stream_name: params.admaStreamName ?? null,
      require_adma: true,
    },
    { timeout: DETECT_TIMEOUT },
  );
  return {
    sampleId: data.sample_id,
    timestampNs: data.timestamp_ns,
    lidar: {
      pointCount: data.lidar.point_count,
      attributes: data.lidar.attributes ?? [],
      metadata: data.lidar.metadata ?? {},
    },
    camera: {
      timestampNs: data.camera.timestamp_ns,
      width: data.camera.width,
      height: data.camera.height,
      encoding: data.camera.encoding,
      sourceId: data.camera.source_id,
      syncDeltaNs: data.camera.sync_delta_ns ?? null,
    },
    adma: data.adma == null ? null : {
      timestampNs: data.adma.timestamp_ns,
      syncStatus: data.adma.sync_status ?? null,
      syncDeltaNs: data.adma.sync_delta_ns ?? null,
      syncToleranceNs: data.adma.sync_tolerance_ns ?? null,
      sampleIndex: data.adma.sample_index ?? null,
      streamName: data.adma.stream_name ?? null,
      values: data.adma.values ?? {},
    },
    sourceMetadata: data.source_metadata ?? {},
  };
}

export async function fetchPcsDatCameraFrame(
  params: PcsDatRequest,
): Promise<Blob> {
  const { data } = await request.post(
    "/autoannotation/dat/camera-frame",
    {
      path: params.path,
      lidar_index: params.lidarIndex,
      lidar_stream_name: params.lidarStreamName ?? null,
      camera_stream_name: params.cameraStreamName ?? null,
      adma_stream_name: params.admaStreamName ?? null,
    },
    { responseType: "blob", timeout: DETECT_TIMEOUT },
  );
  return data;
}

export async function fetchPcsDatInnov3Proposals(
  params: PcsDatRequest,
): Promise<PcsInnov3ProposalResult> {
  const { data } = await request.post(
    "/autoannotation/dat/innov3-proposals",
    {
      path: params.path,
      lidar_index: params.lidarIndex,
      lidar_stream_name: params.lidarStreamName ?? null,
      camera_stream_name: params.cameraStreamName ?? null,
      adma_stream_name: params.admaStreamName ?? null,
    },
    { timeout: DETECT_TIMEOUT },
  );
  return {
    sampleId: data.sample_id,
    timestampNs: data.timestamp_ns,
    provider: data.provider,
    proposals: (data.proposals ?? []).map((proposal: any) => ({
      proposalId: proposal.proposal_id,
      className: proposal.class_name,
      bbox3d: proposal.bbox_3d == null ? null : {
        centerX: proposal.bbox_3d.center_x,
        centerY: proposal.bbox_3d.center_y,
        centerZ: proposal.bbox_3d.center_z,
        length: proposal.bbox_3d.length,
        width: proposal.bbox_3d.width,
        height: proposal.bbox_3d.height,
        yawRad: proposal.bbox_3d.yaw_rad,
      },
      confidence: proposal.confidence ?? null,
      evidence: proposal.evidence ?? [],
    })),
  };
}

export async function fetchPcsDatCameraProposals(
  params: PcsDatRequest & {
    categories: string[];
    useSam2: boolean;
    sam2ScoreThreshold: number;
  },
): Promise<PcsCameraProposalResult> {
  const { data } = await request.post(
    "/autoannotation/dat/camera-proposals",
    {
      path: params.path,
      lidar_index: params.lidarIndex,
      lidar_stream_name: params.lidarStreamName ?? null,
      camera_stream_name: params.cameraStreamName ?? null,
      adma_stream_name: params.admaStreamName ?? null,
      categories: params.categories,
      use_sam2: params.useSam2,
      sam2_score_threshold: params.sam2ScoreThreshold,
    },
    { timeout: DETECT_TIMEOUT },
  );
  return {
    sampleId: data.sample_id,
    timestampNs: data.timestamp_ns,
    cameraTimestampNs: data.camera_timestamp_ns,
    provider: data.provider,
    proposals: (data.proposals ?? []).map((proposal: any) => ({
      proposalId: proposal.proposal_id,
      className: proposal.class_name,
      bbox2d: proposal.bbox_2d == null ? null : {
        x1: proposal.bbox_2d.x1,
        y1: proposal.bbox_2d.y1,
        x2: proposal.bbox_2d.x2,
        y2: proposal.bbox_2d.y2,
      },
      maskPolygon: proposal.mask_polygon ?? null,
      confidence: proposal.confidence ?? null,
      evidence: proposal.evidence ?? [],
    })),
  };
}
