export interface BBox {
  id: string;
  className: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number | null;
  maskPolygon?: number[][] | null;
}

export interface Detection {
  id: string;
  imageName: string;
  categories: string[];
  modelName: string;
  modelType: string | null;
  imageWidth: number;
  imageHeight: number;
  elapsedMs: number | null;
  filterMode: string | null;
  filterNmsIou: number | null;
  status: string;
  createdAt: string;
  boxes: BBox[];
}

export type DetectResponse = Detection;

export interface TrainingJob {
  id: string;
  name: string | null;
  modelVariant: string;
  epochs: number;
  imgsz: number;
  batch: number;
  trainRatio: number;
  valRatio: number;
  taskType: string;
  detectionIds: string[];
  classMap: Record<string, string> | null;
  status: "pending" | "running" | "completed" | "failed";
  metrics: Record<string, unknown> | null;
  modelPath: string | null;
  onnxPath: string | null;
  errorMessage: string | null;
  createdAt: string;
  completedAt: string | null;
}

export interface KeyFrame {
  id: string;
  videoId: string;
  frameNumber: number;
  timestampSeconds: number;
  sceneScore: number | null;
  createdAt: string;
}

export interface VideoInfo {
  id: string;
  fileName: string;
  duration: number | null;
  fps: number | null;
  totalFrames: number | null;
  width: number | null;
  height: number | null;
  status: string;
  createdAt: string;
  keyframes: KeyFrame[];
}

export interface VideoList {
  total: number;
  items: VideoInfo[];
}

export interface ListResponse<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
}


export interface PcsDatStream {
  name: string;
  streamId: number;
  kind: string;
  frameCount: number;
  width: number | null;
  height: number | null;
  fps: number | null;
}

export interface PcsDatInspection {
  path: string;
  streams: PcsDatStream[];
  lidarStreams: string[];
  cameraStreams: string[];
  admaStreams: string[];
  admaNativeApi: boolean;
}

export interface PcsDatSampleSummary {
  sampleId: string;
  timestampNs: number;
  lidar: {
    pointCount: number;
    attributes: string[];
    metadata: Record<string, unknown>;
  };
  camera: {
    timestampNs: number;
    width: number;
    height: number;
    encoding: string;
    sourceId: string;
    syncDeltaNs: number | null;
  };
  adma: null | {
    timestampNs: number;
    syncStatus: string | null;
    syncDeltaNs: number | null;
    syncToleranceNs: number | null;
    sampleIndex: number | null;
    streamName: string | null;
    values: Record<string, unknown>;
  };
  sourceMetadata: Record<string, unknown>;
}

export interface PcsCameraProposal {
  proposalId: string;
  className: string;
  bbox2d: null | { x1: number; y1: number; x2: number; y2: number };
  maskPolygon: number[][] | null;
  confidence: number | null;
  evidence: Array<Record<string, unknown>>;
}

export interface PcsCameraProposalResult {
  sampleId: string;
  timestampNs: number;
  cameraTimestampNs: number;
  provider: string;
  proposals: PcsCameraProposal[];
}

export interface PcsInnov3Proposal {
  proposalId: string;
  className: string;
  bbox3d: null | {
    centerX: number;
    centerY: number;
    centerZ: number;
    length: number;
    width: number;
    height: number;
    yawRad: number;
  };
  confidence: number | null;
  evidence: Array<Record<string, unknown>>;
}

export interface PcsInnov3ProposalResult {
  sampleId: string;
  timestampNs: number;
  provider: string;
  proposals: PcsInnov3Proposal[];
}
