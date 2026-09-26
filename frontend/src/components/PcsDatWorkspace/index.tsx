import { useCallback, useEffect, useMemo, useState } from "react";
import { CategoryInput } from "@/components/CategoryInput";
import { DetectionCanvas } from "@/components/DetectionCanvas";
import {
  fetchPcsDatCameraFrame,
  fetchPcsDatCameraProposals,
  fetchPcsDatInnov3Proposals,
  fetchPcsDatSample,
  inspectPcsDat,
} from "@/services/api";
import { useAppStore } from "@/store/useAppStore";
import type {
  BBox,
  PcsCameraProposalResult,
  PcsDatInspection,
  PcsDatSampleSummary,
  PcsInnov3ProposalResult,
} from "@/types";

const DAT_PATH_STORAGE_KEY = "pcsAutoannotation.datPath";

export function PcsDatWorkspace() {
  const { categories, setCategories } = useAppStore();
  const [path, setPath] = useState(() => localStorage.getItem(DAT_PATH_STORAGE_KEY) ?? "");
  const [inspection, setInspection] = useState<PcsDatInspection | null>(null);
  const [lidarStream, setLidarStream] = useState("");
  const [cameraStream, setCameraStream] = useState("");
  const [admaStream, setAdmaStream] = useState("");
  const [frameIndex, setFrameIndex] = useState(0);
  const [sample, setSample] = useState<PcsDatSampleSummary | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [cameraResult, setCameraResult] = useState<PcsCameraProposalResult | null>(null);
  const [innov3Result, setInnov3Result] = useState<PcsInnov3ProposalResult | null>(null);
  const [useCameraAi, setUseCameraAi] = useState(true);
  const [useInnov3, setUseInnov3] = useState(true);
  const [useSam2, setUseSam2] = useState(true);
  const [sam2Threshold, setSam2Threshold] = useState(0);
  const [loadingSource, setLoadingSource] = useState(false);
  const [loadingFrame, setLoadingFrame] = useState(false);
  const [running, setRunning] = useState(false);
  const [runStage, setRunStage] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (categories.length === 0) setCategories(["car", "truck"]);
  }, [categories.length, setCategories]);

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

  const selectedLidarInfo = useMemo(
    () => inspection?.streams.find((stream) => stream.name === lidarStream) ?? null,
    [inspection, lidarStream],
  );
  const maxFrameIndex = Math.max(0, (selectedLidarInfo?.frameCount ?? 1) - 1);

  const requestParams = useCallback(
    (index: number) => ({
      path,
      lidarIndex: index,
      lidarStreamName: lidarStream || null,
      cameraStreamName: cameraStream || null,
      admaStreamName: admaStream || null,
    }),
    [path, lidarStream, cameraStream, admaStream],
  );

  const loadFrame = useCallback(
    async (
      index: number,
      streams?: { lidar: string; camera: string; adma: string },
    ) => {
      const selected = streams ?? {
        lidar: lidarStream,
        camera: cameraStream,
        adma: admaStream,
      };
      if (!path || !selected.lidar || !selected.camera || !selected.adma) return;

      setLoadingFrame(true);
      setError(null);
      setCameraResult(null);
      setInnov3Result(null);
      try {
        const params = {
          path,
          lidarIndex: index,
          lidarStreamName: selected.lidar,
          cameraStreamName: selected.camera,
          admaStreamName: selected.adma,
        };
        const [summary, imageBlob] = await Promise.all([
          fetchPcsDatSample(params),
          fetchPcsDatCameraFrame(params),
        ]);
        const nextUrl = URL.createObjectURL(imageBlob);
        setImageUrl((previous) => {
          if (previous) URL.revokeObjectURL(previous);
          return nextUrl;
        });
        setFrameIndex(index);
        setSample(summary);
      } catch (exc) {
        setError(exc instanceof Error ? exc.message : "Failed to load PCS DAT frame");
      } finally {
        setLoadingFrame(false);
      }
    },
    [path, lidarStream, cameraStream, admaStream],
  );

  const loadSource = useCallback(async () => {
    const trimmed = path.trim();
    if (!trimmed) {
      setError("Enter a DAT path available to the PCS Auto-Annotation server.");
      return;
    }
    setLoadingSource(true);
    setError(null);
    setInspection(null);
    setSample(null);
    setCameraResult(null);
    setInnov3Result(null);
    try {
      const result = await inspectPcsDat(trimmed);
      if (!result.lidarStreams.length) throw new Error("DAT contains no PCS point-cloud stream");
      if (!result.cameraStreams.length) throw new Error("DAT contains no PCS camera stream");
      if (!result.admaStreams.length) throw new Error("DAT contains no PCS ADMA stream");

      const streams = {
        lidar: result.lidarStreams[0],
        camera: result.cameraStreams[0],
        adma: result.admaStreams[0],
      };
      localStorage.setItem(DAT_PATH_STORAGE_KEY, trimmed);
      setPath(trimmed);
      setInspection(result);
      setLidarStream(streams.lidar);
      setCameraStream(streams.camera);
      setAdmaStream(streams.adma);
      await loadFrame(0, streams);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Failed to inspect PCS DAT source");
    } finally {
      setLoadingSource(false);
    }
  }, [path, loadFrame]);

  const runAnnotations = useCallback(async () => {
    if (!inspection || !sample) return;
    if (useCameraAi && categories.length === 0) {
      setError("Add at least one camera target category.");
      return;
    }
    if (!useCameraAi && !useInnov3) {
      setError("Enable at least one annotation provider.");
      return;
    }

    setRunning(true);
    setError(null);
    const failures: string[] = [];
    const params = requestParams(frameIndex);

    if (useInnov3) {
      setRunStage("Running Innov3 on PCS LiDAR frame…");
      try {
        setInnov3Result(await fetchPcsDatInnov3Proposals(params));
      } catch (exc) {
        setInnov3Result(null);
        failures.push(`Innov3: ${exc instanceof Error ? exc.message : "failed"}`);
      }
    } else {
      setInnov3Result(null);
    }

    if (useCameraAi) {
      setRunStage(useSam2 ? "Running LocateAnything + SAM2…" : "Running LocateAnything…");
      try {
        setCameraResult(
          await fetchPcsDatCameraProposals({
            ...params,
            categories,
            useSam2,
            sam2ScoreThreshold: sam2Threshold,
          }),
        );
      } catch (exc) {
        setCameraResult(null);
        failures.push(`Camera AI: ${exc instanceof Error ? exc.message : "failed"}`);
      }
    } else {
      setCameraResult(null);
    }

    setRunStage("");
    setRunning(false);
    if (failures.length) setError(failures.join(" | "));
  }, [
    inspection,
    sample,
    useCameraAi,
    useInnov3,
    categories,
    requestParams,
    frameIndex,
    useSam2,
    sam2Threshold,
  ]);

  const cameraBoxes = useMemo<BBox[]>(
    () =>
      (cameraResult?.proposals ?? [])
        .filter((proposal) => proposal.bbox2d != null)
        .map((proposal) => ({
          id: proposal.proposalId,
          className: proposal.className,
          x1: proposal.bbox2d!.x1,
          y1: proposal.bbox2d!.y1,
          x2: proposal.bbox2d!.x2,
          y2: proposal.bbox2d!.y2,
          confidence: proposal.confidence,
          maskPolygon: proposal.maskPolygon,
        })),
    [cameraResult],
  );

  const moveFrame = useCallback(
    (delta: number) => {
      const next = Math.min(maxFrameIndex, Math.max(0, frameIndex + delta));
      if (next !== frameIndex) void loadFrame(next);
    },
    [frameIndex, maxFrameIndex, loadFrame],
  );

  return (
    <div className="mx-auto w-full max-w-[1500px] space-y-4">
      <section className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-lg font-semibold text-gray-800">PCS DAT Companion</h1>
            <p className="mt-1 text-xs text-gray-500">
              PCS native owns DAT decoding and synchronization. This view only orchestrates annotation evidence.
            </p>
          </div>
          <div className="rounded border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
            Camera–LiDAR calibration WIP in PCS · fusion disabled
          </div>
        </div>

        <div className="mt-4 flex gap-2">
          <input
            value={path}
            onChange={(event) => setPath(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") void loadSource();
            }}
            placeholder="/path/to/trace.dat on the server"
            className="min-w-0 flex-1 rounded border border-gray-300 px-3 py-2 text-sm outline-none focus:border-primary-500"
          />
          <button
            type="button"
            disabled={loadingSource || !path.trim()}
            onClick={() => void loadSource()}
            className="rounded bg-primary-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
          >
            {loadingSource ? "Opening…" : "Open DAT"}
          </button>
        </div>

        {inspection && (
          <div className="mt-3 grid gap-3 md:grid-cols-3">
            <StreamSelect
              label="LiDAR"
              value={lidarStream}
              options={inspection.lidarStreams}
              onChange={(value) => setLidarStream(value)}
            />
            <StreamSelect
              label="Camera"
              value={cameraStream}
              options={inspection.cameraStreams}
              onChange={(value) => setCameraStream(value)}
            />
            <StreamSelect
              label="ADMA"
              value={admaStream}
              options={inspection.admaStreams}
              onChange={(value) => setAdmaStream(value)}
            />
          </div>
        )}
      </section>

      {inspection && (
        <section className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              disabled={loadingFrame || frameIndex <= 0}
              onClick={() => moveFrame(-1)}
              className="rounded border border-gray-300 px-3 py-1.5 text-sm disabled:opacity-40"
            >
              Previous
            </button>
            <input
              type="number"
              min={0}
              max={maxFrameIndex}
              value={frameIndex}
              onChange={(event) => {
                const next = Number(event.target.value);
                if (Number.isFinite(next)) {
                  setFrameIndex(Math.min(maxFrameIndex, Math.max(0, Math.trunc(next))));
                }
              }}
              className="w-24 rounded border border-gray-300 px-2 py-1.5 text-sm"
            />
            <span className="text-sm text-gray-500">/ {maxFrameIndex}</span>
            <button
              type="button"
              disabled={loadingFrame}
              onClick={() => void loadFrame(frameIndex)}
              className="rounded border border-primary-300 px-3 py-1.5 text-sm text-primary-700 disabled:opacity-40"
            >
              {loadingFrame ? "Loading…" : "Load frame"}
            </button>
            <button
              type="button"
              disabled={loadingFrame || frameIndex >= maxFrameIndex}
              onClick={() => moveFrame(1)}
              className="rounded border border-gray-300 px-3 py-1.5 text-sm disabled:opacity-40"
            >
              Next
            </button>
            <input
              type="range"
              min={0}
              max={maxFrameIndex}
              value={frameIndex}
              onChange={(event) => setFrameIndex(Number(event.target.value))}
              className="min-w-[220px] flex-1 accent-primary-600"
            />
          </div>
        </section>
      )}

      {sample && (
        <>
          <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <Metric label="LiDAR points" value={sample.lidar.pointCount.toLocaleString()} />
            <Metric
              label="Camera sync"
              value={formatDelta(sample.camera.syncDeltaNs)}
              detail={cameraStream}
            />
            <Metric
              label="ADMA sync"
              value={formatDelta(sample.adma?.syncDeltaNs ?? null)}
              detail={sample.adma?.syncStatus ?? "missing"}
            />
            <Metric
              label="Topology"
              value={`${sample.lidar.metadata.num_echoes ?? "?"} echoes`}
              detail={String(sample.lidar.metadata.structure_kind ?? "PCS native")}
            />
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
            <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div>
                  <h2 className="font-semibold text-gray-800">Camera evidence</h2>
                  <p className="text-xs text-gray-500">
                    {cameraResult
                      ? `${cameraResult.proposals.length} proposals · ${cameraResult.provider}`
                      : "Load a frame, then run camera annotation."}
                  </p>
                </div>
                {cameraResult && (
                  <span className="rounded bg-green-50 px-2 py-1 text-xs text-green-700">
                    visible in browser
                  </span>
                )}
              </div>
              {imageUrl ? (
                <DetectionCanvas
                  imageUrl={imageUrl}
                  boxes={cameraBoxes}
                  imgWidth={sample.camera.width}
                  imgHeight={sample.camera.height}
                  mode="view"
                  hiddenIndices={new Set()}
                  onModeChange={() => undefined}
                  onDrawBox={() => undefined}
                  readOnly
                />
              ) : (
                <div className="flex h-64 items-center justify-center rounded bg-gray-50 text-sm text-gray-400">
                  Camera frame unavailable
                </div>
              )}
            </div>

            <aside className="space-y-4">
              <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
                <h2 className="font-semibold text-gray-800">Annotation providers</h2>
                <label className="mt-3 flex items-center gap-2 text-sm text-gray-700">
                  <input
                    type="checkbox"
                    checked={useInnov3}
                    onChange={(event) => setUseInnov3(event.target.checked)}
                  />
                  Innov3 DSVT · LiDAR 3D
                </label>
                <label className="mt-2 flex items-center gap-2 text-sm text-gray-700">
                  <input
                    type="checkbox"
                    checked={useCameraAi}
                    onChange={(event) => setUseCameraAi(event.target.checked)}
                  />
                  LocateAnything · camera
                </label>
                {useCameraAi && (
                  <label className="mt-2 flex items-center gap-2 pl-6 text-xs text-gray-600">
                    <input
                      type="checkbox"
                      checked={useSam2}
                      onChange={(event) => setUseSam2(event.target.checked)}
                    />
                    Refine with SAM2 masks
                  </label>
                )}
                {useCameraAi && useSam2 && (
                  <div className="mt-2 pl-6">
                    <div className="mb-1 text-xs text-gray-500">
                      SAM2 score ≥ {sam2Threshold.toFixed(1)}
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.1}
                      value={sam2Threshold}
                      onChange={(event) => setSam2Threshold(Number(event.target.value))}
                      className="w-full accent-primary-600"
                    />
                  </div>
                )}
                <p className="mt-3 text-[11px] leading-4 text-gray-400">
                  SAM3 remains available in Image/Video mode. It is not part of the current PCS DAT baseline.
                </p>
              </div>

              <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
                <h2 className="font-semibold text-gray-800">Camera targets</h2>
                <div className="mt-3">
                  <CategoryInput
                    categories={categories}
                    onChange={setCategories}
                    disabled={running}
                    recentCategories={[]}
                  />
                </div>
                <button
                  type="button"
                  disabled={running || loadingFrame || (!useCameraAi && !useInnov3)}
                  onClick={() => void runAnnotations()}
                  className="mt-4 w-full rounded bg-primary-600 py-2.5 text-sm font-semibold text-white disabled:opacity-50"
                >
                  {running ? runStage || "Running…" : "Run annotations"}
                </button>
              </div>

              <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-xs text-amber-900">
                <div className="font-semibold">Fusion / PCS review</div>
                <div className="mt-1 leading-5">
                  Camera and LiDAR proposals are currently independent. The association and
                  <code className="mx-1">pcs.scene_objects</code>
                  handoff are implemented but intentionally gated until PCS calibration is verified.
                </div>
              </div>
            </aside>
          </section>

          <section className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="font-semibold text-gray-800">Innov3 LiDAR proposals</h2>
                <p className="text-xs text-gray-500">
                  3D boxes stay authoritative in PCS; this table is evidence only.
                </p>
              </div>
              <span className="text-sm font-medium text-gray-600">
                {innov3Result?.proposals.length ?? 0} proposals
              </span>
            </div>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full min-w-[780px] text-left text-xs">
                <thead className="border-b border-gray-200 text-gray-500">
                  <tr>
                    <th className="py-2 pr-3">Class</th>
                    <th className="py-2 pr-3">Confidence</th>
                    <th className="py-2 pr-3">Center XYZ [m]</th>
                    <th className="py-2 pr-3">L × W × H [m]</th>
                    <th className="py-2">Yaw [rad]</th>
                  </tr>
                </thead>
                <tbody>
                  {(innov3Result?.proposals ?? []).map((proposal) => (
                    <tr key={proposal.proposalId} className="border-b border-gray-100 text-gray-700">
                      <td className="py-2 pr-3 font-medium">{proposal.className}</td>
                      <td className="py-2 pr-3">{formatConfidence(proposal.confidence)}</td>
                      <td className="py-2 pr-3">
                        {proposal.bbox3d
                          ? formatTriple(
                              proposal.bbox3d.centerX,
                              proposal.bbox3d.centerY,
                              proposal.bbox3d.centerZ,
                            )
                          : "—"}
                      </td>
                      <td className="py-2 pr-3">
                        {proposal.bbox3d
                          ? formatTriple(
                              proposal.bbox3d.length,
                              proposal.bbox3d.width,
                              proposal.bbox3d.height,
                            )
                          : "—"}
                      </td>
                      <td className="py-2">{proposal.bbox3d?.yawRad.toFixed(3) ?? "—"}</td>
                    </tr>
                  ))}
                  {!innov3Result?.proposals.length && (
                    <tr>
                      <td colSpan={5} className="py-6 text-center text-gray-400">
                        No Innov3 proposals loaded for this frame.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      {error && (
        <div className="rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}
    </div>
  );
}

function StreamSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="text-xs font-medium text-gray-600">
      {label}
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="mt-1 w-full rounded border border-gray-300 bg-white px-2 py-2 text-sm font-normal text-gray-700"
      >
        {options.map((option) => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </label>
  );
}

function Metric({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="mt-1 text-lg font-semibold text-gray-800">{value}</div>
      {detail && <div className="mt-1 truncate text-[11px] text-gray-400">{detail}</div>}
    </div>
  );
}

function formatDelta(value: number | null): string {
  return value == null ? "—" : `${(Math.abs(value) / 1_000_000).toFixed(3)} ms`;
}

function formatConfidence(value: number | null): string {
  return value == null ? "—" : value.toFixed(3);
}

function formatTriple(x: number, y: number, z: number): string {
  return `${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}`;
}
