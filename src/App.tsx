import * as React from "react";
import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import JSZip from "jszip";
import {
  Trash2, ZoomIn, ZoomOut, Settings, Play, FileArchive,
  CheckCircle2, Loader2, Cpu, Video, Target, Brain, Globe,
  Download, Shell, Waves, Undo, Save, X, ArrowLeft, ArrowRight,
  DownloadCloud, LayoutGrid, XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { ImageData, Detection, Annotation, VideoFrameCapture } from "./types";

// ─── Types ────────────────────────────────────────────────────────────────────
interface ModelEntry {
  id: string;
  name: string;
  type: string;
  status: string; // "Not loaded" | "Ready"
  path?: string;
  classNames?: string[]; // populated after model loads
}

const DEFAULT_MODELS: ModelEntry[] = [
  { id: "cots",      name: "CoTS Model",       type: "YOLOv8", status: "Not loaded" },
  { id: "clam",      name: "Giant Clam Model", type: "YOLOv8", status: "Not loaded" },
  { id: "bleaching", name: "Bleaching Model",  type: "YOLOv8", status: "Not loaded" },
  { id: "fish",      name: "Fish Model",        type: "YOLOv8", status: "Not loaded" },
];

// Detection colours by label keyword
function colorForLabel(label: string): string {
  const l = label.toLowerCase();
  if (l.includes("cot") || l.includes("starfish")) return "#ef4444";
  if (l.includes("clam"))                           return "#3b82f6";
  if (l.includes("bleach"))                         return "#f97316";
  if (l.includes("fish"))                           return "#a855f7";
  return "#10b981";
}

export default function App() {
  // ── Core state ──────────────────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState("models");

  // Models: always start cleared — user must upload their model each session.
  // We intentionally do not restore from localStorage so stale paths never auto-load.
  const [models, setModels] = useState<ModelEntry[]>(DEFAULT_MODELS);

  // Images always start empty — use Load Progress (header) to restore a session.
  const [images, setImages] = useState<ImageData[]>([]);

  const [geminiKey, setGeminiKey]   = useState(() => localStorage.getItem("reef_gemini_key") || "");
  const [confidence, setConfidence] = useState(() => parseFloat(localStorage.getItem("reef_confidence") || "0.55"));
  const [inferenceMode, setInferenceMode] = useState<"cloud" | "local">(() =>
    (localStorage.getItem("reef_inference_mode") as any) || "local"
  );
  const [filterMode, setFilterMode] = useState<"all" | "detected" | "zero" | "annotated">("all");
  const [folderPath, setFolderPath]   = useState("");
  const [folderScanning, setFolderScanning] = useState(false);
  const [folderRecursive, setFolderRecursive] = useState(false);
  const [ytUrl, setYtUrl]                     = useState("");
  const [ytInterval, setYtInterval]           = useState(5);
  const [ytMaxFrames, setYtMaxFrames]         = useState(200);
  const [ytRunning, setYtRunning]             = useState(false);
  const [ytProgress, setYtProgress]           = useState<{stage:string;message:string;frames?:number} | null>(null);
  const [isDraggingOver, setIsDraggingOver]   = useState(false);
  const [exportMode, setExportMode]           = useState<"annotated_only" | "all_processed">("annotated_only");
  const [augmentation, setAugmentation]       = useState<"off" | "standard" | "heavy">("heavy");
  const [modelName, setModelName]             = useState("");
  const [scrapeQuery, setScrapeQuery]       = useState("");
  const [updateAvailable, setUpdateAvailable] = useState<string | null>(null);

  // Check GitHub for latest version on mount
  useEffect(() => {
    fetch("https://raw.githubusercontent.com/ResilientReefsFoundation/aidetect/main/src/App.tsx")
      .then(r => r.text())
      .then(text => {
        const match = text.match(/v(\d+\.\d+)/);
        if (match) {
          const latest = match[1];
          const current = "4.43";
          if (latest !== current) setUpdateAvailable(latest);
        }
      })
      .catch(() => {}); // silent fail if offline
  }, []);

  const [trainingHistory, setTrainingHistory] = useState<{
    date: string; modelName: string; mAP50: number; epochs: number;
    datasetSize: number; baseModel: string; notes?: string;
  }[]>(() => {
    try { return JSON.parse(localStorage.getItem("reef_training_history") || "[]"); }
    catch { return []; }
  });
  const [scrapeMode, setScrapeMode]         = useState<"download" | "screenshot">("download");
  const [scrapeCommon, setScrapeCommon]     = useState("");
  const [scrapeCount, setScrapeCount]       = useState(50);
  const [scrapeRunning, setScrapeRunning]   = useState(false);
  const [scrapeProgress, setScrapeProgress] = useState<{found:number;downloaded:number;failed:number;status:string} | null>(null);

  // ── Training state ────────────────────────────────────────────────────────
  const [trainDatasets, setTrainDatasets]   = useState<{name:string;yaml_path:string;image_count:number}[]>([]);
  const [trainDatasetPath, setTrainDatasetPath] = useState("");
  const [trainBaseModel, setTrainBaseModel] = useState("");
  const [trainEpochs, setTrainEpochs]       = useState(50);
  const [trainImgSize, setTrainImgSize]     = useState(640);
  const [trainBatch, setTrainBatch]         = useState(-1);
  const [trainRunning, setTrainRunning]     = useState(false);
  const [trainProgress, setTrainProgress]   = useState<{epoch:number;total:number;mAP50:number;box_loss:number|null;cls_loss:number|null;precision:number;recall:number}[]>([]);
  const [trainDone, setTrainDone]           = useState<{model_path:string;model_name:string;mAP50?:number;epochs?:number}|null>(null);
  const [trainError, setTrainError]         = useState<string|null>(null);
  const [trainUploading, setTrainUploading] = useState(false);
  const trainEsRef = useRef<EventSource|null>(null);
  const trainFileRef = useRef<HTMLInputElement>(null);
  const [selectedImageId, setSelectedImageId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing]       = useState(false);
  const [processedCount, setProcessedCount]   = useState(0);
  const [totalToProcess, setTotalToProcess]   = useState(0);

  // ── Model picker for upload ─────────────────────────────────────────────────
  // Which model slot to upload into — default is COTS (first slot)
  const [activeModelId, setActiveModelId] = useState("cots");
  // Ref so the upload handler always reads the current value without closure issues
  const activeModelIdRef = useRef("cots");
  useEffect(() => { activeModelIdRef.current = activeModelId; }, [activeModelId]);

  // FIX: keep a ref to the latest models so runDetection never reads stale state
  const modelsRef = useRef<ModelEntry[]>(models);
  useEffect(() => { modelsRef.current = models; }, [models]);

  // Ref for images so runDetection can read URLs without triggering state updates
  const imagesRef = useRef<ImageData[]>(images);
  useEffect(() => { imagesRef.current = images; }, [images]);

  // FIX: keep a ref to the latest inferenceMode for the same reason
  const inferenceModeRef = useRef(inferenceMode);
  useEffect(() => { inferenceModeRef.current = inferenceMode; }, [inferenceMode]);

  const confidenceRef = useRef(confidence);
  useEffect(() => { confidenceRef.current = confidence; }, [confidence]);

  const geminiKeyRef = useRef(geminiKey);
  useEffect(() => { geminiKeyRef.current = geminiKey; }, [geminiKey]);

  // Concurrency guard — tracks which imageIds are currently being processed
  const processingRef = useRef(new Set<string>());
  // Allows cancelling a running batch
  const abortRef = useRef<AbortController | null>(null);

  // ── Annotate sub-mode: images or video ──────────────────────────────────────
  const [annotateMode, setAnnotateMode] = useState<"images" | "video">("images");

  // ── Per-image undo history ──────────────────────────────────────────────────
  const [historyMap, setHistoryMap] = useState<Record<string, Annotation[][]>>({});

  // ── Video ───────────────────────────────────────────────────────────────────
  const [videoUrl, setVideoUrl]               = useState<string | null>(null);
  const [videoDuration, setVideoDuration]     = useState(0);
  const [videoCurrentTime, setVideoCurrentTime] = useState(0);
  const [videoPlaying, setVideoPlaying]       = useState(false);
  const [captures, setCaptures]               = useState<VideoFrameCapture[]>([]);

  // ── Drawing ─────────────────────────────────────────────────────────────────
  const [zoom, setZoom]               = useState(1);
  const [isDrawing, setIsDrawing]     = useState(false);
  const [startPos, setStartPos]       = useState<{ x: number; y: number } | null>(null);
  const [currentBox, setCurrentBox]   = useState<[number, number, number, number] | null>(null);
  const [isPanning, setIsPanning]     = useState(false);
  const [panOffset, setPanOffset]     = useState({ x: 0, y: 0 });
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });

  // ── Refs ────────────────────────────────────────────────────────────────────
  const imageContainerRef = useRef<HTMLDivElement>(null);
  const canvasWrapperRef  = useRef<HTMLDivElement>(null); // outer div for non-passive wheel
  const videoRef          = useRef<HTMLVideoElement>(null);
  const fileInputRef      = useRef<HTMLInputElement>(null);
  const modelInputRef     = useRef<HTMLInputElement>(null);
  const videoInputRef     = useRef<HTMLInputElement>(null);
  const sessionInputRef   = useRef<HTMLInputElement>(null);

  // ── Persistence ─────────────────────────────────────────────────────────────
  // List of .pt/.onnx files already sitting in the models/ folder on disk
  const [savedModels, setSavedModels] = useState<{ name: string; path: string }[]>([]);
  // Per-slot URL input state: { [modelId]: { visible: bool, value: string } }
  const [urlInputs, setUrlInputs] = useState<Record<string, { visible: boolean; value: string }>>({});
  const [renamingModel, setRenamingModel] = useState<{name: string; value: string} | null>(null);

  const fetchTrainDatasets = () => {
    fetch("/api/train/datasets")
      .then(r => r.json())
      .then(d => { if (d.datasets) setTrainDatasets(d.datasets); })
      .catch(() => {});
  };

  // Do NOT auto-select a model — user must choose explicitly to avoid training from wrong base

  const scanLocalModels = useCallback(() => {
    fetch("/api/local-models")
      .then(r => r.json())
      .then(data => {
        if (data.models) setSavedModels(data.models);
      })
      .catch(() => {});
  }, []);

  const deleteModel = async (filename: string) => {
    if (!confirm("Delete " + filename + "? This cannot be undone.")) return;
    const r = await fetch("/api/model/delete", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename }),
    });
    if (r.ok) { toast.success("Deleted " + filename); scanLocalModels(); }
    else { const d = await r.json(); toast.error(d.error ?? "Delete failed"); }
  };

  const renameModel = async (oldName: string, newName: string) => {
    const r = await fetch("/api/model/rename", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ old_name: oldName, new_name: newName }),
    });
    const d = await r.json();
    if (r.ok) { toast.success("Renamed to " + d.new_name); scanLocalModels(); setRenamingModel(null); }
    else { toast.error(d.error ?? "Rename failed"); }
  };

  // Scan for saved models on mount so the dropdown is populated
  useEffect(() => { scanLocalModels(); }, [scanLocalModels]);

  // Activate a model that already exists on disk — no upload needed
  const loadSavedModel = async (savedPath: string, targetId: string) => {
    const fileName = savedPath.split(/[\/]/).pop() ?? savedPath;

    // Mark Ready immediately — don't wait for the class fetch
    setModels(prev => prev.map(m =>
      m.id === targetId
        ? { ...m, status: "Ready", name: fileName, path: savedPath }
        : m
    ));
    // model status badge updates

    // Fetch class names in the background (only works when Python is running)
    try {
      const classRes = await fetch("/api/model-classes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: savedPath }),
      });
      if (classRes.ok) {
        const classData = await classRes.json().catch(() => null);
        const classNames: string[] | undefined = classData?.classes;
        if (classNames?.length) {
          setModels(prev => prev.map(m =>
            m.id === targetId ? { ...m, classNames } : m
          ));
        }
      }
    } catch {
      // Python not running — class names unavailable, model still usable
    }
  };

  const loadModelFromUrl = async (url: string, targetId: string) => {
    if (!url.trim()) { toast.error("Please enter a URL"); return; }
    // Hide input and show loading
    setUrlInputs(prev => ({ ...prev, [targetId]: { visible: false, value: "" } }));
    const toastId = toast.loading("Downloading model…");
    try {
      const res = await fetch("/api/download-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url.trim() }),
      });
      const data = await res.json();
      if (!res.ok) { toast.error(data.error ?? "Download failed", { id: toastId }); return; }
      // Model downloaded — now load it
      const { name, path: modelPath } = data.model;
      toast.success(`Downloaded ${name}`, { id: toastId });
      // Mark Ready immediately
      setModels(prev => prev.map(m =>
        m.id === targetId ? { ...m, status: "Ready", name, path: modelPath } : m
      ));
      scanLocalModels(); // refresh the dropdown
      // Fetch class names in background
      try {
        const classRes = await fetch("/api/model-classes", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_path: modelPath }),
        });
        if (classRes.ok) {
          const classData = await classRes.json().catch(() => null);
          if (classData?.classes?.length) {
            setModels(prev => prev.map(m =>
              m.id === targetId ? { ...m, classNames: classData.classes } : m
            ));
          }
        }
      } catch { /* Python not running — fine */ }
    } catch (err: any) {
      toast.error(err?.message ?? "Download failed", { id: toastId });
    }
  };

  // No auto-scan on mount — models always start as 'Not loaded' until user uploads
  // Auto-save annotations to localStorage on every change (best effort)
  // Large base64 sessions may exceed quota — that's OK, we just skip silently
  const lsSaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    // Debounce — wait 3s after last change before writing
    if (lsSaveTimerRef.current) clearTimeout(lsSaveTimerRef.current);
    lsSaveTimerRef.current = setTimeout(() => {
      try {
        // Only save annotations and detections counts — not the full base64 URLs
        // This lets us at least recover annotation data even if images are gone
        const slim = images.map(img => ({
          id: img.id, name: img.name, status: img.status,
          annotations: img.annotations,
          detections: img.detections,
          // Only include url if it's a local path ref (tiny) not a huge base64 blob
          url: img.url.startsWith("/api/image?") ? img.url : "",
        }));
        localStorage.setItem("reef_annotations", JSON.stringify(slim));
      } catch { /* quota exceeded — silent fail */ }
    }, 3000);
  }, [images]);
  useEffect(() => { localStorage.setItem("reef_gemini_key",    geminiKey);                    }, [geminiKey]);
  useEffect(() => { localStorage.setItem("reef_inference_mode", inferenceMode);               }, [inferenceMode]);
  useEffect(() => { localStorage.setItem("reef_confidence", String(confidence));              }, [confidence]);

  // ── Derived ─────────────────────────────────────────────────────────────────
  const filteredImages = useMemo(() => {
    if (filterMode === "detected")  return images.filter(img => img.detections.length > 0);
    if (filterMode === "zero")      return images.filter(img => img.detections.length === 0);
    if (filterMode === "annotated") return images.filter(img => img.annotations.length > 0);
    return images;
  }, [images, filterMode]);

  const selectedImage = useMemo(
    () => images.find(img => img.id === selectedImageId) ?? null,
    [images, selectedImageId]
  );

  const currentImageHistory = selectedImage ? (historyMap[selectedImage.id] ?? []) : [];

  // FIX: wrap in useCallback with correct deps so keyboard handler is always fresh
  const navigateImages = useCallback((dir: "next" | "prev") => {
    if (!filteredImages.length) return;
    const idx  = filteredImages.findIndex(img => img.id === selectedImageId);
    const next = dir === "next"
      ? (idx < filteredImages.length - 1 ? idx + 1 : 0)
      : (idx > 0 ? idx - 1 : filteredImages.length - 1);
    setSelectedImageId(filteredImages[next].id);
    setZoom(1);
    setPanOffset({ x: 0, y: 0 });
  }, [filteredImages, selectedImageId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (activeTab !== "annotate" || annotateMode !== "images") return;
      if (document.activeElement?.tagName === "INPUT") return;
      if (e.key === "ArrowRight") navigateImages("next");
      if (e.key === "ArrowLeft")  navigateImages("prev");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeTab, navigateImages]);

  // ── Session ──────────────────────────────────────────────────────────────────
  // Auto-backup to disk file every 5 minutes if there are annotations
  // This is a safety net in case the browser crashes or localStorage fills up
  const saveSession = () => {
    // Save everything needed to fully restore a session including all annotations
    const data = {
      version:       "4.0",
      timestamp:     new Date().toISOString(),
      confidence,
      inferenceMode,
      // images contains .detections, .annotations, .status, .url (base64) per image
      images: images.map(img => ({
        ...img,
        // Ensure annotations always present even if empty
        annotations: img.annotations ?? [],
        detections:  img.detections  ?? [],
      })),
    };
    const json = JSON.stringify(data, null, 2);
    const url  = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    Object.assign(document.createElement("a"), {
      href: url,
      download: `reef_session_${new Date().toISOString().split("T")[0]}.json`,
    }).click();
    URL.revokeObjectURL(url);
    const annotCount = images.reduce((a, i) => a + (i.annotations?.length ?? 0), 0);
    const detCount   = images.reduce((a, i) => a + (i.detections?.length  ?? 0), 0);
    toast.success(`Session saved — ${images.length} images, ${detCount} detections, ${annotCount} annotations`);
  };

  const loadSession = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      try {
        const d = JSON.parse(ev.target?.result as string);
        if (!d.images) { toast.error("Invalid session file — no images found"); return; }

        // Restore images (includes detections + annotations)
        const loaded = (d.images as any[]).map(img => ({
          ...img,
          annotations: img.annotations ?? [],
          detections:  img.detections  ?? [],
          // Drop stale blob: URLs from very old sessions
          url: img.url?.startsWith("blob:") ? "" : img.url,
        })).filter(img => img.url); // drop any that lost their URL

        setImages(loaded);
        if (d.confidence)    setConfidence(d.confidence);
        if (d.inferenceMode) setInferenceMode(d.inferenceMode);
        // Don't restore models — always start fresh for safety

        const annotCount = loaded.reduce((a: number, i: any) => a + (i.annotations?.length ?? 0), 0);
        const detCount   = loaded.reduce((a: number, i: any) => a + (i.detections?.length  ?? 0), 0);
        toast.success(`Loaded ${loaded.length} images, ${detCount} detections, ${annotCount} annotations`);

        // Select first image automatically
        if (loaded.length > 0 && !selectedImageId) setSelectedImageId(loaded[0].id);
      } catch (err) {
        toast.error("Invalid session file — could not parse");
        console.error(err);
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  // ── File upload ──────────────────────────────────────────────────────────────


  // ── File upload ──────────────────────────────────────────────────────────────
  const loadFileBatch = async (files: File[]) => {
    if (!files.length) return;
    const mimeForExt = (name: string) => {
      const ext = name.split(".").pop()?.toLowerCase();
      if (ext === "png")  return "image/png";
      if (ext === "webp") return "image/webp";
      return "image/jpeg";
    };
    const BATCH = 20;
    let firstId: string | null = null;
    for (const file of files) {
      if (file.name.toLowerCase().endsWith(".zip")) {
        const toastId = toast.loading(`Extracting ${file.name}…`);
        try {
          const JSZipMod = (await import("jszip")).default;
          const zip = new JSZipMod();
          const contents = await zip.loadAsync(file);
          const entries: any[] = [];
          for (const fname in contents.files) {
            const entry = contents.files[fname];
            if (!entry.dir && /\.(jpe?g|png|webp)$/i.test(entry.name)) entries.push(entry);
          }
          for (let i = 0; i < entries.length; i += BATCH) {
            const batch = entries.slice(i, i + BATCH);
            const imgs = await Promise.all(batch.map(async (entry: any) => {
              const b64 = await entry.async("base64");
              const id = crypto.randomUUID();
              if (!firstId) firstId = id;
              return { id, name: entry.name, url: `data:${mimeForExt(entry.name)};base64,${b64}`, detections: [], annotations: [], status: "pending" as const };
            }));
            setImages(prev => [...prev, ...imgs]);
          }
          toast.success(`Extracted ${entries.length} images from ${file.name}`, { id: toastId });
        } catch (err: any) {
          toast.error(`Failed to unzip ${file.name}: ${err?.message ?? "unknown error"}`, { id: toastId });
        }
      } else if (/\.(jpe?g|png|webp)$/i.test(file.name)) {
        const img = await new Promise<ImageData>(resolve => {
          const r = new FileReader();
          r.onloadend = () => {
            const id = crypto.randomUUID();
            if (!firstId) firstId = id;
            resolve({ id, name: file.name, url: r.result as string, detections: [], annotations: [], status: "pending" });
          };
          r.readAsDataURL(file);
        });
        setImages(prev => [...prev, img]);
      }
    }
    if (firstId) setSelectedImageId((prev: string | null) => prev ?? firstId);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []) as File[];
    await loadFileBatch(files);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingOver(false);
    const items = Array.from(e.dataTransfer.items || []) as DataTransferItem[];
    const files: File[] = [];
    const readEntry = (entry: any): Promise<void> => {
      if (entry.isFile) {
        return new Promise(res => entry.file((f: File) => { files.push(f); res(); }));
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        return new Promise(res => {
          const readAll = () => reader.readEntries(async (entries: any[]) => {
            if (!entries.length) return res();
            await Promise.all(entries.map(readEntry));
            readAll();
          });
          readAll();
        });
      }
      return Promise.resolve();
    };
    if (items.length && items[0].webkitGetAsEntry) {
      await Promise.all(items.map(item => {
        const entry = item.webkitGetAsEntry();
        return entry ? readEntry(entry) : Promise.resolve();
      }));
    } else {
      files.push(...Array.from(e.dataTransfer.files) as File[]);
    }
    const imageFiles = files.filter(f => /\.(jpe?g|png|webp|zip)$/i.test(f.name));
    if (!imageFiles.length) { toast.error("No supported image files found"); return; }
    await loadFileBatch(imageFiles);
  };

  // ── Detection ────────────────────────────────────────────────────────────────
  // FIX: reads models/inferenceMode/confidence/geminiKey from REFS, not closure —
  // this is the root cause of "no model stays loaded" because these values were
  // stale inside the async loop.
  const runDetection = useCallback(async (imageId: string, signal?: AbortSignal) => {
    // Concurrency guard — skip if already in-flight
    if (processingRef.current.has(imageId)) return;
    processingRef.current.add(imageId);

    // Read image data directly from ref — no async state round-trip needed
    const img = imagesRef.current.find(i => i.id === imageId);
    if (!img) { processingRef.current.delete(imageId); return; }

    // Mark image as processing
    setImages(prev => prev.map(i =>
      i.id === imageId ? { ...i, status: "processing" } : i
    ));

    try {
      // Images are stored as base64 data URLs — use directly, no re-encoding needed
      // If image is a local path reference, fetch it first and convert to base64
      let base64 = img.url;
      if (img.url.startsWith("/api/image?")) {
        try {
          const resp = await fetch(img.url);
          const blob = await resp.blob();
          base64 = await new Promise<string>(res => {
            const r = new FileReader();
            r.onloadend = () => res(r.result as string);
            r.readAsDataURL(blob);
          });
        } catch {
          throw new Error("Could not read local image file");
        }
      }

      let detections: Detection[] = [];

      // FIX: read from refs — always current, never stale
      if (inferenceModeRef.current === "cloud") {
        const cloudRes = await fetch("/api/detect-cloud", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: base64, customKey: geminiKeyRef.current }),
        });
        if (!cloudRes.ok) {
          const err = await cloudRes.json().catch(() => ({ error: "Cloud detection failed" }));
          throw new Error(err.error);
        }
        detections = await cloudRes.json();
      } else {
        // Find first Ready model from ref
        const readyModel = modelsRef.current.find(m => m.status === "Ready" && m.path);
        if (!readyModel?.path) {
          toast.error("No model loaded — upload a .pt file in the MODELS tab first.");
          setImages(prev => prev.map(im => im.id === imageId ? { ...im, status: "idle" } : im));
          return;
        }
        const res = await fetch("/api/detect-local", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: base64, model_path: readyModel.path, confidence: confidenceRef.current }),
        });
        if (!res.ok) {
          const text = await res.text().catch(() => "");
          let errMsg = "Detection failed";
          try { errMsg = JSON.parse(text)?.error ?? text ?? "Detection failed"; } catch { errMsg = text || `HTTP ${res.status}`; }
          console.error(`Detection HTTP ${res.status}:`, errMsg);
          // 503 = Python backend not reachable — stop the whole batch
          if (res.status === 503) throw Object.assign(new Error("BACKEND_DOWN"), { message: "BACKEND_DOWN" });
          throw new Error(errMsg);
        }
        const results = await res.json();
        // Use the model slot name as the display label if it's more descriptive
        // e.g. "Fish Model" is clearer than "cots-detection" when detecting fish
        const slotName = readyModel?.name ?? "";
        const slotLabel = slotName
          .replace(/_best\.pt$/i, "")
          .replace(/reef_train_\d+_?/i, "")
          .replace(/_v\d+$/i, "")
          .replace(/[_-]/g, " ")
          .trim();
        detections = results.map((r: any, idx: number) => {
          // Use slot label if it's meaningful, otherwise use raw model class name
          const displayLabel = slotLabel && slotLabel.toLowerCase() !== r.label.toLowerCase()
            ? slotLabel : r.label;
          return {
            id:         `det-${Date.now()}-${idx}`,
            label:      displayLabel,
            confidence: r.confidence,
            bbox:       r.bbox as [number, number, number, number],
            color:      colorForLabel(displayLabel),
          };
        });
      }

      setImages(prev => prev.map(im =>
        im.id === imageId ? { ...im, detections, status: "completed" } : im
      ));
    } catch (error: any) {
      console.error("Detection error:", error);
      setImages(prev => prev.map(im => im.id === imageId ? { ...im, status: "error" } : im));
      if (error.message === "BACKEND_DOWN") {
        throw error; // let runAllDetections handle this and stop the batch
      } else if (error.message === "MISSING_API_KEY") {
        toast.error("Gemini API key missing — add it in the MODELS tab.");
      } else {
        toast.error(error.message?.length > 120 ? error.message.slice(0, 120) + "…" : error.message);
      }
    } finally {
      // Always release the concurrency lock
      processingRef.current.delete(imageId);
    }
  }, []); // no deps — reads everything from refs

  const runAllDetections = async () => {
    const pendingIds = images.filter(img => img.status === "pending").map(img => img.id);
    if (!pendingIds.length) { toast.info("No pending images to process"); return; }

    // Preflight: check Python backend is up before hammering all images
    if (inferenceModeRef.current === "local") {
      try {
        const health = await fetch("http://localhost:5000/health", { signal: AbortSignal.timeout(3000) });
        if (!health.ok) throw new Error("unhealthy");
      } catch {
        toast.error(
          "Python backend is not running. Open a terminal in the project folder and run: python app.py",
          { duration: 8000 }
        );
        return; // abort before touching any images
      }
    }

    const controller = new AbortController();
    abortRef.current = controller;
    setIsProcessing(true);
    setProcessedCount(0);
    setTotalToProcess(pendingIds.length);
    let backendDown = false;
    for (let i = 0; i < pendingIds.length; i++) {
      if (controller.signal.aborted || backendDown) break;
      try {
        await runDetection(pendingIds[i], controller.signal);
      } catch (e: any) {
        if (e?.message === "BACKEND_DOWN") {
          backendDown = true;
          toast.error("Python backend stopped responding — detection paused. Restart python app.py then click Reset errors to retry.", { duration: 8000 });
          break;
        }
      }
      setProcessedCount(i + 1);
    }
    setIsProcessing(false);
    abortRef.current = null;
    if (!backendDown && !controller.signal.aborted) {
      // silent — stats panel shows the result
    }
  };

  const cancelDetection = () => {
    abortRef.current?.abort();
    setIsProcessing(false);
  };

  // Retry a single image — reset it to pending then run
  const retryImage = (id: string) => {
    setImages(prev => prev.map(img =>
      img.id === id ? { ...img, status: "pending" } : img
    ));
    // Small delay so state flushes before runDetection reads it
    setTimeout(() => runDetection(id), 0);
  };

  // ── Model upload ─────────────────────────────────────────────────────────────
  const handleModelUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return;
    const fd       = new FormData(); fd.append("model", file);
    const targetId = activeModelIdRef.current;
    const promise = (async () => {
      const res = await fetch("/api/upload-model", { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || "Upload failed");
      }
      const data = await res.json();
      const modelPath = data.model.path;

      // Try to fetch class names — only works when Python backend is running.
      // If it fails (Python not started yet), model still loads fine.
      let classNames: string[] | undefined;
      try {
        const classRes = await fetch("/api/model-classes", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_path: modelPath }),
        });
        if (classRes.ok) {
          const classData = await classRes.json();
          classNames = classData?.classes;
        }
      } catch {
        // Python backend not running — class names unavailable, that's OK
      }

      setModels(prev => prev.map(m =>
        m.id === targetId
          ? { ...m, status: "Ready", name: data.model.name, path: modelPath, classNames }
          : m
      ));
      scanLocalModels();
      if (modelInputRef.current) modelInputRef.current.value = "";
      return data;
    })();
    toast.promise(promise, {
      loading: "Uploading model…",
      success: d => `${d.model.name} loaded successfully`,
      error:   err => `Upload failed: ${err.message}`,
    });
  };

  // ── Annotations ──────────────────────────────────────────────────────────────
  const toggleFalsePositive = (detId: string) => {
    if (!selectedImage) return;
    setImages(prev => prev.map(img =>
      img.id !== selectedImage.id ? img : {
        ...img,
        isAnnotated: true,
        detections: img.detections.map(d =>
          d.id !== detId ? d : { ...d, isFalsePositive: !d.isFalsePositive, color: d.isFalsePositive ? colorForLabel(d.label) : "#ef4444" }
        ),
      }
    ));
  };

  // FIX: annotations are always "Missed Target" — no confusing label picker
  const addAnnotation = (box: [number, number, number, number]) => {
    if (!selectedImage) return;
    const ann: Annotation = {
      id:    crypto.randomUUID(),
      label: "Missed Target",
      bbox:  box,
      type:  "missed_target",
      color: "#eab308",
    };
    setImages(prev => prev.map(img => {
      if (img.id !== selectedImage.id) return img;
      setHistoryMap(h => ({ ...h, [img.id]: [...(h[img.id] ?? []), img.annotations] }));
      return { ...img, annotations: [...img.annotations, ann], isAnnotated: true };
    }));
  };

  const deleteAnnotation = (annId: string) => {
    if (!selectedImage) return;
    setImages(prev => prev.map(img => {
      if (img.id !== selectedImage.id) return img;
      setHistoryMap(h => ({ ...h, [img.id]: [...(h[img.id] ?? []), img.annotations] }));
      return { ...img, annotations: img.annotations.filter(a => a.id !== annId) };
    }));
  };

  const undoAnnotation = () => {
    if (!selectedImage) return;
    const hist = historyMap[selectedImage.id] ?? [];
    if (!hist.length) return;
    setImages(prev => prev.map(img =>
      img.id === selectedImage.id ? { ...img, annotations: hist[hist.length - 1] } : img
    ));
    setHistoryMap(h => ({ ...h, [selectedImage.id]: hist.slice(0, -1) }));
  };

  const removeImage = (id: string) => {
    setImages(prev => {
      if (selectedImageId === id) {
        const idx  = prev.findIndex(img => img.id === id);
        const next = prev[idx + 1] ?? prev[idx - 1];
        setSelectedImageId(next?.id ?? null);
      }
      return prev.filter(img => img.id !== id);
    });
  };

  // ── Mouse drawing & pan ──────────────────────────────────────────────────────
  // Controls:
  //   Left drag          → draw annotation box
  //   Space + left drag  → pan the canvas
  //   Ctrl + scroll      → zoom in/out centred on cursor
  //   Zoom buttons       → in the toolbar (not floating over canvas)

  const spaceHeldRef = useRef(false);

  // Track space key globally so it works even when canvas isn't focused
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && e.target === document.body) {
        spaceHeldRef.current = true;
        e.preventDefault(); // prevent page scroll
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        spaceHeldRef.current = false;
        // If we were panning, stop
        setIsPanning(false);
      }
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup",   up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (activeTab !== "annotate" || annotateMode !== "images") return;
    e.preventDefault();

    if (spaceHeldRef.current || e.button === 1) {
      // Pan mode: space+drag or middle-mouse drag
      setIsPanning(true);
      setLastMousePos({ x: e.clientX, y: e.clientY });
      return;
    }

    if (e.button !== 0) return; // ignore right-click etc

    // Draw mode: convert client coords → SVG 0-1000 space
    // Must account for current zoom & pan so the box lands correctly
    const rect = imageContainerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = ((e.clientX - rect.left) / rect.width)  * 1000;
    const y = ((e.clientY - rect.top)  / rect.height) * 1000;
    setIsDrawing(true);
    setStartPos({ x, y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning) {
      const dx = e.clientX - lastMousePos.x;
      const dy = e.clientY - lastMousePos.y;
      setPanOffset(p => ({ x: p.x + dx, y: p.y + dy }));
      setLastMousePos({ x: e.clientX, y: e.clientY });
      return;
    }
    if (!isDrawing || !startPos) return;
    const rect = imageContainerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = Math.max(0, Math.min(1000, ((e.clientX - rect.left) / rect.width)  * 1000));
    const y = Math.max(0, Math.min(1000, ((e.clientY - rect.top)  / rect.height) * 1000));
    setCurrentBox([
      Math.min(startPos.x, x), Math.min(startPos.y, y),
      Math.max(startPos.x, x), Math.max(startPos.y, y),
    ]);
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    if (isPanning) { setIsPanning(false); return; }
    if (isDrawing && currentBox) {
      const w = currentBox[2] - currentBox[0];
      const h = currentBox[3] - currentBox[1];
      if (w > 5 && h > 5) addAnnotation(currentBox);
    }
    setIsDrawing(false);
    setStartPos(null);
    setCurrentBox(null);
  };

  // Global mouseup so releasing outside the canvas still clears draw state
  useEffect(() => {
    const up = () => {
      if (isDrawing) { setIsDrawing(false); setStartPos(null); setCurrentBox(null); }
      if (isPanning) setIsPanning(false);
    };
    window.addEventListener("mouseup", up);
    return () => window.removeEventListener("mouseup", up);
  }, [isDrawing, isPanning]);

  // ── Touch / Apple Pencil support ─────────────────────────────────────────────
  // Converts a touch/pointer event coordinate to SVG 0-1000 space
  const clientToSVG = (clientX: number, clientY: number) => {
    const rect = imageContainerRef.current?.getBoundingClientRect();
    if (!rect) return null;
    return {
      x: ((clientX - rect.left) / rect.width)  * 1000,
      y: ((clientY - rect.top)  / rect.height) * 1000,
    };
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    if (activeTab !== "annotate" || annotateMode !== "images") return;
    if (e.touches.length === 2) {
      // Two-finger touch = pan
      setIsPanning(true);
      const cx = (e.touches[0].clientX + e.touches[1].clientX) / 2;
      const cy = (e.touches[0].clientY + e.touches[1].clientY) / 2;
      setLastMousePos({ x: cx, y: cy });
      return;
    }
    e.preventDefault();
    const touch = e.touches[0];
    const pos = clientToSVG(touch.clientX, touch.clientY);
    if (!pos) return;
    setIsDrawing(true);
    setStartPos(pos);
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (e.touches.length === 2 && isPanning) {
      const cx = (e.touches[0].clientX + e.touches[1].clientX) / 2;
      const cy = (e.touches[0].clientY + e.touches[1].clientY) / 2;
      const dx = cx - lastMousePos.x;
      const dy = cy - lastMousePos.y;
      setPanOffset(p => ({ x: p.x + dx, y: p.y + dy }));
      setLastMousePos({ x: cx, y: cy });
      return;
    }
    if (!isDrawing || !startPos || e.touches.length !== 1) return;
    e.preventDefault();
    const touch = e.touches[0];
    const pos = clientToSVG(touch.clientX, touch.clientY);
    if (!pos) return;
    setCurrentBox([
      Math.min(startPos.x, Math.max(0, Math.min(1000, pos.x))),
      Math.min(startPos.y, Math.max(0, Math.min(1000, pos.y))),
      Math.max(startPos.x, Math.max(0, Math.min(1000, pos.x))),
      Math.max(startPos.y, Math.max(0, Math.min(1000, pos.y))),
    ]);
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    if (isPanning) { setIsPanning(false); return; }
    if (isDrawing && currentBox) {
      const w = currentBox[2] - currentBox[0];
      const h = currentBox[3] - currentBox[1];
      if (w > 5 && h > 5) addAnnotation(currentBox);
    }
    setIsDrawing(false);
    setStartPos(null);
    setCurrentBox(null);
  };

  // Attach wheel listener as non-passive so preventDefault() actually works.
  // React's synthetic onWheel is passive in modern browsers and cannot prevent scroll.
  useEffect(() => {
    const el = canvasWrapperRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (activeTab !== "annotate" || annotateMode !== "images") return;
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.12 : 0.90;
      setZoom(prev => {
        const next = Math.min(Math.max(0.25, prev * factor), 10);
        const outerRect = el.getBoundingClientRect();
        const ox = e.clientX - outerRect.left - outerRect.width  / 2;
        const oy = e.clientY - outerRect.top  - outerRect.height / 2;
        setPanOffset(p => ({
          x: ox - (ox - p.x) * (next / prev),
          y: oy - (oy - p.y) * (next / prev),
        }));
        return next;
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [activeTab, annotateMode]);

  // Keep the React handler as a no-op (onWheel prop still needed to suppress passive warning)
  const handleWheel = () => {};

  // ── Video ─────────────────────────────────────────────────────────────────────
  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(URL.createObjectURL(file));
    setVideoCurrentTime(0); setVideoDuration(0);
    toast.success("Video loaded");
  };
  const captureFrame = () => {
    const v = videoRef.current; if (!v) return;
    const c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext("2d")?.drawImage(v, 0, 0);
    setCaptures(prev => [...prev, { id: crypto.randomUUID(), timestamp: v.currentTime, thumbnailUrl: c.toDataURL("image/jpeg", 0.5), fullImageUrl: c.toDataURL("image/jpeg", 1.0), annotations: [] }]);
    toast.success("Frame captured");
  };

  // ── Export ────────────────────────────────────────────────────────────────────
  // Auto-download a backup JSON every 5 minutes when annotations exist
  // Auto-backup removed — use SAVE button in header

  // Helper: resolve any image URL (data URL or local path ref) to base64 string
  const resolveImageBase64 = async (url: string): Promise<string> => {
    if (url.startsWith("data:")) {
      // Already a data URL — strip the prefix
      return url.includes(",") ? url.split(",")[1] : url;
    }
    if (url.startsWith("/api/image?")) {
      // Local path reference — use Node's read-file endpoint (no Python needed)
      const params = new URLSearchParams(url.replace("/api/image?", ""));
      const filePath = params.get("path") ?? "";
      const resp = await fetch(`/api/read-file?path=${encodeURIComponent(filePath)}`);
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: "Read failed" }));
        throw new Error(err.error ?? "Could not read image from disk");
      }
      const d = await resp.json();
      return d.base64;
    }
    // Any other URL — fetch and convert
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch image: ${url}`);
    const blob = await resp.blob();
    return new Promise<string>((resolve, reject) => {
      const r = new FileReader();
      r.onloadend = () => {
        const result = r.result as string;
        resolve(result.includes(",") ? result.split(",")[1] : result);
      };
      r.onerror = reject;
      r.readAsDataURL(blob);
    });
  };

  const exportAndTrain = async () => {
    const exportable = images.filter(img =>
      exportMode === "annotated_only"
        ? (img.annotations.length > 0 || img.detections.length > 0)
        : (img.status === "completed" || img.annotations.length > 0)
    );
    if (!exportable.length) {
      toast.error("No annotated or processed images to export.");
      return;
    }

    const toastId = toast.loading(`Preparing dataset…`);

    try {
      const JSZipMod = (await import("jszip")).default;
      const zip = new JSZipMod();
      const imgF = zip.folder("images"), lblF = zip.folder("labels");

      // CRITICAL: class names must match the BASE MODEL being fine-tuned from.
      // We always use class index 0 since we're fine-tuning a single-class model.
      // Get class name from loaded model, default to "target" as a safe fallback.
      const readyModel = models.find(m => m.status === "Ready");
      const modelClassName = readyModel?.classNames?.[0] ?? "target";
      const classNames: string[] = [modelClassName];

      let processed = 0;
      for (const img of exportable) {
        processed++;
        if (processed % 5 === 0) toast.loading(`Preparing images… ${processed}/${exportable.length}`, { id: toastId });
        let b64: string;
        try {
          b64 = await resolveImageBase64(img.url);
        } catch (fetchErr: any) {
          toast.error(`Could not read image ${img.name}: ${fetchErr?.message ?? "fetch failed"}. Is Python running?`, { id: toastId });
          return;
        }
        const imgFileName = img.name.split("/").pop() ?? img.name;
        imgF?.file(imgFileName, b64, { base64: true });

        // Valid detections + manual annotations = positive examples (class 0)
        // False positive images get empty label file = negative example
        const validBoxes = [
          ...img.detections.filter(d => !d.isFalsePositive),
          ...img.annotations,
        ];
        const labelLines = validBoxes.map(b => {
          const [x1, y1, x2, y2] = b.bbox.map(v => Math.max(0, Math.min(1000, v)));
          const cx = ((x1+x2)/2)/1000, cy = ((y1+y2)/2)/1000;
          const w = (x2-x1)/1000,      h  = (y2-y1)/1000;
          if (w <= 0 || h <= 0) return null;
          // Always class 0 — we're fine-tuning a single-class model
          return `0 ${cx.toFixed(6)} ${cy.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}`;
        }).filter(Boolean) as string[];

        const imgBaseName = imgFileName.replace(/\.[^.]+$/, "");
        lblF?.file(imgBaseName + ".txt", labelLines.join("\n"));
      }

      // data.yaml: single class, absolute paths set by Python path fixer
      const yaml = [
        `train: images`,
        `val: images`,
        `nc: 1`,
        `names: ['${modelClassName}']`,
      ].join("\n");
      zip.file("data.yaml", yaml);
      zip.file("classes.txt", modelClassName);

      toast.loading("Uploading dataset to training engine…", { id: toastId });

      const blob = await zip.generateAsync({ type: "blob" });
      const fd = new FormData();
      fd.append("file", blob, `reef_train_${new Date().toISOString().split("T")[0]}.zip`);

      const r = await fetch("/api/train/upload-dataset", { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) { toast.error(d.error ?? "Upload failed", { id: toastId }); return; }

      toast.success(`Dataset ready — ${exportable.length} images uploaded`, { id: toastId });
      setTrainDatasetPath(d.dataset_path);
      fetchTrainDatasets();
      setActiveTab("train");
    } catch (err: any) {
      toast.error(err?.message ?? "Export failed", { id: toastId });
    }
  };

  const exportToRoboflow = async () => {
    const zip = new JSZip();
    const imgF = zip.folder("images"), lblF = zip.folder("labels");
    

    // Build class name list from the first ready model, falling back to defaults
    const readyModel = models.find(m => m.status === "Ready");
    const classNames: string[] = readyModel?.classNames ?? ["cots", "clam", "bleaching"];

    // Include: any image that has been processed OR has manual annotations.
    // Roboflow requires a paired .txt for every image — we always write one,
    // even if it is empty (which is valid YOLO format for a negative sample).
    const exportable = images.filter(img =>
      img.status === "completed" || img.annotations.length > 0
    );

    if (!exportable.length) {
      toast.error("Nothing to export — run detection or add annotations first.");
      return;
    }

    for (const img of exportable) {
      // Extract raw base64 from data URL
      const b64 = await resolveImageBase64(img.url);
      imgF?.file(img.name, b64, { base64: true });

      // Combine valid detections + manual annotations into YOLO label lines
      const boxes = [
        ...img.detections.filter(d => !d.isFalsePositive),
        ...img.annotations,
      ];

      const labelLines = boxes.map(b => {
        const [x1, y1, x2, y2] = b.bbox.map(v => Math.max(0, Math.min(1000, v)));
        const cx = ((x1 + x2) / 2) / 1000;
        const cy = ((y1 + y2) / 2) / 1000;
        const w  = (x2 - x1) / 1000;
        const h  = (y2 - y1) / 1000;
        // Skip degenerate boxes
        if (w <= 0 || h <= 0) return null;
        const lbl = b.label.toLowerCase();
        let cls = classNames.findIndex(c => lbl.includes(c.toLowerCase()));
        if (cls === -1) cls = 0;
        return `${cls} ${cx.toFixed(6)} ${cy.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}`;
      }).filter(Boolean) as string[];

      // Always write the label file — empty = negative sample, which is valid
      const baseName = img.name.replace(/\.[^.]+$/, "");
      lblF?.file(`${baseName}.txt`, labelLines.join("\n"));
    }

    // data.yaml so Roboflow / ultralytics knows the class names
    const yaml = [
      `nc: ${classNames.length}`,
      `names: [${classNames.map(c => "'" + c + "'").join(", ")}]`,
    ].join("\n");
    zip.file("data.yaml", yaml);

    // classes.txt for compatibility
    zip.file("classes.txt", classNames.join("\n"));

    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    Object.assign(document.createElement("a"), {
      href: url,
      download: `reef_export_${new Date().toISOString().split("T")[0]}.zip`,
    }).click();
    URL.revokeObjectURL(url);
    toast.success(`Exported ${exportable.length} images with paired label files.`);
  };

  const exportCaptures = async () => {
    if (!captures.length) return;
    const zip = new JSZip();
    captures.forEach((cap, i) => zip.file(`capture_${cap.timestamp.toFixed(2)}s_${i}.jpg`, cap.fullImageUrl.split(",")[1], { base64: true }));
    const url = URL.createObjectURL(await zip.generateAsync({ type: "blob" }));
    Object.assign(document.createElement("a"), { href: url, download: `captures_${new Date().toISOString().split("T")[0]}.zip` }).click();
    URL.revokeObjectURL(url);
    toast.success("Captures exported");
  };

  const formatTime = (t: number) =>
    `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

  const tabs = [
    { id: "setup",    label: "SETUP"       },
    { id: "models",   label: "1. MODELS"   },
    { id: "upload",   label: "2. UPLOAD"   },
    { id: "detect",   label: "3. DETECT"   },
    { id: "annotate", label: "4. ANNOTATE" },
    { id: "train",    label: "5. TRAIN"    },
    { id: "export",   label: "EXPORT"      },
    { id: "scrape",   label: "🔍 SCRAPE"    },
    { id: "history",  label: "📊 HISTORY"   },
    { id: "remote",   label: "REMOTE ACCESS" },
  ];

  const anyModelReady = models.some(m => m.status === "Ready");

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-screen w-full bg-[var(--bg-main)] text-[var(--text-main)] font-sans overflow-hidden">
      <Toaster position="top-right" theme="dark" duration={2500} visibleToasts={2} />

      {/* Processing bar */}
      {isProcessing && (
        <div className="fixed top-0 left-0 w-full h-1 z-[100] bg-black/20">
          <div className="h-full bg-[var(--accent)] transition-all duration-300"
            style={{ width: `${(processedCount / totalToProcess) * 100}%` }} />
        </div>
      )}

      {/* Header */}
      <header className="h-[64px] bg-[var(--bg-header)] border-b border-[var(--border)] flex items-center justify-between px-6 flex-shrink-0 z-20">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-[var(--accent)] rounded flex items-center justify-center flex-shrink-0">
            <Waves className="w-5 h-5 text-black" />
          </div>
          <div className="hidden sm:block">
            <h1 className="font-bold text-lg tracking-tight text-white">Reef AI Detection Suite</h1>
            <div className="text-[9px] font-mono text-[var(--text-dim)] tracking-widest">v4.43</div>
          </div>
          <div className="block sm:hidden">
            <h1 className="font-bold text-sm tracking-tight text-white">Reef AI</h1>
            <div className="text-[8px] font-mono text-[var(--text-dim)]">v4.24</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {/* Save / Load — always accessible from every tab */}
          <Button variant="ghost" size="sm"
            className="text-[10px] font-mono text-[var(--accent)] hover:text-white hover:bg-[var(--accent)]/10 gap-1.5"
            onClick={saveSession}
            title="Save session to file">
            <Save className="w-3 h-3" /><span className="hidden sm:inline">SAVE</span>
          </Button>
          <Button variant="ghost" size="sm"
            className="text-[10px] font-mono text-[var(--text-dim)] hover:text-white hover:bg-white/5 gap-1.5"
            onClick={() => sessionInputRef.current?.click()}
            title="Load session from file">
            <DownloadCloud className="w-3 h-3" /><span className="hidden sm:inline">LOAD</span>
          </Button>
          <div className="w-px h-4 bg-[var(--border)]" />
          <Button variant="ghost" size="sm" className="text-[10px] font-mono text-red-500 hover:text-red-400 hover:bg-red-500/10 gap-2"
            onClick={() => { if (confirm("Clear all data and reload?")) { localStorage.clear(); window.location.reload(); } }}>
            <Trash2 className="w-3 h-3" /><span className="hidden sm:inline"> RESET</span>
          </Button>
          {updateAvailable && (
            <button
              className="bg-yellow-500/20 border border-yellow-500/50 rounded-full px-3 py-1 flex items-center gap-1.5 text-[10px] font-bold text-yellow-400 hover:bg-yellow-500/30 transition-colors"
              onClick={async () => {
                if (!confirm(`Update to v${updateAvailable}? The app will update automatically and you will need to restart.`)) return;
                // Fetch latest release zip URL from GitHub
                const toastId = toast.loading("Checking latest release...");
                try {
                  const rel = await fetch("https://api.github.com/repos/ResilientReefsFoundation/aidetect/releases/latest");
                  const relData = await rel.json();
                  const zipAsset = relData.assets?.find((a: any) => a.name.endsWith(".zip"));
                  const zipUrl = zipAsset?.browser_download_url ?? null;
                  if (!zipUrl) {
                    toast.error("No release zip found — ask the developer to publish a release.", { id: toastId });
                    return;
                  }
                  toast.loading("Downloading update...", { id: toastId });
                  const r = await fetch("/api/update", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ zipUrl }),
                  });
                  const reader = r.body?.getReader();
                  const decoder = new TextDecoder();
                  if (!reader) return;
                  let buf = "";
                  while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buf += decoder.decode(value, { stream: true });
                    const lines = buf.split("\n");
                    buf = lines.pop() ?? "";
                    for (const line of lines) {
                      if (!line.startsWith("data:")) continue;
                      try {
                        const msg = JSON.parse(line.slice(5).trim());
                        if (msg.stage === "error") toast.error(msg.message, { id: toastId });
                        else if (msg.stage === "done") {
                          toast.success("Update applied! Please close and reopen the app.", { id: toastId, duration: 10000 });
                        } else toast.loading(msg.message, { id: toastId });
                      } catch {}
                    }
                  }
                } catch (err: any) {
                  toast.error(err?.message ?? "Update failed", { id: toastId });
                }
              }}>
              ↑ v{updateAvailable} available — click to update
            </button>
          )}
          <div className="bg-black/40 border border-[var(--border)] rounded-full px-4 py-1 flex items-center gap-2">
            <div className={cn("w-2 h-2 rounded-full animate-pulse", anyModelReady ? "bg-emerald-500" : "bg-red-500")} />
            <span className="text-[10px] font-mono text-[var(--text-dim)]">
              {anyModelReady ? "Model Ready" : "No model loaded"}
            </span>
          </div>
        </div>
      </header>

      {/* Nav tabs */}
      <nav className="h-[48px] bg-[var(--bg-header)] border-b border-[var(--border)] flex items-center px-2 md:px-6 gap-4 md:gap-8 flex-shrink-0 overflow-x-auto no-scrollbar">
        {tabs.map(tab => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={cn("flex items-center h-full px-2 text-[10px] md:text-[11px] font-bold tracking-widest transition-all border-b-2 whitespace-nowrap flex-shrink-0",
              activeTab === tab.id ? "text-[var(--accent)] border-[var(--accent)]" : "text-[var(--text-dim)] border-transparent hover:text-white")}>
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="flex-1 flex overflow-hidden">

        {/* ══ MODELS ══════════════════════════════════════════════════════════ */}
        {activeTab === "models" && (
          <div className="flex-1 p-8 overflow-y-auto no-scrollbar space-y-10">

            {/* Model slots */}
            <div>
              <h2 className="text-2xl font-bold text-white uppercase tracking-widest mb-6">Your Models</h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
                {models.map(m => (
                  <div key={m.id} className={cn("reef-card space-y-4 transition-all",
                    activeModelId === m.id ? "border-[var(--accent)]" : "")}>
                    <div className="flex items-center gap-3">
                      <Brain className="w-5 h-5 text-pink-500" />
                      <div>
                        <div className="text-sm font-bold text-white uppercase tracking-wider">{m.name}</div>
                        <div className="text-[10px] text-[var(--text-dim)]">{m.type}</div>
                      </div>
                    </div>
                    {/* Status badge */}
                    <div className={cn("text-[11px] font-bold uppercase tracking-widest px-3 py-1 rounded-full w-fit",
                      m.status === "Ready" ? "bg-emerald-500/10 text-emerald-500 border border-emerald-500/30" : "bg-red-500/10 text-red-500 border border-red-500/30")}>
                      {m.status}
                    </div>
                    {m.path && (
                      <div className="text-[9px] text-[var(--text-dim)] font-mono truncate" title={m.path}>{m.path.split(/[\\/]/).pop()}</div>
                    )}
                    {/* Option A — browse for a file */}
                    <button
                      className={cn("w-full border-2 border-dashed rounded-lg py-3 text-center transition-all text-[11px] font-bold uppercase tracking-wider",
                        activeModelId === m.id ? "border-[var(--accent)] text-[var(--accent)]" : "border-[var(--border)] text-[var(--text-dim)] hover:border-[var(--accent)] hover:text-[var(--accent)]")}
                      onClick={() => { setActiveModelId(m.id); activeModelIdRef.current = m.id; modelInputRef.current?.click(); }}>
                      {m.status === "Ready" ? "↑ Replace — browse file" : "↑ Browse for .pt / .onnx"}
                    </button>

                    {/* Option B — filtered by slot prefix, with rename/delete */}
                    <div className="space-y-2">
                      <div className="relative">
                        <select className="w-full bg-[#1a1a2e] border border-[var(--border)] rounded-lg px-3 py-2.5 text-[11px] font-bold uppercase tracking-wider text-white appearance-none cursor-pointer hover:border-[var(--accent)] transition-all"
                          value=""
                          onChange={e => {
                            const path = e.target.value; if (!path) return;
                            setActiveModelId(m.id); activeModelIdRef.current = m.id;
                            loadSavedModel(path, m.id);
                          }}>
                          <option value="">{savedModels.length === 0 ? "No saved models found" : "↓ Pick from models/ folder"}</option>
                          {savedModels.filter(sm => sm.name.toLowerCase().startsWith(m.id)).length > 0 && (
                            <optgroup label={"── " + m.id.toUpperCase() + " models ──"}>
                              {savedModels.filter(sm => sm.name.toLowerCase().startsWith(m.id)).map(sm => (
                                <option key={sm.path} value={sm.path}>{sm.name}</option>
                              ))}
                            </optgroup>
                          )}
                          {savedModels.filter(sm => !sm.name.toLowerCase().startsWith(m.id)).length > 0 && (
                            <optgroup label="── Other models ──">
                              {savedModels.filter(sm => !sm.name.toLowerCase().startsWith(m.id)).map(sm => (
                                <option key={sm.path} value={sm.path}>{sm.name}</option>
                              ))}
                            </optgroup>
                          )}
                        </select>
                        <div className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-[var(--text-dim)]">▾</div>
                      </div>
                      {savedModels.length > 0 && (
                        renamingModel ? (
                          <div className="flex gap-1">
                            <input autoFocus type="text" value={renamingModel.value}
                              onChange={e => setRenamingModel({ ...renamingModel, value: e.target.value })}
                              onKeyDown={e => { if (e.key === "Enter") renameModel(renamingModel.name, renamingModel.value); if (e.key === "Escape") setRenamingModel(null); }}
                              className="flex-1 bg-black/40 border border-[var(--accent)] rounded px-2 py-1 text-[10px] text-white font-mono focus:outline-none" />
                            <Button size="sm" className="reef-button-primary h-6 px-2 text-[9px]" onClick={() => renameModel(renamingModel.name, renamingModel.value)}>Save</Button>
                            <Button size="sm" variant="outline" className="h-6 px-2 text-[9px] border-[var(--border)] text-[var(--text-dim)]" onClick={() => setRenamingModel(null)}>✕</Button>
                          </div>
                        ) : (
                          <div className="grid grid-cols-2 gap-1 w-full">
                            <select className="w-full bg-black/40 border border-[var(--border)] rounded px-2 py-1 text-[9px] text-[var(--text-dim)] truncate"
                              defaultValue=""
                              onChange={e => { if (e.target.value) setRenamingModel({ name: e.target.value, value: e.target.value.replace(/\.pt$/, "").replace(/\.onnx$/, "") }); e.target.value = ""; }}>
                              <option value="">✏ Rename…</option>
                              {savedModels.map(sm => <option key={sm.name} value={sm.name}>{sm.name}</option>)}
                            </select>
                            <select className="w-full bg-black/40 border border-red-500/30 rounded px-2 py-1 text-[9px] text-red-400 truncate"
                              defaultValue=""
                              onChange={e => { if (e.target.value) deleteModel(e.target.value); e.target.value = ""; }}>
                              <option value="">🗑 Delete…</option>
                              {savedModels.map(sm => <option key={sm.name} value={sm.name}>{sm.name}</option>)}
                            </select>
                          </div>
                        )
                      )}
                    </div>

                    {/* Option C — paste a URL */}
                    {urlInputs[m.id]?.visible ? (
                      <div className="space-y-2">
                        <input
                          autoFocus
                          type="url"
                          placeholder="https://example.com/model.pt"
                          value={urlInputs[m.id]?.value ?? ""}
                          onChange={e => setUrlInputs(prev => ({ ...prev, [m.id]: { visible: true, value: e.target.value } }))}
                          onKeyDown={e => {
                            if (e.key === "Enter") loadModelFromUrl(urlInputs[m.id]?.value ?? "", m.id);
                            if (e.key === "Escape") setUrlInputs(prev => ({ ...prev, [m.id]: { visible: false, value: "" } }));
                          }}
                          className="w-full bg-black/40 border border-[var(--accent)]/50 rounded-lg px-3 py-2 text-[11px] text-white placeholder-[var(--text-dim)] focus:border-[var(--accent)] outline-none"
                        />
                        <div className="flex gap-2">
                          <Button size="sm" className="flex-1 reef-button-primary h-8 text-[10px]"
                            onClick={() => loadModelFromUrl(urlInputs[m.id]?.value ?? "", m.id)}>
                            Download
                          </Button>
                          <Button size="sm" variant="outline" className="h-8 text-[10px] border-[var(--border)] text-[var(--text-dim)]"
                            onClick={() => setUrlInputs(prev => ({ ...prev, [m.id]: { visible: false, value: "" } }))}>
                            Cancel
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <button
                        className="w-full border border-[var(--border)] rounded-lg py-2 text-center transition-all text-[11px] font-bold uppercase tracking-wider text-[var(--text-dim)] hover:border-[var(--accent)] hover:text-[var(--accent)]"
                        onClick={() => setUrlInputs(prev => ({ ...prev, [m.id]: { visible: true, value: "" } }))}>
                        🔗 Load from URL
                      </button>
                    )}
                  </div>
                ))}
              </div>
              <input type="file" ref={modelInputRef} onChange={handleModelUpload} accept=".pt,.onnx" className="hidden" />
            </div>

          </div>
        )}

        {/* ══ SETUP ═══════════════════════════════════════════════════════════ */}
        {activeTab === "setup" && (
          <div className="flex-1 p-8 overflow-y-auto no-scrollbar">
            <div className="max-w-3xl space-y-8">

              {/* Setup & Startup */}
              <div className="space-y-5">
                <div className="flex items-center gap-3">
                  <Cpu className="w-7 h-7 text-pink-500" />
                  <div>
                    <h2 className="text-xl font-bold text-white uppercase tracking-widest">Setup &amp; Startup</h2>
                    <p className="text-[11px] text-[var(--text-dim)] mt-0.5">New machine? Do steps 1–3 once. After that, only step 4 every session.</p>
                  </div>
                </div>
                <div className="reef-card space-y-4">
                  {[
                    { s: 1, t: "Copy the project folder to this machine.", sub: "USB drive, network share, or download — put it anywhere you like." },
                    { s: 2, t: "Install Python 3.10+ and Node.js LTS.", sub: "Include the installers on the thumb drive for offline machines." },
                    { s: 3, t: "Double-click setup_dependencies.bat — wait for it to finish.", sub: "Takes a few minutes. Only needed once per machine.", dl: "setup" },
                    { s: 4, t: "Double-click run_reef_ai.bat to start the app.", sub: "Opens the browser automatically. Do this every session.", dl: "launcher", highlight: true },
                  ].map(i => (
                    <div key={i.s} className={cn("flex gap-4 items-start p-3 rounded-lg transition-all", i.highlight ? "bg-emerald-500/5 border border-emerald-500/20" : "")}>
                      <div className={cn("w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 text-[12px] font-bold mt-0.5", i.highlight ? "bg-emerald-500/20 text-emerald-400" : "bg-pink-500/20 text-pink-500")}>{i.s}</div>
                      <div className="flex-1">
                        <p className={cn("text-[13px]", i.highlight ? "text-white font-semibold" : "")}>{i.t}</p>
                        {i.sub && <p className="text-[11px] text-[var(--text-dim)] mt-0.5">{i.sub}</p>}
                      </div>
                      {i.dl === "launcher" && (
                        <Button variant="outline" size="sm" className="border-emerald-500/50 text-emerald-500 hover:bg-emerald-500/10 gap-1.5 text-[10px] flex-shrink-0"
                          onClick={() => Object.assign(document.createElement("a"), { href: "/run_reef_ai.bat", download: "run_reef_ai.bat" }).click()}>
                          <DownloadCloud className="w-3 h-3" />Download
                        </Button>
                      )}
                      {i.dl === "setup" && (
                        <Button variant="outline" size="sm" className="border-blue-500/50 text-blue-400 hover:bg-blue-500/10 gap-1.5 text-[10px] flex-shrink-0"
                          onClick={() => Object.assign(document.createElement("a"), { href: "/setup_dependencies.bat", download: "setup_dependencies.bat" }).click()}>
                          <DownloadCloud className="w-3 h-3" />Download
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Cloud AI Settings */}
              <div className="space-y-4 pt-4 border-t border-[var(--border)]">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Settings className="w-6 h-6 text-[var(--accent)]" />
                    <h2 className="text-xl font-bold text-white uppercase tracking-widest">Cloud AI Settings</h2>
                  </div>
                  <Button variant="outline" size="sm" className="border-[var(--border)] text-[var(--text-dim)] hover:text-white gap-2"
                    onClick={() => { scanLocalModels(); }}>
                    <Shell className="w-3 h-3" />Rescan models/
                  </Button>
                </div>
                <div className="reef-card space-y-3">
                  <label className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest block">Gemini API Key (for Cloud mode)</label>
                  <div className="flex gap-2">
                    <input type="password" value={geminiKey} onChange={e => setGeminiKey(e.target.value)} placeholder="Paste key here…"
                      className="flex-1 bg-black/40 border border-[var(--border)] rounded px-4 py-2 text-sm text-white focus:border-[var(--accent)] outline-none" />
                    <Button variant="outline" className="border-[var(--border)] text-[var(--accent)]" onClick={() => { setGeminiKey(""); }}>Clear</Button>
                  </div>
                  <p className="text-[10px] text-[var(--text-dim)] italic">Only needed for Gemini Cloud mode. Stored in browser local storage.</p>
                </div>
              </div>

              {/* Danger zone */}
              <div className="pt-4 border-t border-[var(--border)]">
                <div className="reef-card border-red-500/20 bg-red-500/5 space-y-3">
                  <h3 className="text-red-500 font-bold uppercase tracking-widest text-[10px]">DANGER ZONE</h3>
                  <p className="text-[11px] text-[var(--text-dim)]">Clears all images, detections, and annotations. Cannot be undone.</p>
                  <Button variant="outline" className="border-red-500/50 text-red-500 hover:bg-red-500/10 h-10 gap-2"
                    onClick={() => { setImages([]); toast.success("Session cleared"); }}>
                    <Trash2 className="w-4 h-4" />CLEAR SESSION DATA
                  </Button>
                </div>
              </div>

            </div>
          </div>
        )}

        {/* ══ UPLOAD ══════════════════════════════════════════════════════════ */}
        {activeTab === "upload" && (
          <div className="flex-1 p-8 overflow-y-auto no-scrollbar space-y-8">
            <div className="max-w-3xl space-y-8">
              <h2 className="text-2xl font-bold text-white uppercase tracking-widest">Upload Survey Images</h2>
              <div
                className={cn("border-2 border-dashed rounded-xl p-16 text-center transition-colors cursor-pointer bg-black/20",
                  isDraggingOver ? "border-[var(--accent)] bg-[var(--accent)]/5 scale-[1.01]" : "border-[var(--border)] hover:border-[var(--accent)]")}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={e => { e.preventDefault(); setIsDraggingOver(true); }}
                onDragLeave={() => setIsDraggingOver(false)}
                onDrop={handleDrop}>
                <div className="text-xl font-bold text-white mb-2">{images.length > 0 ? `${images.length} Images Loaded` : "Drop images, folders or a .zip here"}</div>
                <p className="text-[var(--text-dim)] max-w-sm mx-auto text-sm">{images.length > 0 ? "Click to add more, or drop more files." : "JPG, PNG, ZIP archives and folders supported."}</p>
                <input type="file" ref={fileInputRef} onChange={handleFileUpload} multiple accept="image/*,.zip" className="hidden" />
              </div>
              <div className="border-2 border-dashed border-[var(--border)] rounded-xl p-12 text-center hover:border-[var(--accent)] transition-colors cursor-pointer bg-black/20"
                onClick={() => videoInputRef.current?.click()}>
                <Video className="w-12 h-12 mx-auto mb-4 text-[var(--text-dim)]" />
                <div className="text-xl font-bold text-white mb-2">{videoUrl ? "Video Loaded" : "Drop a video file"}</div>
                <p className="text-[var(--text-dim)] text-sm">{videoUrl ? "Click to change." : "MP4, MOV, AVI supported."}</p>
                <input type="file" ref={videoInputRef} className="hidden" accept="video/*" onChange={handleVideoUpload} />
              </div>
              {/* Folder scan — for large local datasets */}
              <div className="reef-card space-y-3">
                <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest">Scan Local Folder</h3>
                <p className="text-[11px] text-[var(--text-dim)]">For large datasets (1000+ images) — images are served directly from disk, no memory crash.</p>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={folderPath}
                    onChange={e => setFolderPath(e.target.value)}
                    placeholder="e.g. D:\CotsImages	rain\images"
                    className="flex-1 bg-black/40 border border-[var(--border)] rounded px-3 py-2 text-[12px] text-white placeholder-[var(--text-dim)] focus:border-[var(--accent)] outline-none font-mono"
                  />
                  <Button size="sm" className="reef-button-primary gap-2 px-4 flex-shrink-0"
                    disabled={!folderPath.trim() || folderScanning}
                    onClick={async () => {
                      setFolderScanning(true);
                      const toastId = toast.loading("Scanning folder…");
                      try {
                        const r = await fetch("/api/scan-folder", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ folder: folderPath.trim(), recursive: folderRecursive }),
                        });
                        const d = await r.json();
                        if (!r.ok) { toast.error(d.error ?? "Scan failed", { id: toastId }); return; }
                        if (!d.images?.length) { toast.error("No images found in that folder", { id: toastId }); return; }
                        // Add images as local URL references — no base64, no memory spike
                        const newImgs: ImageData[] = d.images.map((img: any) => ({
                          id: crypto.randomUUID(),
                          name: img.name,
                          url: `/api/image?path=${encodeURIComponent(img.path)}`,
                          detections: [], annotations: [], status: "pending" as const,
                          localPath: img.path,
                        }));
                        setImages(prev => [...prev, ...newImgs]);
                        if (!selectedImageId && newImgs.length > 0) setSelectedImageId(newImgs[0].id);
                        toast.success(`Loaded ${d.total} images from folder`, { id: toastId });
                      } catch (err: any) {
                        toast.error(err?.message ?? "Scan failed", { id: toastId });
                      } finally {
                        setFolderScanning(false);
                      }
                    }}>
                    {folderScanning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Shell className="w-4 h-4" />}
                    Scan
                  </Button>
                </div>
                <div className="flex gap-3 text-[10px] text-[var(--text-dim)]">
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input type="checkbox" className="accent-[var(--accent)]"
                      checked={folderRecursive}
                      onChange={e => setFolderRecursive(e.target.checked)}
                    />
                    Include subfolders
                  </label>
                </div>
              </div>

              {/* YouTube frame extractor */}
              <div className="reef-card space-y-4">
                <div className="flex items-center gap-2">
                  <span className="text-lg">▶</span>
                  <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest">Extract Frames from YouTube</h3>
                </div>
                <p className="text-[11px] text-[var(--text-dim)]">Paste a YouTube URL — downloads the video and extracts frames as a zip to your Downloads folder.</p>
                <input
                  type="url"
                  value={ytUrl}
                  onChange={e => setYtUrl(e.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                  className="w-full bg-black/40 border border-[var(--border)] rounded px-3 py-2 text-sm text-white placeholder-[var(--text-dim)] focus:border-[var(--accent)] outline-none font-mono"
                />
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-wider block">Frame every N seconds</label>
                    <input type="number" min={1} max={60} value={ytInterval}
                      onChange={e => setYtInterval(Math.max(1, parseInt(e.target.value) || 1))}
                      className="w-full bg-black/40 border border-[var(--border)] rounded px-3 py-2 text-sm text-white text-center focus:border-[var(--accent)] outline-none" />
                    <p className="text-[9px] text-[var(--text-dim)]">Lower = more frames, more variety</p>
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-wider block">Max frames</label>
                    <input type="number" min={10} max={2000} value={ytMaxFrames}
                      onChange={e => setYtMaxFrames(Math.max(10, parseInt(e.target.value) || 10))}
                      className="w-full bg-black/40 border border-[var(--border)] rounded px-3 py-2 text-sm text-white text-center focus:border-[var(--accent)] outline-none" />
                    <p className="text-[9px] text-[var(--text-dim)]">Cap to avoid huge zips</p>
                  </div>
                </div>
                <Button className="reef-button-primary w-full gap-2"
                  disabled={!ytUrl.trim() || ytRunning}
                  onClick={async () => {
                    setYtRunning(true);
                    setYtProgress({ stage: "starting", message: "Connecting..." });
                    try {
                      const r = await fetch("/api/youtube-frames", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ url: ytUrl.trim(), interval: ytInterval, max_frames: ytMaxFrames, label: "youtube" }),
                      });
                      if (!r.ok) { const d = await r.json(); toast.error(d.error ?? "Failed"); setYtProgress(null); return; }
                      const reader = r.body?.getReader();
                      const decoder = new TextDecoder();
                      if (!reader) return;
                      let buf = "";
                      while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        buf += decoder.decode(value, { stream: true });
                        const lines = buf.split("\n");
                        buf = lines.pop() ?? "";
                        for (const line of lines) {
                          if (!line.startsWith("data:")) continue;
                          try {
                            const msg = JSON.parse(line.slice(5).trim());
                            if (msg.type === "progress") setYtProgress({ stage: msg.stage, message: msg.message, frames: msg.frames });
                            else if (msg.type === "done") {
                              setYtProgress({ stage: "done", message: "✓ " + msg.frames + " frames saved to Downloads as " + msg.zip_name });
                              toast.success(msg.frames + " frames extracted — zip in Downloads folder");
                            } else if (msg.type === "error") {
                              toast.error(msg.message);
                              setYtProgress({ stage: "error", message: msg.message });
                            }
                          } catch {}
                        }
                      }
                    } catch (err: any) {
                      toast.error(err?.message ?? "Failed");
                      setYtProgress(null);
                    } finally {
                      setYtRunning(false);
                    }
                  }}>
                  {ytRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <span>▶</span>}
                  {ytRunning ? "Extracting..." : "Extract Frames"}
                </Button>
                {ytProgress && (
                  <div className={cn("text-[11px] p-3 rounded border", ytProgress.stage === "error" ? "text-red-400 border-red-500/20 bg-red-500/5" : ytProgress.stage === "done" ? "text-emerald-400 border-emerald-500/20 bg-emerald-500/5" : "text-[var(--text-dim)] border-[var(--border)]")}>
                    {ytProgress.message}
                    {ytProgress.frames && ytProgress.stage !== "done" && <span className="ml-2 font-mono text-[var(--accent)]">({ytProgress.frames} frames)</span>}
                  </div>
                )}
                <p className="text-[10px] text-[var(--text-dim)]">Requires <span className="font-mono text-white">yt-dlp</span> — installed automatically by <span className="font-mono text-white">setup_dependencies.bat</span></p>
              </div>

              {images.length > 0 && (
                <div className="flex items-center justify-between p-4 reef-card bg-emerald-500/5 border-emerald-500/20">
                  <div>
                    <div className="text-sm font-bold text-white">Ready for Detection</div>
                    <div className="text-xs text-[var(--text-dim)] mt-1">Go to DETECT tab to run the AI.</div>
                  </div>
                  <Button className="reef-button-primary" onClick={() => setActiveTab("detect")}>Go to Detect →</Button>
                </div>
              )}

            </div>
          </div>
        )}

        {/* ══ DETECT ══════════════════════════════════════════════════════════ */}
        {activeTab === "detect" && (
          <div className="flex-1 p-8 overflow-y-auto no-scrollbar space-y-8">
            <div className="max-w-5xl space-y-8">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Target className="w-8 h-8 text-pink-500" />
                  <h2 className="text-2xl font-bold text-white uppercase tracking-widest">Run AI Detection</h2>
                </div>
                {/* Inference mode toggle */}
                <div className="flex bg-black/40 p-1 rounded border border-[var(--border)]">
                  {(["local", "cloud"] as const).map(m => (
                    <button key={m} onClick={() => setInferenceMode(m)}
                      className={cn("px-4 py-1.5 rounded text-[10px] font-bold uppercase tracking-widest transition-all",
                        inferenceMode === m ? "bg-[var(--accent)] text-black" : "text-[var(--text-dim)] hover:text-white")}>
                      {m === "local" ? "Local GPU" : "Gemini Cloud"}
                    </button>
                  ))}
                </div>
              </div>

              {/* Model status warning */}
              {inferenceMode === "local" && !anyModelReady && (
                <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400 space-y-1">
                  <p>⚠ No model loaded — go to the <button className="underline font-bold" onClick={() => setActiveTab("models")}>MODELS tab</button> and upload a .pt file first.</p>
                </div>
              )}


              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-6">
                  <div className="reef-card space-y-5">
                    <div className="flex items-center justify-between">
                      <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest">Confidence Threshold</h3>
                      <div className="text-[10px] font-mono text-[var(--accent)]">{(confidence * 100).toFixed(0)}%</div>
                    </div>
                    <div className="px-2 py-2">
                      <input
                        type="range"
                        min={0} max={100} step={1}
                        value={Math.round(confidence * 100)}
                        onChange={e => setConfidence(Number(e.target.value) / 100)}
                        className="w-full h-2 rounded-full appearance-none cursor-pointer"
                        style={{
                          background: `linear-gradient(to right, var(--accent) ${Math.round(confidence * 100)}%, var(--border) ${Math.round(confidence * 100)}%)`,
                          accentColor: "var(--accent)",
                        }}
                      />
                    </div>
                    {/* Show which model will be used */}
                    {inferenceMode === "local" && (
                      <div className="text-[11px] text-[var(--text-dim)] border-t border-[var(--border)] pt-3">
                        Using: <span className={cn("font-bold", anyModelReady ? "text-emerald-400" : "text-red-400")}>
                          {models.find(m => m.status === "Ready")?.name ?? "No model loaded"}
                        </span>
                      </div>
                    )}
                  </div>

                  <div className="reef-card space-y-4 p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-lg font-bold text-white">{images.length} Images</div>
                        <div className="text-xs text-[var(--text-dim)] mt-1">
                          {images.filter(i => i.status === "completed").length} processed ·{" "}
                          {images.filter(i => i.status === "pending").length} pending ·{" "}
                          {images.filter(i => i.status === "error").length} errors
                        </div>
                      </div>
                      {isProcessing ? (
                        <Button className="h-14 px-10 gap-3 text-lg border-2 border-red-500 text-red-500 bg-transparent hover:bg-red-500/10"
                          onClick={cancelDetection}>
                          <Loader2 className="w-6 h-6 animate-spin" />
                          <span className="font-mono text-sm">CANCEL ({processedCount}/{totalToProcess})</span>
                        </Button>
                      ) : (
                        <div className="flex flex-col gap-2 items-end">
                          <Button className="reef-button-primary h-14 px-10 gap-3 text-lg"
                            disabled={images.length === 0 || (inferenceMode === "local" && !anyModelReady)}
                            onClick={runAllDetections}>
                            <Play className="w-6 h-6" />RUN DETECTION
                          </Button>
                          {images.filter(i => i.status === "completed").length > 0 && (
                            <Button variant="outline" size="sm"
                              className="border-[var(--border)] text-[var(--text-dim)] hover:text-white gap-2 text-[10px]"
                              disabled={inferenceMode === "local" && !anyModelReady}
                              onClick={() => {
                                if (!confirm(`Reset all ${images.filter(i => i.status === "completed").length} processed images and clear their detections? This cannot be undone.`)) return;
                                setImages(prev => prev.map(img => ({ ...img, status: "pending", detections: [] })));
                                toast.success("All images reset — ready to re-run detection with new model");
                              }}>
                              <Undo className="w-3 h-3" />Re-run with new model
                            </Button>
                          )}
                        </div>
                      )}
                    </div>
                    {/* Retry all errors */}
                    {images.filter(i => i.status === "error").length > 0 && !isProcessing && (
                      <div className="border-t border-[var(--border)] pt-3 flex items-center justify-between">
                        <span className="text-[11px] text-red-400">
                          {images.filter(i => i.status === "error").length} images failed last run
                        </span>
                        <Button variant="outline" size="sm"
                          className="border-red-500/50 text-red-400 hover:bg-red-500/10 gap-2 text-[10px]"
                          disabled={inferenceMode === "local" && !anyModelReady}
                          onClick={() => {
                            setImages(prev => prev.map(img =>
                              img.status === "error" ? { ...img, status: "pending" } : img
                            ));
                            // notification removed — filter panel shows the count
                          }}>
                          <XCircle className="w-3 h-3" />Reset errors to pending
                        </Button>
                      </div>
                    )}
                  </div>
                </div>

                {/* Stats */}
                <div className="reef-card space-y-4">
                  <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest">STATS</h3>
                  {[
                    { label: "TOTAL",      value: images.length,                                         color: "text-white"         },
                    { label: "PROCESSED",  value: images.filter(i => i.status === "completed").length,   color: "text-emerald-500"   },
                    { label: "DETECTIONS", value: images.reduce((a, i) => a + i.detections.length, 0),   color: "text-[var(--accent)]"},
                    { label: "ERRORS",     value: images.filter(i => i.status === "error").length,       color: "text-red-500"       },
                  ].map(s => (
                    <div key={s.label} className="flex justify-between items-end">
                      <span className="text-[10px] text-[var(--text-dim)]">{s.label}</span>
                      <span className={cn("text-2xl font-bold", s.color)}>{s.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══ ANNOTATE ════════════════════════════════════════════════════════ */}
        {activeTab === "annotate" && (
          <div className="flex-1 flex flex-col overflow-hidden">

            {/* ── Mode toggle ─────────────────────────────────────────────── */}
            <div className="h-10 bg-[var(--bg-header)] border-b border-[var(--border)] flex items-center px-4 gap-2 flex-shrink-0">
              {(["images", "video"] as const).map(m => (
                <button key={m} onClick={() => setAnnotateMode(m)}
                  className={cn("px-4 py-1 rounded text-[10px] font-bold uppercase tracking-widest transition-all",
                    annotateMode === m ? "bg-[var(--accent)] text-black" : "text-[var(--text-dim)] hover:text-white")}>
                  {m === "images" ? "📷  Images" : "🎬  Video"}
                </button>
              ))}
            </div>

            {/* ── Images sub-mode ──────────────────────────────────────────── */}
            {annotateMode === "images" && (
            <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
            {/* Left panel */}
            <div className="w-full md:w-[280px] border-b md:border-b-0 md:border-r border-[var(--border)] bg-[var(--bg-side)] p-4 overflow-y-auto no-scrollbar space-y-4 max-h-[40vh] md:max-h-none">
              {/* Filter */}
              <div>
                <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest mb-3">FILTER</h3>
                <div className="space-y-1">
                  {[
                    { id: "all",        label: "All Images",        icon: LayoutGrid,   count: images.length },
                    { id: "detected",   label: "With Detections",   icon: CheckCircle2, count: images.filter(i => i.detections.length > 0).length },
                    { id: "zero",       label: "Zero Detections",   icon: XCircle,      count: images.filter(i => i.detections.length === 0).length },
                    { id: "annotated",  label: "With Annotations",  icon: Target,       count: images.filter(i => i.annotations.length > 0).length },
                  ].map(f => (
                    <button key={f.id} onClick={() => { setFilterMode(f.id as any); setZoom(1); setPanOffset({ x: 0, y: 0 }); }}
                      className={cn("w-full flex items-center gap-2 px-3 py-2 rounded border text-[10px] font-bold uppercase tracking-wider transition-all",
                        filterMode === f.id ? "bg-[var(--accent)]/10 border-[var(--accent)] text-[var(--accent)]" : "bg-black/20 border-[var(--border)] text-[var(--text-dim)] hover:text-white")}>
                      <f.icon className="w-3 h-3" />{f.label}
                      <span className={cn("ml-auto font-mono", f.id === "annotated" && f.count > 0 ? "text-yellow-400 opacity-100" : "opacity-50")}>{f.count}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Detection list */}
              <div>
                <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest mb-2">DETECTIONS &amp; ANNOTATIONS</h3>
                {/* Summary counts — always visible so user knows what loaded */}
                <div className="flex gap-2 mb-3">
                  <div className="flex-1 bg-black/30 rounded px-2 py-1.5 text-center">
                    <div className="text-[14px] font-bold text-emerald-400">{selectedImage?.detections.length ?? 0}</div>
                    <div className="text-[8px] text-[var(--text-dim)] uppercase tracking-wider">Detections</div>
                  </div>
                  <div className="flex-1 bg-black/30 rounded px-2 py-1.5 text-center">
                    <div className={cn("text-[14px] font-bold", (selectedImage?.annotations.length ?? 0) > 0 ? "text-yellow-400" : "text-[var(--text-dim)]")}>
                      {selectedImage?.annotations.length ?? 0}
                    </div>
                    <div className="text-[8px] text-[var(--text-dim)] uppercase tracking-wider">Annotations</div>
                  </div>
                </div>
                {!selectedImage && <p className="text-[10px] text-[var(--text-dim)] italic">Select an image to review.</p>}
                {selectedImage && !selectedImage.detections.length && !selectedImage.annotations.length &&
                  <p className="text-[10px] text-[var(--text-dim)] italic">No detections on this image.</p>}
                <div className="space-y-2">
                  {selectedImage?.detections.map(det => (
                    <div key={det.id}
                      className={cn("reef-card p-3 flex items-center justify-between cursor-pointer transition-all", det.isFalsePositive ? "opacity-40 border-red-500/30" : "border-l-4")}
                      style={{ borderLeftColor: det.isFalsePositive ? undefined : colorForLabel(det.label) }}
                      onClick={() => toggleFalsePositive(det.id)}>
                      <div>
                        <div className="text-[11px] font-bold text-white uppercase">{det.label}</div>
                        <div className="text-[9px] text-[var(--text-dim)] font-mono">{Math.round(det.confidence * 100)}% conf</div>
                      </div>
                      <span className={cn("text-[8px] font-bold uppercase tracking-tighter", det.isFalsePositive ? "text-red-500" : "text-emerald-500")}>
                        {det.isFalsePositive ? "FALSE +" : "VALID"}
                      </span>
                    </div>
                  ))}
                  {selectedImage?.annotations.map(ann => (
                    <div key={ann.id} className="reef-card p-3 flex items-center justify-between border-l-4 group"
                      style={{ borderLeftColor: ann.color ?? "#eab308", backgroundColor: `${ann.color ?? "#eab308"}18` }}>
                      <div>
                        <div className="text-[11px] font-bold uppercase" style={{ color: ann.color ?? "#eab308" }}>{ann.label}</div>
                        <div className="text-[9px] font-mono opacity-60" style={{ color: ann.color ?? "#eab308" }}>Manual</div>
                      </div>
                      <button className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-red-500/20 rounded" onClick={() => deleteAnnotation(ann.id)}>
                        <X className="w-3 h-3 text-red-400" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Guide */}
              <div className="bg-[var(--accent)]/5 p-4 rounded-lg border border-[var(--accent)]/20 text-[11px] text-[var(--text-dim)] space-y-2 leading-relaxed">
                <h3 className="text-[10px] font-bold text-[var(--accent)] uppercase tracking-widest mb-2">HOW TO ANNOTATE</h3>
                <p><span className="text-[var(--accent)] font-bold">Click</span> any detection box to mark it as a false positive (turns red). Or click the item in the list on the left.</p>
                <p><span className="text-[var(--accent)] font-bold">Drag</span> on the image to draw a "Missed Target" box around anything the AI missed.</p>
                <p>Hover a yellow box and click <span className="text-white font-bold">✕</span> to remove it, or press <span className="text-white font-bold">UNDO</span>.</p>
                <div className="mt-2 space-y-1 text-[10px]">
                  <p><span className="inline-block w-2 h-2 rounded-sm bg-red-500 mr-1"></span>CoTS / starfish</p>
                  <p><span className="inline-block w-2 h-2 rounded-sm bg-blue-500 mr-1"></span>Giant clam</p>
                  <p><span className="inline-block w-2 h-2 rounded-sm bg-orange-500 mr-1"></span>Bleaching</p>
                  <p><span className="inline-block w-2 h-2 rounded-sm bg-purple-500 mr-1"></span>Fish</p>
                  <p><span className="inline-block w-2 h-2 rounded-sm bg-emerald-500 mr-1"></span>Other</p>
                </div>
              </div>
            </div>

            {/* Canvas */}
            <div className="flex-1 bg-black flex flex-col relative overflow-hidden">
              {/* Toolbar */}
              <div className="h-12 bg-black/40 border-b border-[var(--border)] flex items-center justify-between px-6 z-10">
                <div className="flex items-center gap-3">
                  <Button variant="ghost" size="sm" className="text-[var(--text-dim)] hover:text-white gap-2" onClick={() => navigateImages("prev")}><ArrowLeft className="w-4 h-4" />PREV</Button>
                  <Button variant="ghost" size="sm" className="text-[var(--text-dim)] hover:text-white gap-2" onClick={() => navigateImages("next")}>NEXT<ArrowRight className="w-4 h-4" /></Button>
                  {selectedImage && <span className="text-[10px] font-mono text-[var(--text-dim)]">{filteredImages.findIndex(i => i.id === selectedImageId) + 1} / {filteredImages.length}</span>}
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="ghost" size="sm" className="text-[var(--text-dim)] hover:text-white gap-2" onClick={undoAnnotation} disabled={!currentImageHistory.length}><Undo className="w-4 h-4" />UNDO</Button>
                  <div className="w-px h-4 bg-[var(--border)]" />
                  <Button variant="ghost" size="icon" className="h-7 w-7 text-[var(--text-dim)] hover:text-white" onClick={() => { setZoom(z => Math.max(0.5, z - 0.2)); }}><ZoomOut className="w-4 h-4" /></Button>
                  <span className="text-[10px] font-mono w-10 text-center text-[var(--text-dim)]">{Math.round(zoom * 100)}%</span>
                  <Button variant="ghost" size="icon" className="h-7 w-7 text-[var(--text-dim)] hover:text-white" onClick={() => { setZoom(z => Math.min(8, z + 0.2)); }}><ZoomIn className="w-4 h-4" /></Button>
                  <Button variant="ghost" size="sm" className="text-[var(--text-dim)] hover:text-white text-[9px]" onClick={() => { setZoom(1); setPanOffset({ x: 0, y: 0 }); }}>FIT</Button>
                  <div className="w-px h-4 bg-[var(--border)]" />
                  <span className="text-[10px] text-yellow-400 font-mono gap-2 flex items-center" title="Use Save in the header to save your session">
                    <Save className="w-3 h-3" />SAVE OFTEN
                  </span>
                  <div className="w-px h-4 bg-[var(--border)]" />
                  <Button size="sm" className="reef-button-primary h-7 px-3 text-[10px] gap-1.5"
                    onClick={exportAndTrain}
                    disabled={images.filter(i => i.status === "completed" || i.annotations.length > 0).length === 0}>
                    <Cpu className="w-3 h-3" />Export &amp; Retrain
                  </Button>
                </div>
              </div>

              {/* Control hint */}
              <div className="px-4 py-1 bg-black/30 border-b border-[var(--border)]/50 text-[9px] text-[var(--text-dim)] flex gap-4">
                <span><span className="text-white">Drag</span> — draw box</span>
                <span><span className="text-white">Space+drag</span> — pan</span>
                <span><span className="text-white">Scroll</span> — zoom</span>
                <span className="ml-auto opacity-60">📱 Apple Pencil supported</span>
              </div>

              {/* Image + overlays */}
              <div ref={canvasWrapperRef} className="flex-1 relative flex items-center justify-center overflow-hidden" onWheel={handleWheel}>
                {selectedImage ? (
                  <div ref={imageContainerRef}
                    className="relative"
                    onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp}
                    onTouchStart={handleTouchStart} onTouchMove={handleTouchMove} onTouchEnd={handleTouchEnd}
                    style={{
                      transform: `scale(${zoom}) translate(${panOffset.x}px, ${panOffset.y}px)`,
                      cursor: isPanning ? "grabbing" : "crosshair",
                      userSelect: "none",
                      touchAction: "none",
                    }}>
                    <img src={selectedImage.url} className="max-h-[75vh] w-auto object-contain select-none" draggable={false} />
                    <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 1000 1000" preserveAspectRatio="none">
                      {selectedImage.detections.map(det => (
                        <g key={det.id} className="cursor-pointer pointer-events-auto" onClick={e => { e.stopPropagation(); toggleFalsePositive(det.id); }}>
                          <rect x={det.bbox[0]} y={det.bbox[1]} width={det.bbox[2]-det.bbox[0]} height={det.bbox[3]-det.bbox[1]}
                            fill={det.isFalsePositive ? "rgba(239,68,68,0.15)" : "transparent"}
                            stroke={det.isFalsePositive ? "#ef4444" : colorForLabel(det.label)} strokeWidth="2" />
                          <text x={det.bbox[0]+2} y={det.bbox[1]-4} fill={det.isFalsePositive ? "#ef4444" : "white"} fontSize="11" fontWeight="bold">
                            {det.isFalsePositive ? "FALSE +" : `${det.label} ${Math.round(det.confidence*100)}%`}
                          </text>
                        </g>
                      ))}
                      {selectedImage.annotations.map(ann => (
                        <g key={ann.id}>
                          <rect x={ann.bbox[0]} y={ann.bbox[1]} width={ann.bbox[2]-ann.bbox[0]} height={ann.bbox[3]-ann.bbox[1]}
                            fill={`${ann.color ?? "#eab308"}28`} stroke={ann.color ?? "#eab308"} strokeWidth="2" strokeDasharray="6 3" />
                          <text x={ann.bbox[0]+2} y={ann.bbox[1]-4} fill={ann.color ?? "#eab308"} fontSize="11" fontWeight="bold">{ann.label.toUpperCase()}</text>
                        </g>
                      ))}
                      {currentBox && (
                        <rect x={currentBox[0]} y={currentBox[1]} width={currentBox[2]-currentBox[0]} height={currentBox[3]-currentBox[1]}
                          fill="rgba(234,179,8,0.1)" stroke="#eab308" strokeWidth="2" strokeDasharray="4 4" />
                      )}
                    </svg>
                  </div>
                ) : (
                  <div className="text-center space-y-3">
                    <Target className="w-14 h-14 mx-auto text-[var(--border)]" />
                    <h2 className="text-xl font-bold text-white">No image selected</h2>
                    <p className="text-[var(--text-dim)] text-sm max-w-xs mx-auto">Run detection first, then come here to review results.</p>
                  </div>
                )}
              </div>


            </div>
            </div>
            )} {/* end images */}

            {/* ── Video sub-mode ───────────────────────────────────────────── */}
            {annotateMode === "video" && (
              <div className="flex-1 flex overflow-hidden">
                <div className="w-[300px] border-r border-[var(--border)] bg-[var(--bg-side)] p-5 space-y-6 overflow-y-auto no-scrollbar">
                  <div>
                    <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest mb-3">UPLOAD VIDEO</h3>
                    <div className="border-2 border-dashed border-[var(--border)] rounded-lg p-8 text-center hover:border-[var(--accent)] transition-colors cursor-pointer" onClick={() => videoInputRef.current?.click()}>
                      <Video className="w-8 h-8 mx-auto mb-3 text-[var(--text-dim)]" />
                      <div className="text-[12px] font-bold text-white">{videoUrl ? "Video loaded ✓" : "Drop video here"}</div>
                      <div className="text-[10px] text-[var(--text-dim)] mt-1">MP4, MOV, AVI</div>
                      <input type="file" ref={videoInputRef} className="hidden" accept="video/*" onChange={handleVideoUpload} />
                    </div>
                  </div>
                  <Button className="reef-button-primary w-full h-11 gap-2" onClick={captureFrame} disabled={!videoUrl}>
                    <Target className="w-4 h-4" />Capture Frame
                  </Button>
                  <div>
                    <div className="flex justify-between items-center mb-3">
                      <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest">CAPTURES ({captures.length})</h3>
                      {captures.length > 0 && <button className="text-[10px] text-[var(--accent)] hover:underline" onClick={() => setCaptures([])}>CLEAR ALL</button>}
                    </div>
                    {captures.length === 0 && <p className="text-[10px] text-[var(--text-dim)] italic">No frames captured yet.</p>}
                    <div className="space-y-2">
                      {captures.map(cap => (
                        <div key={cap.id} className="reef-card p-2 flex gap-3 group relative">
                          <div className="w-[72px] h-11 bg-black rounded overflow-hidden flex-shrink-0">
                            <img src={cap.thumbnailUrl} className="w-full h-full object-cover" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="text-[10px] font-mono text-white">{cap.timestamp.toFixed(2)}s</div>
                            <a href={cap.fullImageUrl} download={`frame_${cap.timestamp.toFixed(2)}s.jpg`}
                              className="text-[9px] text-[var(--accent)] hover:underline flex items-center gap-1 mt-1">
                              <Download className="w-3 h-3" />Download
                            </a>
                          </div>
                          <button className="absolute -top-1 -right-1 bg-red-500 rounded-full p-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                            onClick={() => setCaptures(p => p.filter(c => c.id !== cap.id))}>
                            <X className="w-3 h-3 text-white" />
                          </button>
                        </div>
                      ))}
                    </div>
                    {captures.length > 0 && (
                      <Button variant="outline" className="w-full mt-3 gap-2 border-[var(--accent)] text-[var(--accent)] hover:bg-[var(--accent)]/10 h-10"
                        onClick={exportCaptures}>
                        <DownloadCloud className="w-4 h-4" />Export All ({captures.length})
                      </Button>
                    )}
                  </div>
                </div>
                <div className="flex-1 bg-black flex flex-col">
                  {videoUrl ? (
                    <>
                      <div className="flex-1 flex items-center justify-center">
                        <video ref={videoRef} src={videoUrl} className="max-h-[70vh] w-auto"
                          onTimeUpdate={() => setVideoCurrentTime(videoRef.current?.currentTime ?? 0)}
                          onLoadedMetadata={() => setVideoDuration(videoRef.current?.duration ?? 0)}
                          onPlay={() => setVideoPlaying(true)} onPause={() => setVideoPlaying(false)} />
                      </div>
                      <div className="h-20 bg-black/60 border-t border-[var(--border)] flex items-center px-12 gap-6">
                        <Button variant="ghost" size="icon" className="text-white h-12 w-12"
                          onClick={() => videoPlaying ? videoRef.current?.pause() : videoRef.current?.play()}>
                          <Play className="w-7 h-7" />
                        </Button>
                        <div className="flex-1 h-2 bg-[var(--border)] rounded-full relative cursor-pointer"
                          onClick={e => {
                            if (!videoRef.current || !videoDuration) return;
                            const r = e.currentTarget.getBoundingClientRect();
                            videoRef.current.currentTime = ((e.clientX - r.left) / r.width) * videoDuration;
                          }}>
                          <div className="absolute inset-y-0 left-0 bg-[var(--accent)] rounded-full pointer-events-none"
                            style={{ width: videoDuration ? `${(videoCurrentTime / videoDuration) * 100}%` : "0%" }} />
                        </div>
                        <span className="text-[12px] font-mono text-white whitespace-nowrap">
                          {formatTime(videoCurrentTime)} / {formatTime(videoDuration)}
                        </span>
                      </div>
                    </>
                  ) : (
                    <div className="flex-1 flex items-center justify-center">
                      <div className="text-center space-y-3">
                        <Video className="w-14 h-14 mx-auto text-[var(--border)]" />
                        <h2 className="text-xl font-bold text-white">Load a video</h2>
                        <p className="text-[var(--text-dim)] text-sm">Upload a reef survey video, scrub to a frame, and capture it.</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )} {/* end video */}

          </div>
        )} {/* end annotate tab */}

        {/* ══ TRAIN ══════════════════════════════════════════════════════════ */}
        {activeTab === "train" && (
          <div className="flex-1 p-8 overflow-y-auto no-scrollbar space-y-8">
            <div className="max-w-4xl space-y-8">

              <div className="flex items-center gap-3">
                <Cpu className="w-8 h-8 text-pink-500" />
                <div>
                  <h2 className="text-2xl font-bold text-white uppercase tracking-widest">Train Local Model</h2>
                  <p className="text-[11px] text-[var(--text-dim)] mt-0.5">Train YOLOv8 on your annotated images — no Roboflow subscription needed.</p>
                </div>
              </div>

              {/* ── Current session summary ─────────────────────────────── */}
              {(() => {
                const exportable   = images.filter(i => i.status === "completed" || i.annotations.length > 0);
                const validDets    = exportable.reduce((a, i) => a + i.detections.filter(d => !d.isFalsePositive).length, 0);
                const manualAnns   = exportable.reduce((a, i) => a + i.annotations.length, 0);
                const falsePos     = images.reduce((a, i) => a + i.detections.filter(d => d.isFalsePositive).length, 0);
                const totalBoxes   = validDets + manualAnns;
                const unannotated  = images.filter(i => i.detections.length === 0 && i.annotations.length === 0).length;
                const readyModel   = models.find(m => m.status === "Ready");
                const classNames   = readyModel?.classNames ?? [];
                return (
                  <div className="reef-card space-y-4 border-[var(--accent)]/30 bg-[var(--accent)]/5">
                    <div className="flex items-center justify-between">
                      <h3 className="text-[10px] font-bold text-[var(--accent)] uppercase tracking-widest">Current Session — Training Summary</h3>
                      <span className="text-[9px] text-[var(--text-dim)]">from loaded images</span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      {[
                        { label: "Images to train on", value: exportable.length,  color: exportable.length > 0 ? "text-white" : "text-[var(--text-dim)]", sub: `of ${images.length} total` },
                        { label: "Total label boxes",  value: totalBoxes,          color: totalBoxes > 0 ? "text-emerald-400" : "text-[var(--text-dim)]", sub: `${validDets} AI + ${manualAnns} manual` },
                        { label: "False positives",    value: falsePos,            color: falsePos > 0 ? "text-orange-400" : "text-[var(--text-dim)]", sub: "trained as negatives ✓" },
                      ].map(s => (
                        <div key={s.label} className="bg-black/30 rounded-lg p-3 text-center">
                          <div className={`text-[22px] font-bold ${s.color}`}>{s.value}</div>
                          <div className="text-[9px] text-white uppercase tracking-wider mt-0.5">{s.label}</div>
                          <div className="text-[8px] text-[var(--text-dim)] mt-0.5">{s.sub}</div>
                        </div>
                      ))}
                    </div>
                    {classNames.length > 0 && (
                      <div className="text-[10px] text-[var(--text-dim)]">
                        Classes: {classNames.map((c, i) => (
                          <span key={c} className="text-white font-mono">{c}{i < classNames.length-1 ? ", " : ""}</span>
                        ))}
                      </div>
                    )}
                    {unannotated > 0 && (
                      <div className="p-2 bg-yellow-500/10 border border-yellow-500/20 rounded text-[10px] text-yellow-300">
                        💡 {unannotated} images have no detections or annotations yet — annotating more will improve the model.
                        Each retrain cycle the model finds more, reducing your manual work.
                      </div>
                    )}
                    {exportable.length > 0 && (
                      <div className="space-y-3">
                        <div className="flex gap-2 p-1 bg-black/30 rounded border border-[var(--border)]">
                          {([
                            { id: "annotated_only", label: "Annotated only", desc: `${images.filter(i => i.annotations.length > 0 || i.detections.length > 0).length} images` },
                            { id: "all_processed",  label: "All processed",  desc: `${exportable.length} images` },
                          ] as const).map(opt => (
                            <button key={opt.id} onClick={() => setExportMode(opt.id)}
                              className={cn("flex-1 py-1.5 px-2 rounded text-[10px] font-bold transition-all text-center",
                                exportMode === opt.id ? "bg-[var(--accent)] text-black" : "text-[var(--text-dim)] hover:text-white")}>
                              {opt.label}
                              <span className="block text-[8px] font-normal opacity-70">{opt.desc}</span>
                            </button>
                          ))}
                        </div>
                        {exportMode === "all_processed" && (
                          <p className="text-[10px] text-yellow-400">⚠ Including unannotated images can reduce accuracy — only use if most images have detections.</p>
                        )}
                        {exportMode === "annotated_only" && images.filter(i => i.annotations.length > 0 || i.detections.some(d => !d.isFalsePositive)).length < 50 && (
                          <p className="text-[10px] text-yellow-400">⚠ Fewer than 50 annotated images — consider annotating more before retraining for best results.</p>
                        )}
                        <Button size="sm" className="reef-button-primary w-full gap-2 h-9"
                          onClick={exportAndTrain}>
                          <Cpu className="w-4 h-4" />Export &amp; Retrain from current session →
                        </Button>
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* ── Step 1: Dataset ─────────────────────────────────────── */}
              <div className="reef-card space-y-4">
                <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest">Step 1 — Dataset</h3>

                <div className="grid grid-cols-2 gap-4">
                  {/* Upload a zip */}
                  <div className="border-2 border-dashed border-[var(--border)] rounded-lg p-6 text-center hover:border-[var(--accent)] transition-colors cursor-pointer"
                    onClick={() => trainFileRef.current?.click()}>
                    <DownloadCloud className="w-8 h-8 mx-auto mb-2 text-[var(--text-dim)]" />
                    <p className="text-[12px] font-bold text-white">Upload Export Zip</p>
                    <p className="text-[10px] text-[var(--text-dim)] mt-1">Use the zip from the EXPORT tab</p>
                    {trainUploading && <p className="text-[10px] text-[var(--accent)] mt-2 animate-pulse">Uploading…</p>}
                    <input type="file" ref={trainFileRef} accept=".zip" className="hidden" onChange={async e => {
                      const file = e.target.files?.[0]; if (!file) return;
                      setTrainUploading(true);
                      const fd = new FormData(); fd.append("file", file);
                      try {
                        const r = await fetch("/api/train/upload-dataset", { method: "POST", body: fd });
                        const d = await r.json();
                        if (r.ok) { setTrainDatasetPath(d.dataset_path); fetchTrainDatasets(); toast.success("Dataset ready — " + file.name); }
                        else toast.error(d.error ?? "Upload failed");
                      } catch { toast.error("Upload failed"); }
                      setTrainUploading(false);
                      e.target.value = "";
                    }} />
                  </div>

                  {/* Pick existing */}
                  <div className="space-y-2">
                    <p className="text-[10px] text-[var(--text-dim)] uppercase tracking-wider font-bold">Or pick a saved dataset</p>
                    <Button variant="outline" size="sm" className="border-[var(--border)] text-[var(--text-dim)] hover:text-white gap-2 text-[10px]"
                      onClick={fetchTrainDatasets}>
                      <Shell className="w-3 h-3" />Refresh list
                    </Button>
                    {trainDatasets.length === 0
                      ? <p className="text-[10px] text-[var(--text-dim)] italic">No saved datasets yet — upload a zip first.</p>
                      : <div className="space-y-1 max-h-40 overflow-y-auto">
                          {trainDatasets.map(ds => (
                            <div key={ds.yaml_path}
                              className={cn("p-2 rounded border cursor-pointer text-[11px] transition-all",
                                trainDatasetPath === ds.yaml_path
                                  ? "border-[var(--accent)] text-[var(--accent)] bg-[var(--accent)]/5"
                                  : "border-[var(--border)] text-[var(--text-dim)] hover:border-[var(--accent)] hover:text-white")}
                              onClick={() => setTrainDatasetPath(ds.yaml_path)}>
                              <span className="font-bold">{ds.name}</span>
                              <span className="ml-2 opacity-60">{ds.image_count.toLocaleString()} images</span>
                            </div>
                          ))}
                        </div>
                    }
                    {trainDatasetPath && (
                      <p className="text-[9px] font-mono text-emerald-400 truncate" title={trainDatasetPath}>✓ {trainDatasetPath}</p>
                    )}
                  </div>
                </div>
              </div>

              {/* ── Step 2: Config ──────────────────────────────────────── */}
              <div className="reef-card space-y-4">
                <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest">Step 2 — Configuration</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div>
                      <label className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-wider block mb-2">Base Model</label>
                      <select value={trainBaseModel} onChange={e => setTrainBaseModel(e.target.value)}
                        className="w-full bg-[#1a1a2e] border border-[var(--border)] rounded px-3 py-2 text-sm text-white focus:border-[var(--accent)] outline-none">
                        <option value="">— Select a base model —</option>
                        <optgroup label="── Fine-tune existing model (RECOMMENDED) ──">
                          {savedModels.map(m => (
                            <option key={m.path} value={m.path}>⭐ Fine-tune: {m.name}</option>
                          ))}
                        </optgroup>
                        <optgroup label="── Train from scratch ──">
                          <option value="yolov8n.pt">YOLOv8 Nano — fastest (good for testing only)</option>
                          <option value="yolov8s.pt">YOLOv8 Small — balanced</option>
                          <option value="yolov8m.pt">YOLOv8 Medium — more accurate</option>
                          <option value="yolov8l.pt">YOLOv8 Large — high accuracy, needs good GPU</option>
                        </optgroup>
                      </select>
                      {!trainBaseModel && (
                        <p className="text-[10px] text-red-400 mt-1">⚠ Please select a base model before training.</p>
                      )}
                      {trainBaseModel && savedModels.some(m => m.path === trainBaseModel) && (
                        <p className="text-[10px] text-emerald-400 mt-1">✓ Fine-tuning — your existing model's knowledge is preserved and improved.</p>
                      )}
                      {trainBaseModel && !savedModels.some(m => m.path === trainBaseModel) && trainBaseModel !== "" && (
                        <p className="text-[10px] text-yellow-400 mt-1">⚠ Training from scratch — no existing knowledge carried over.</p>
                      )}
                    </div>
                    <div>
                      <label className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-wider block mb-2">Epochs — {trainEpochs}</label>
                      <input type="range" min={10} max={300} step={10} value={trainEpochs}
                        onChange={e => setTrainEpochs(Number(e.target.value))}
                        className="w-full h-2 rounded-full appearance-none cursor-pointer"
                        style={{ background: `linear-gradient(to right, var(--accent) ${(trainEpochs-10)/290*100}%, var(--border) ${(trainEpochs-10)/290*100}%)`, accentColor: "var(--accent)" }} />
                      <div className="flex justify-between text-[9px] text-[var(--text-dim)] mt-1"><span>10 (quick test)</span><span>300 (full train)</span></div>
                    </div>
                    <div>
                      <label className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-wider block mb-2">Data Augmentation</label>
                      <div className="flex gap-1 p-1 bg-black/30 rounded border border-[var(--border)]">
                        {([
                          { id: "off",      label: "Off",      desc: "No augmentation" },
                          { id: "standard", label: "Standard", desc: "Flip + brightness" },
                          { id: "heavy",    label: "Heavy ⭐",  desc: "Best for small datasets" },
                        ] as const).map(opt => (
                          <button key={opt.id} onClick={() => setAugmentation(opt.id)}
                            className={cn("flex-1 py-1.5 px-1 rounded text-[9px] font-bold transition-all text-center",
                              augmentation === opt.id ? "bg-[var(--accent)] text-black" : "text-[var(--text-dim)] hover:text-white")}>
                            {opt.label}
                            <span className="block text-[8px] font-normal opacity-70">{opt.desc}</span>
                          </button>
                        ))}
                      </div>
                      {augmentation === "heavy" && (
                        <p className="text-[10px] text-emerald-400 mt-1">✓ Flipping, rotation, colour jitter, mosaic — multiplies your dataset ~8×</p>
                      )}
                      {augmentation === "off" && (
                        <p className="text-[10px] text-yellow-400 mt-1">⚠ Not recommended — only use if images are already augmented</p>
                      )}
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div>
                      <label className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-wider block mb-2">Image Size — {trainImgSize}px</label>
                      <select value={trainImgSize} onChange={e => setTrainImgSize(Number(e.target.value))}
                        className="w-full bg-[#1a1a2e] border border-[var(--border)] rounded px-3 py-2 text-sm text-white focus:border-[var(--accent)] outline-none">
                        <option value={320}>320px — fastest</option>
                        <option value={480}>480px</option>
                        <option value={640}>640px — recommended</option>
                        <option value={1280}>1280px — high detail, needs VRAM</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-wider block mb-2">Batch Size</label>
                      <select value={trainBatch} onChange={e => setTrainBatch(Number(e.target.value))}
                        className="w-full bg-[#1a1a2e] border border-[var(--border)] rounded px-3 py-2 text-sm text-white focus:border-[var(--accent)] outline-none">
                        <option value={-1}>Auto (recommended)</option>
                        <option value={4}>4 — low VRAM GPU</option>
                        <option value={8}>8</option>
                        <option value={16}>16</option>
                        <option value={32}>32 — high VRAM GPU</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-wider block mb-2">Model Name <span className="text-[9px] font-normal opacity-50">(prefix: cots_, fish_, clam_)</span></label>
                      <input
                        type="text"
                        value={modelName}
                        onChange={e => setModelName(e.target.value.replace(/[^a-zA-Z0-9_-]/g, "_"))}
                        placeholder="e.g. angelfish_v1, cots_survey_2026"
                        className="w-full bg-black/40 border border-[var(--border)] rounded px-3 py-2 text-sm text-white placeholder-[var(--text-dim)] focus:border-[var(--accent)] outline-none font-mono"
                      />
                      <p className="text-[9px] text-[var(--text-dim)] mt-1">Saved as <span className="font-mono text-white">{modelName ? modelName + ".pt" : "reef_train_[timestamp].pt"}</span></p>
                    </div>
                  </div>
                </div>
              </div>

              {/* ── Step 3: Run ─────────────────────────────────────────── */}
              <div className="reef-card space-y-4">
                <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest">Step 3 — Train</h3>

                <div className="flex gap-3">
                  {!trainRunning ? (
                    <Button className="reef-button-primary h-12 px-8 gap-2"
                      disabled={!trainDatasetPath || !trainBaseModel}
                      onClick={async () => {
                        setTrainRunning(true); setTrainProgress([]); setTrainDone(null); setTrainError(null);
                        const r = await fetch("/api/train/start", {
                          method: "POST", headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ dataset_path: trainDatasetPath, base_model: trainBaseModel, epochs: trainEpochs, img_size: trainImgSize, batch: trainBatch, augmentation, model_name: modelName.trim() || "" }),
                        });
                        const d = await r.json();
                        if (!r.ok) { setTrainRunning(false); toast.error(d.error ?? "Failed to start"); return; }
                        // Connect SSE
                        const es = new EventSource("/api/train/progress");
                        trainEsRef.current = es;
                        es.onmessage = ev => {
                          const msg = JSON.parse(ev.data);
                          if (msg.type === "epoch") setTrainProgress(prev => [...prev, msg]);
                          else if (msg.type === "done") {
                            setTrainProgress(prev => {
                              const lastEp = prev[prev.length - 1];
                              const result = { ...msg, mAP50: lastEp?.mAP50, epochs: lastEp?.epoch };
                              setTrainDone(result);
                              // Save to training history
                              if (lastEp?.mAP50) {
                                const entry = {
                                  date: new Date().toISOString(),
                                  modelName: msg.model_name || "reef_train",
                                  mAP50: lastEp.mAP50,
                                  epochs: lastEp.epoch,
                                  datasetSize: trainDatasets.find(d => d.yaml_path === trainDatasetPath)?.image_count ?? 0,
                                  baseModel: trainBaseModel.split(/[\/]/).pop() ?? trainBaseModel,
                                };
                                setTrainingHistory(prev2 => {
                                  const updated = [...prev2, entry];
                                  localStorage.setItem("reef_training_history", JSON.stringify(updated));
                                  return updated;
                                });
                              }
                              return prev;
                            });
                            setTrainRunning(false);
                            toast.success("Training complete! Model saved to models/");
                            es.close(); scanLocalModels();
                          } else if (msg.type === "error") {
                            setTrainError(msg.message); setTrainRunning(false);
                            toast.error("Training failed: " + msg.message); es.close();
                          } else if (msg.type === "cancelled") {
                            setTrainRunning(false); toast.info("Training cancelled"); es.close();
                          }
                        };
                        es.onerror = () => { setTrainRunning(false); es.close(); };
                      }}>
                      <Play className="w-5 h-5" />Start Training
                    </Button>
                  ) : (
                    <Button className="h-12 px-8 gap-2 border-2 border-red-500 text-red-500 bg-transparent hover:bg-red-500/10"
                      onClick={async () => {
                        await fetch("/api/train/cancel", { method: "POST" });
                        trainEsRef.current?.close();
                      }}>
                      <Loader2 className="w-5 h-5 animate-spin" />Cancel Training
                    </Button>
                  )}
                </div>

                {/* Progress */}
                {trainRunning && trainProgress.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-[var(--text-dim)]">Epoch {trainProgress[trainProgress.length-1].epoch} / {trainProgress[trainProgress.length-1].total}</span>
                      <span className="text-[var(--accent)] font-mono">mAP50: {(trainProgress[trainProgress.length-1].mAP50 * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-[var(--border)] rounded-full h-2">
                      <div className="bg-[var(--accent)] h-2 rounded-full transition-all"
                        style={{ width: `${(trainProgress[trainProgress.length-1].epoch / trainProgress[trainProgress.length-1].total) * 100}%` }} />
                    </div>
                    <div className="grid grid-cols-4 gap-3">
                      {[
                        { label: "mAP50",     val: (trainProgress[trainProgress.length-1].mAP50*100).toFixed(1)+"%" },
                        { label: "Precision", val: (trainProgress[trainProgress.length-1].precision*100).toFixed(1)+"%" },
                        { label: "Recall",    val: (trainProgress[trainProgress.length-1].recall*100).toFixed(1)+"%" },
                        { label: "Box Loss",  val: trainProgress[trainProgress.length-1].box_loss?.toFixed(4) ?? "—" },
                      ].map(s => (
                        <div key={s.label} className="bg-black/40 rounded p-3 text-center">
                          <div className="text-[9px] text-[var(--text-dim)] uppercase">{s.label}</div>
                          <div className="text-lg font-bold text-white mt-1">{s.val}</div>
                        </div>
                      ))}
                    </div>
                    {/* Loss curve sparkline */}
                    {trainProgress.length > 1 && (
                      <div className="h-20 bg-black/30 rounded p-2 relative overflow-hidden">
                        <svg width="100%" height="100%" viewBox={`0 0 ${trainProgress.length} 100`} preserveAspectRatio="none">
                          <polyline
                            fill="none" stroke="var(--accent)" strokeWidth="1.5"
                            points={trainProgress.map((p, i) => `${i},${100 - (p.mAP50 * 100)}`).join(" ")} />
                        </svg>
                        <div className="absolute bottom-1 left-2 text-[8px] text-[var(--text-dim)]">mAP50 over epochs</div>
                      </div>
                    )}
                  </div>
                )}

                {/* Done */}
                {trainDone && (
                  <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg space-y-4">
                    <p className="text-emerald-400 font-bold text-sm">✓ Training complete!</p>
                    <p className="text-[11px] text-[var(--text-dim)] font-mono">{trainDone.model_name}</p>
                    {trainDone.mAP50 !== undefined && (
                      <div className="grid grid-cols-3 gap-3">
                        <div className="bg-black/30 rounded p-3 text-center">
                          <div className={cn("text-[22px] font-bold", (trainDone.mAP50 ?? 0) >= 0.7 ? "text-emerald-400" : (trainDone.mAP50 ?? 0) >= 0.5 ? "text-yellow-400" : "text-red-400")}>
                            {((trainDone.mAP50 ?? 0) * 100).toFixed(1)}%
                          </div>
                          <div className="text-[9px] text-[var(--text-dim)] uppercase mt-0.5">Final mAP50</div>
                        </div>
                        <div className="bg-black/30 rounded p-3 text-center">
                          <div className="text-[22px] font-bold text-white">{trainDone.epochs ?? "—"}</div>
                          <div className="text-[9px] text-[var(--text-dim)] uppercase mt-0.5">Epochs run</div>
                        </div>
                        <div className="bg-black/30 rounded p-3 text-center">
                          <div className={cn("text-[11px] font-bold mt-2", (trainDone.mAP50 ?? 0) >= 0.7 ? "text-emerald-400" : (trainDone.mAP50 ?? 0) >= 0.5 ? "text-yellow-400" : "text-red-400")}>
                            {(trainDone.mAP50 ?? 0) >= 0.7 ? "✓ Good" : (trainDone.mAP50 ?? 0) >= 0.5 ? "⚠ Needs more data" : "✗ Low accuracy"}
                          </div>
                          <div className="text-[9px] text-[var(--text-dim)] uppercase mt-0.5">Assessment</div>
                        </div>
                      </div>
                    )}
                    <p className="text-[11px] text-[var(--text-dim)]">Model saved to <span className="font-mono text-emerald-400">models/</span> — go to MODELS tab to load it.</p>
                    <div className="flex gap-2">
                      <Button size="sm" className="reef-button-primary gap-2" onClick={() => setActiveTab("models")}>
                        <Brain className="w-4 h-4" />Go to Models →
                      </Button>
                      {(trainDone.mAP50 ?? 0) < 0.7 && (
                        <Button size="sm" variant="outline" className="border-yellow-500/50 text-yellow-400 hover:bg-yellow-500/10 gap-2"
                          onClick={() => setActiveTab("annotate")}>
                          Annotate more →
                        </Button>
                      )}
                    </div>
                  </div>
                )}

                {/* Error */}
                {trainError && (
                  <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
                    <p className="text-red-400 font-bold text-sm">Training failed</p>
                    <p className="text-[11px] text-[var(--text-dim)] mt-1 font-mono">{trainError}</p>
                  </div>
                )}
              </div>

              {/* Estimated time */}
              <div className="reef-card text-[11px] text-[var(--text-dim)] space-y-1">
                <p className="font-bold text-white text-[12px]">Estimated training time</p>
                <p>With a modern GPU (RTX 3060+): ~1–2 min/epoch at 640px on 1,000 images.</p>
                <p>10,000 images × 50 epochs ≈ 2–4 hours. Leave it running overnight for 100+ epochs.</p>
                <p>CPU-only: ~10× slower — use Nano model and fewer epochs for testing.</p>
              </div>

            </div>
          </div>
        )}

        {/* ══ SCRAPE ══════════════════════════════════════════════════════════ */}
        {activeTab === "scrape" && (
          <div className="flex-1 p-8 overflow-y-auto no-scrollbar">
            <div className="max-w-3xl space-y-8">

              <div className="flex items-center gap-3">
                <Globe className="w-8 h-8 text-[var(--accent)]" />
                <div>
                  <h2 className="text-2xl font-bold text-white uppercase tracking-widest">Image Scraper</h2>
                  <p className="text-[11px] text-[var(--text-dim)] mt-0.5">Search the web for training images by species name — downloads to a zip ready for training.</p>
                </div>
              </div>

              {/* Search form */}
              <div className="reef-card space-y-5">
                <div className="flex gap-1 p-1 bg-black/30 rounded border border-[var(--border)]">
                  {([
                    { id: "download",   label: "Direct Download", desc: "Fast, ~30-60% success rate" },
                    { id: "screenshot", label: "Enhanced Search", desc: "Google + Bing combined" },
                  ] as const).map(opt => (
                    <button key={opt.id} onClick={() => setScrapeMode(opt.id)}
                      className={cn("flex-1 py-1.5 px-2 rounded text-[9px] font-bold transition-all text-center",
                        scrapeMode === opt.id ? "bg-[var(--accent)] text-black" : "text-[var(--text-dim)] hover:text-white")}>
                      {opt.label}
                      <span className="block text-[8px] font-normal opacity-70">{opt.desc}</span>
                    </button>
                  ))}
                </div>
                {scrapeMode === "screenshot" && (
                  <p className="text-[10px] text-emerald-400">✓ Searches both Google Images and Bing — no browser required, gets more results than Direct Download.</p>
                )}
                <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest">Search Parameters</h3>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-wider block">Scientific Name <span className="text-[9px] font-normal opacity-50">(or use common name)</span></label>
                    <input
                      type="text"
                      value={scrapeQuery}
                      onChange={e => setScrapeQuery(e.target.value)}
                      placeholder="e.g. Acanthaster planci"
                      className="w-full bg-black/40 border border-[var(--border)] rounded px-3 py-2 text-sm text-white placeholder-[var(--text-dim)] focus:border-[var(--accent)] outline-none font-mono"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-wider block">Common Name <span className="text-[9px] font-normal opacity-50">(or use scientific name)</span></label>
                    <input
                      type="text"
                      value={scrapeCommon}
                      onChange={e => setScrapeCommon(e.target.value)}
                      placeholder="e.g. Crown of Thorns starfish"
                      className="w-full bg-black/40 border border-[var(--border)] rounded px-3 py-2 text-sm text-white placeholder-[var(--text-dim)] focus:border-[var(--accent)] outline-none"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-wider block">Number of Images to Download</label>
                  <div className="flex items-center gap-3">
                    <input
                      type="number"
                      min={1} max={10000}
                      value={scrapeCount}
                      onChange={e => setScrapeCount(Math.max(1, parseInt(e.target.value) || 1))}
                      className="w-32 bg-black/40 border border-[var(--border)] rounded px-3 py-2 text-lg font-bold text-white text-center focus:border-[var(--accent)] outline-none"
                    />
                    <span className="text-[11px] text-[var(--text-dim)]">images</span>
                  </div>
                </div>

                <div className="p-3 bg-yellow-500/10 border border-yellow-500/20 rounded text-[10px] text-yellow-300 space-y-1">
                  <p className="font-bold">⚠ Image rights notice</p>
                  <p>Downloaded images are for research and training purposes only. Check licensing before commercial use. The scraper searches public image sources using the Bing Image Search API via Python.</p>
                </div>

                <Button className="reef-button-primary w-full h-12 gap-2"
                  disabled={(!scrapeQuery.trim() && !scrapeCommon.trim()) || scrapeRunning}
                  onClick={async () => {
                    setScrapeRunning(true);
                    setScrapeProgress({ found: 0, downloaded: 0, failed: 0, status: "Starting search…" });
                    try {
                      const query = [scrapeQuery.trim(), scrapeCommon.trim()].filter(Boolean).join(" ");
                      const label = (scrapeQuery.trim() || scrapeCommon.trim()).replace(/ /g, "_");
                      const endpoint = scrapeMode === "screenshot" ? "/api/screenshot-scrape" : "/api/scrape-images";
                      const body = scrapeMode === "screenshot"
                        ? { query, max_images: scrapeCount, label }
                        : { query, count: scrapeCount, label };
                      const r = await fetch(endpoint, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(body),
                      });
                      if (!r.ok) {
                        const d = await r.json();
                        toast.error(d.error ?? "Scrape failed");
                        setScrapeProgress(prev => prev ? { ...prev, status: "Failed: " + (d.error ?? "Unknown error") } : null);
                        return;
                      }
                      // Stream progress via SSE
                      const reader = r.body?.getReader();
                      const decoder = new TextDecoder();
                      if (!reader) return;
                      let buf = "";
                      while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        buf += decoder.decode(value, { stream: true });
                        const lines = buf.split("\n");
                        buf = lines.pop() ?? "";
                        for (const line of lines) {
                          if (!line.startsWith("data:")) continue;
                          try {
                            const msg = JSON.parse(line.slice(5).trim());
                            if (msg.type === "progress") {
                              const dl = msg.downloaded ?? msg.saved ?? 0;
                              setScrapeProgress({ found: msg.found ?? scrapeCount, downloaded: dl, failed: (msg.found ?? scrapeCount) - dl, status: msg.message ?? msg.status ?? "Working..." });
                            } else if (msg.type === "done") {
                              const dl = msg.downloaded ?? msg.saved ?? 0;
                              setScrapeProgress({ found: scrapeCount, downloaded: dl, failed: scrapeCount - dl, status: `✓ Complete! Zip saved to Downloads folder.` });
                              toast.success(`${dl} images captured — zip saved to Downloads`);
                            } else if (msg.type === "error") {
                              setScrapeProgress(prev => prev ? { ...prev, status: "Error: " + msg.message } : null);
                              toast.error(msg.message);
                            }
                          } catch {}
                        }
                      }
                    } catch (err: any) {
                      toast.error(err?.message ?? "Scrape failed");
                      setScrapeProgress(prev => prev ? { ...prev, status: "Failed" } : null);
                    } finally {
                      setScrapeRunning(false);
                    }
                  }}>
                  {scrapeRunning ? <Loader2 className="w-5 h-5 animate-spin" /> : <Globe className="w-5 h-5" />}
                  {scrapeRunning ? "Searching…" : "Search & Download Images"}
                </Button>
              </div>

              {/* Progress */}
              {scrapeProgress && (
                <div className="reef-card space-y-4">
                  <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest">Progress</h3>
                  <div className="grid grid-cols-3 gap-3">
                    <div className="bg-black/30 rounded p-3 text-center">
                      <div className="text-[22px] font-bold text-white">{scrapeProgress.found}</div>
                      <div className="text-[9px] text-[var(--text-dim)] uppercase mt-0.5">Found</div>
                    </div>
                    <div className="bg-black/30 rounded p-3 text-center">
                      <div className="text-[22px] font-bold text-emerald-400">{scrapeProgress.downloaded}</div>
                      <div className="text-[9px] text-[var(--text-dim)] uppercase mt-0.5">Downloaded</div>
                    </div>
                    <div className="bg-black/30 rounded p-3 text-center">
                      <div className="text-[22px] font-bold text-red-400">{scrapeProgress.failed}</div>
                      <div className="text-[9px] text-[var(--text-dim)] uppercase mt-0.5">Failed</div>
                    </div>
                  </div>
                  {scrapeRunning && (
                    <div className="w-full bg-[var(--border)] rounded-full h-2">
                      <div className="bg-[var(--accent)] h-2 rounded-full transition-all"
                        style={{ width: `${(scrapeProgress.downloaded / scrapeCount) * 100}%` }} />
                    </div>
                  )}
                  <p className="text-[11px] text-[var(--text-dim)]">{scrapeProgress.status}</p>
                </div>
              )}

              {/* Instructions */}
              <div className="reef-card text-[11px] text-[var(--text-dim)] space-y-2">
                <p className="font-bold text-white">How to use scraped images for training</p>
                <div className="space-y-2 mt-1">
                  {[
                    { n:1, text: "Search for your species above — zip saves to your Downloads folder automatically." },
                    { n:2, text: "Go to 2. UPLOAD tab → load that zip — all images appear in your session." },
                    { n:3, text: "Go to 4. ANNOTATE tab → draw boxes around the species in each image." },
                    { n:4, text: "Go to 5. TRAIN tab → Export & Retrain — uses your annotated images to build the model." },
                  ].map(s => (
                    <div key={s.n} className="flex gap-3 items-start">
                      <div className="w-5 h-5 rounded-full bg-[var(--accent)]/20 text-[var(--accent)] flex items-center justify-center flex-shrink-0 text-[10px] font-bold">{s.n}</div>
                      <p>{s.text}</p>
                    </div>
                  ))}
                </div>
                <p className="text-[10px] pt-1 border-t border-[var(--border)] mt-2">Requires <span className="font-mono text-white">icrawler</span> — installed automatically by <span className="font-mono text-white">setup_dependencies.bat</span></p>
              </div>

            </div>
          </div>
        )}

        {/* ══ HISTORY ════════════════════════════════════════════════════════ */}
        {activeTab === "history" && (
          <div className="flex-1 p-8 overflow-y-auto no-scrollbar">
            <div className="max-w-4xl space-y-8">

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-3xl">📊</span>
                  <div>
                    <h2 className="text-2xl font-bold text-white uppercase tracking-widest">Model Training History</h2>
                    <p className="text-[11px] text-[var(--text-dim)] mt-0.5">Track your model accuracy over time — every training run logged automatically.</p>
                  </div>
                </div>
                {trainingHistory.length > 0 && (
                  <Button variant="outline" size="sm"
                    className="border-red-500/30 text-red-400 hover:bg-red-500/10 text-[10px]"
                    onClick={() => { if (confirm("Clear all training history?")) { setTrainingHistory([]); localStorage.removeItem("reef_training_history"); }}}>
                    Clear History
                  </Button>
                )}
              </div>

              {trainingHistory.length === 0 ? (
                <div className="reef-card text-center py-16 space-y-3">
                  <div className="text-5xl opacity-30">📈</div>
                  <p className="text-white font-bold">No training runs yet</p>
                  <p className="text-[11px] text-[var(--text-dim)]">Every time you complete a training run it will appear here automatically.</p>
                </div>
              ) : (
                <>
                  {/* mAP50 chart */}
                  <div className="reef-card space-y-4">
                    <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest">mAP50 Over Time</h3>
                    <div className="relative h-48 bg-black/30 rounded-lg p-4">
                      <svg width="100%" height="100%" viewBox={`0 0 ${Math.max(trainingHistory.length * 60, 300)} 120`} preserveAspectRatio="none">
                        {/* Grid lines */}
                        {[0, 25, 50, 75, 100].map(pct => (
                          <g key={pct}>
                            <line x1="0" y1={120 - (pct / 100) * 120} x2="10000" y2={120 - (pct / 100) * 120}
                              stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
                          </g>
                        ))}
                        {/* 70% threshold line */}
                        <line x1="0" y1={120 - 0.7 * 120} x2="10000" y2={120 - 0.7 * 120}
                          stroke="rgba(234,179,8,0.3)" strokeWidth="1" strokeDasharray="4 4" />
                        {/* Area fill */}
                        <polygon
                          fill="rgba(0,212,180,0.1)"
                          points={[
                            ...trainingHistory.map((r, i) => `${i * 60 + 30},${120 - r.mAP50 * 120}`),
                            `${(trainingHistory.length - 1) * 60 + 30},120`,
                            `30,120`,
                          ].join(" ")}
                        />
                        {/* Line */}
                        <polyline
                          fill="none" stroke="var(--accent)" strokeWidth="2.5"
                          points={trainingHistory.map((r, i) => `${i * 60 + 30},${120 - r.mAP50 * 120}`).join(" ")}
                        />
                        {/* Data points */}
                        {trainingHistory.map((r, i) => (
                          <g key={i}>
                            <circle cx={i * 60 + 30} cy={120 - r.mAP50 * 120} r="4"
                              fill={r.mAP50 >= 0.7 ? "#10b981" : r.mAP50 >= 0.5 ? "#eab308" : "#ef4444"}
                              stroke="white" strokeWidth="1.5" />
                          </g>
                        ))}
                      </svg>
                      {/* Y axis labels */}
                      <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-[8px] text-[var(--text-dim)] pr-1 pointer-events-none">
                        <span>100%</span><span>75%</span><span>50%</span><span>25%</span><span>0%</span>
                      </div>
                    </div>
                    <div className="flex gap-4 text-[9px] text-[var(--text-dim)]">
                      <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500 inline-block"></span>≥70% Good</span>
                      <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-yellow-500 inline-block"></span>50-70% OK</span>
                      <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500 inline-block"></span>&lt;50% Needs work</span>
                      <span className="flex items-center gap-1"><span className="w-4 border-t border-dashed border-yellow-500/50 inline-block"></span>70% target</span>
                    </div>
                  </div>

                  {/* Summary stats */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    {[
                      { label: "Training Runs", value: trainingHistory.length, color: "text-white" },
                      { label: "Best mAP50", value: (Math.max(...trainingHistory.map(r => r.mAP50)) * 100).toFixed(1) + "%", color: "text-emerald-400" },
                      { label: "Latest mAP50", value: (trainingHistory[trainingHistory.length - 1].mAP50 * 100).toFixed(1) + "%",
                        color: trainingHistory[trainingHistory.length - 1].mAP50 >= 0.7 ? "text-emerald-400" : trainingHistory[trainingHistory.length - 1].mAP50 >= 0.5 ? "text-yellow-400" : "text-red-400" },
                    ].map(s => (
                      <div key={s.label} className="reef-card text-center py-4">
                        <div className={`text-3xl font-bold ${s.color}`}>{s.value}</div>
                        <div className="text-[10px] text-[var(--text-dim)] uppercase mt-1">{s.label}</div>
                      </div>
                    ))}
                  </div>

                  {/* Run table */}
                  <div className="reef-card space-y-3">
                    <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest">All Runs</h3>
                    <div className="space-y-2">
                      {[...trainingHistory].reverse().map((run, i) => (
                        <div key={i} className="flex items-center gap-4 p-3 bg-black/20 rounded border border-[var(--border)] text-[11px]">
                          <div className={`w-2 h-2 rounded-full flex-shrink-0 ${run.mAP50 >= 0.7 ? "bg-emerald-500" : run.mAP50 >= 0.5 ? "bg-yellow-500" : "bg-red-500"}`} />
                          <div className="flex-1 font-mono text-white truncate">{run.modelName}</div>
                          <div className={`font-bold w-16 text-right ${run.mAP50 >= 0.7 ? "text-emerald-400" : run.mAP50 >= 0.5 ? "text-yellow-400" : "text-red-400"}`}>
                            {(run.mAP50 * 100).toFixed(1)}%
                          </div>
                          <div className="text-[var(--text-dim)] w-20 text-right">{run.epochs} epochs</div>
                          <div className="text-[var(--text-dim)] w-24 text-right">{run.datasetSize > 0 ? run.datasetSize + " imgs" : ""}</div>
                          <div className="text-[var(--text-dim)] w-32 text-right opacity-50">{new Date(run.date).toLocaleDateString()}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Trend analysis */}
                  {trainingHistory.length >= 2 && (() => {
                    const last = trainingHistory[trainingHistory.length - 1].mAP50;
                    const prev = trainingHistory[trainingHistory.length - 2].mAP50;
                    const diff = last - prev;
                    const trend = diff > 0.01 ? "improving" : diff < -0.01 ? "declining" : "stable";
                    return (
                      <div className={`reef-card border-l-4 ${trend === "improving" ? "border-emerald-500 bg-emerald-500/5" : trend === "declining" ? "border-red-500 bg-red-500/5" : "border-yellow-500 bg-yellow-500/5"}`}>
                        <div className="flex items-center gap-3">
                          <span className="text-2xl">{trend === "improving" ? "📈" : trend === "declining" ? "📉" : "➡️"}</span>
                          <div>
                            <p className={`font-bold text-sm ${trend === "improving" ? "text-emerald-400" : trend === "declining" ? "text-red-400" : "text-yellow-400"}`}>
                              {trend === "improving" ? `Improving — up ${(diff * 100).toFixed(1)}% from last run` :
                               trend === "declining" ? `Declining — down ${(Math.abs(diff) * 100).toFixed(1)}% from last run` :
                               "Stable — similar to last run"}
                            </p>
                            <p className="text-[11px] text-[var(--text-dim)] mt-0.5">
                              {trend === "declining" ? "Check your annotations are correct and try increasing augmentation or epochs." :
                               trend === "improving" ? "Keep annotating more images to continue improving." :
                               "Add more annotated images to push accuracy higher."}
                            </p>
                          </div>
                        </div>
                      </div>
                    );
                  })()}
                </>
              )}
            </div>
          </div>
        )}

        {/* ══ REMOTE ══════════════════════════════════════════════════════════ */}
        {activeTab === "remote" && (
          <div className="flex-1 p-8 overflow-y-auto no-scrollbar">
            <div className="max-w-3xl space-y-8">
              <div className="flex items-center gap-3">
                <Globe className="w-7 h-7 text-[var(--accent)]" />
                <h2 className="text-2xl font-bold text-white uppercase tracking-widest">Remote Access via Cloudflare Tunnel</h2>
              </div>
              <p className="text-[var(--text-dim)] text-sm">Access from phone, tablet, or boat laptop — free, no hosting costs.</p>
              {[
                { title: "ONE-TIME SETUP", steps: [
                  { s: 1, t: "Download cloudflared from developers.cloudflare.com → Connect Networks → Downloads → Windows 64-bit." },
                  { s: 2, t: "Open a terminal and run:", code: "cloudflared tunnel --url http://localhost:3000" },
                  { s: 3, t: "It will print a temporary URL like https://random-name.trycloudflare.com — open that on any device." },
                  { s: 4, t: "URL changes each restart. For a permanent URL, create a free Cloudflare account and set up a named tunnel." },
                ]},
                { title: "EACH SESSION", steps: [
                  { s: 1, t: "Start the app (run_reef_ai.bat or npm run dev), then run:", code: "cloudflared tunnel --url http://localhost:3000" },
                ]},
              ].map(section => (
                <div key={section.title} className="reef-card space-y-4">
                  <h3 className="text-[var(--accent)] font-bold uppercase tracking-widest text-[11px]">{section.title}</h3>
                  {section.steps.map(step => (
                    <div key={step.s} className="flex gap-4">
                      <div className="w-6 h-6 rounded-full bg-[var(--accent)]/20 text-[var(--accent)] flex items-center justify-center flex-shrink-0 text-[12px] font-bold">{step.s}</div>
                      <div className="space-y-2">
                        <p className="text-[13px]">{step.t}</p>
                        {step.code && <div className="bg-black/60 p-3 rounded border border-[var(--border)] font-mono text-[12px] text-[var(--accent)]">{step.code}</div>}
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ══ EXPORT ══════════════════════════════════════════════════════════ */}
        {activeTab === "export" && (
          <div className="flex-1 p-8 overflow-y-auto no-scrollbar space-y-8">
            <div className="max-w-3xl space-y-8">
              <div className="flex items-center gap-3">
                <FileArchive className="w-7 h-7 text-pink-500" />
                <h2 className="text-2xl font-bold text-white uppercase tracking-widest">Training Export</h2>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                {[
                  { label: "Processed",       value: images.filter(i => i.status === "completed").length,                                    color: "text-white"         },
                  { label: "Annotated Only",   value: images.filter(i => i.status !== "completed" && i.annotations.length > 0).length,      color: "text-yellow-400"    },
                  { label: "Valid Detections", value: images.reduce((a, i) => a + i.detections.filter(d => !d.isFalsePositive).length, 0),  color: "text-[var(--accent)]"},
                  { label: "Manual Boxes",     value: images.reduce((a, i) => a + i.annotations.length, 0),                               color: "text-yellow-400"    },
                ].map(s => (
                  <div key={s.label} className="reef-card text-center py-5">
                    <div className={cn("text-3xl font-bold", s.color)}>{s.value}</div>
                    <div className="text-[10px] text-[var(--text-dim)] uppercase mt-2">{s.label}</div>
                  </div>
                ))}
              </div>
              <div className="reef-card space-y-4">
                <h3 className="text-[10px] font-bold text-[var(--text-dim)] uppercase tracking-widest">YOLO Dataset Export</h3>
                <p className="text-sm text-[var(--text-dim)] leading-relaxed">
                  Creates a ZIP with your images and YOLOv8 label files, including your false-positive corrections and missed-target annotations. Use for training or upload to any compatible platform.
                </p>
                <Button className="reef-button-primary w-full h-14 gap-3 text-base"
                  onClick={exportToRoboflow} disabled={images.filter(i => i.status === "completed" || i.annotations.length > 0).length === 0}>
                  <FileArchive className="w-5 h-5" />Export YOLO Dataset
                </Button>
              </div>
            </div>
          </div>
        )}

      </main>

      {/* Global hidden inputs — always in DOM regardless of active tab */}
      <input type="file" ref={sessionInputRef} onChange={loadSession} accept=".json" className="hidden" />

      {/* ══ THUMBNAIL REEL ══════════════════════════════════════════════════ */}
      {/* FIX: show on annotate/video/export — and show ALL images not just completed ones */}
      {(activeTab === "annotate" || activeTab === "export") && images.length > 0 && (
        <footer className="h-[120px] md:h-[112px] bg-[var(--bg-side)] border-t border-[var(--border)] flex flex-col flex-shrink-0">
          <div className="h-6 px-6 flex items-center text-[9px] font-bold text-[var(--text-dim)] uppercase tracking-widest bg-black/20 border-b border-[var(--border)]/30">
            IMAGE REEL — {filteredImages.length} {filterMode !== "all" ? filterMode : "total"}
          </div>
          <div className="flex-1 flex items-center gap-2 px-4 overflow-x-auto no-scrollbar">
            {filteredImages.map((img, reelIdx) => {
              const dCount  = img.detections.filter(d => !d.isFalsePositive).length;
              const fpCount = img.detections.filter(d => d.isFalsePositive).length;
              const mCount  = img.annotations.length;
              const statusColor = img.status === "completed" ? "border-emerald-500/40"
                               : img.status === "error"     ? "border-red-500/40"
                               : img.status === "processing"? "border-yellow-500/40"
                               : "border-transparent";
              return (
                <div key={img.id}
                  className={cn("relative flex-shrink-0 w-24 md:w-28 h-[80px] md:h-[76px] rounded border-2 transition-all cursor-pointer overflow-hidden",
                    selectedImageId === img.id ? "border-[var(--accent)] scale-105" : img.isAnnotated ? "border-yellow-500/70" : statusColor, "hover:opacity-100")}
                  onClick={() => setSelectedImageId(img.id)}>
                  <img src={img.url} className="w-full h-full object-cover opacity-70 hover:opacity-100 transition-opacity" />
                  {/* Status / count badges */}
                  <div className="absolute top-1 left-1 flex flex-col gap-0.5">
                    {dCount  > 0 && <div className="bg-teal-500 text-black text-[7px] font-bold px-1 rounded-sm leading-tight">{dCount}</div>}
                    {fpCount > 0 && <div className="bg-red-500 text-white text-[7px] font-bold px-1 rounded-sm leading-tight">✗{fpCount}</div>}
                    {mCount  > 0 && <div className="bg-yellow-500 text-black text-[7px] font-bold px-1 rounded-sm leading-tight">+{mCount}</div>}
                    {img.status === "error"      && (
                      <div
                        className="bg-red-700 text-white text-[7px] font-bold px-1 rounded-sm cursor-pointer hover:bg-red-500 transition-colors"
                        title="Click to retry"
                        onClick={e => { e.stopPropagation(); retryImage(img.id); }}>
                        ↺ ERR
                      </div>
                    )}
                    {img.status === "processing" && <div className="bg-yellow-600 text-black text-[7px] font-bold px-1 rounded-sm">…</div>}
                    {img.status === "pending"    && <div className="bg-zinc-600 text-white text-[7px] font-bold px-1 rounded-sm">—</div>}
                  </div>
                  <div className="absolute bottom-0.5 right-1 text-[8px] font-mono font-bold text-white/80">
                    {String(reelIdx + 1).padStart(3, "0")}
                  </div>
                  <button className="absolute top-0.5 right-0.5 bg-red-600/80 rounded p-0.5 opacity-0 hover:opacity-100 transition-opacity"
                    onClick={e => { e.stopPropagation(); removeImage(img.id); }}>
                    <X className="w-2.5 h-2.5 text-white" />
                  </button>
                </div>
              );
            })}
          </div>
        </footer>
      )}
    </div>
  );
}
