import express, { Request, Response } from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import multer from "multer";
import fs from "fs";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const MODEL_SIZE_LIMIT  = 500 * 1024 * 1024; // 500 MB
const IMAGE_SIZE_LIMIT  =  50 * 1024 * 1024; //  50 MB

async function startServer() {
  const app  = express();
  const PORT = 3000;

  // ── Image upload storage ────────────────────────────────────────────────────
  const imageStorage = multer.diskStorage({
    destination: (req, file, cb) => {
      const dir = path.join(__dirname, "uploads");
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      cb(null, dir);
    },
    filename: (req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`),
  });

  // ── Model upload storage ────────────────────────────────────────────────────
  const modelStorage = multer.diskStorage({
    destination: (req, file, cb) => {
      const dir = path.join(process.cwd(), "models");
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      cb(null, dir);
    },
    // Keep the original filename so the UI can display a friendly name
    filename: (req, file, cb) => cb(null, file.originalname),
  });

  const uploadImages = multer({ storage: imageStorage, limits: { fileSize: IMAGE_SIZE_LIMIT } });
  // FIX: 500 MB cap on model uploads to prevent disk exhaustion
  const uploadModel  = multer({ storage: modelStorage, limits: { fileSize: MODEL_SIZE_LIMIT } });

  app.use(express.json({ limit: "50mb" }));

  // ── Routes ──────────────────────────────────────────────────────────────────

  // Batch image upload (unused by the frontend but kept for future use)
  app.post("/api/upload", uploadImages.array("files"), (req: Request, res: Response) => {
    const files = req.files as Express.Multer.File[];
    res.json({
      success: true,
      files: files.map(f => ({ name: f.originalname, path: `/uploads/${f.filename}`, type: f.mimetype })),
    });
  });

  // Model upload
  app.post("/api/upload-model", uploadModel.single("model"), (req: Request, res: Response) => {
    if (!req.file) {
      return res.status(400).json({ error: "No model file uploaded" });
    }
    res.json({
      success: true,
      model: {
        name: req.file.originalname,
        path: req.file.path,
        type: "Custom YOLO",
      },
    });
  });

  // Local YOLO inference — proxied to the Python backend
  app.post("/api/detect-local", async (req: Request, res: Response) => {
    try {
      const { image, model_path, confidence } = req.body;
      const response = await fetch("http://localhost:5000/detect", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ image, model_path, confidence }),
      });
      if (!response.ok) {
        const text = await response.text().catch(() => "");
        let errMsg = "Detection failed";
        try { errMsg = JSON.parse(text)?.error ?? text ?? "Detection failed"; } catch { errMsg = text || `HTTP ${response.status}`; }
        console.error(`Python backend error ${response.status}:`, errMsg);
        return res.status(response.status).json({ error: errMsg });
      }
      res.json(await response.json());
    } catch (error: any) {
      console.error("Local detection proxy error:", error?.message);
      res.status(503).json({ error: "Python backend not reachable. Make sure 'python app.py' is running." });
    }
  });

  // Model management — delete and rename
  app.post("/api/model/delete", async (req: Request, res: Response) => {
    try {
      const r = await fetch("http://localhost:5000/model/delete", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      });
      res.status(r.status).json(await r.json());
    } catch { res.status(503).json({ error: "Python backend not reachable" }); }
  });

  app.post("/api/model/rename", async (req: Request, res: Response) => {
    try {
      const r = await fetch("http://localhost:5000/model/rename", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      });
      res.status(r.status).json(await r.json());
    } catch { res.status(503).json({ error: "Python backend not reachable" }); }
  });

  // List .pt / .onnx files in ./models/
  app.get("/api/local-models", (req: Request, res: Response) => {
    const dir = path.join(process.cwd(), "models");
    if (!fs.existsSync(dir)) return res.json({ models: [] });
    try {
      const files = fs.readdirSync(dir).filter(f => f.endsWith(".pt") || f.endsWith(".onnx"));
      res.json({ models: files.map(f => ({ name: f, path: path.join(dir, f) })) });
    } catch (err) {
      console.error("Error reading models dir:", err);
      res.status(500).json({ error: "Failed to read models directory" });
    }
  });

  // Download a model from a URL and save it to the models/ folder
  app.post("/api/download-model", async (req: Request, res: Response) => {
    const { url } = req.body;
    if (!url || typeof url !== "string") return res.status(400).json({ error: "Missing url" });

    // Basic URL validation
    let parsed: URL;
    try { parsed = new URL(url); } catch { return res.status(400).json({ error: "Invalid URL" }); }
    if (!["http:", "https:"].includes(parsed.protocol)) return res.status(400).json({ error: "Only http/https URLs allowed" });

    // Derive a safe filename from the URL
    const rawName = path.basename(parsed.pathname) || "model";
    const safeName = rawName.replace(/[^a-zA-Z0-9._-]/g, "_");
    if (!safeName.endsWith(".pt") && !safeName.endsWith(".onnx")) {
      return res.status(400).json({ error: "URL must point to a .pt or .onnx file" });
    }

    const dir = path.join(process.cwd(), "models");
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    const dest = path.join(dir, safeName);

    try {
      console.log(`Downloading model from ${url} → ${dest}`);
      const response = await fetch(url);
      if (!response.ok) return res.status(502).json({ error: `Download failed: HTTP ${response.status}` });

      const contentLength = response.headers.get("content-length");
      const sizeMB = contentLength ? (parseInt(contentLength) / 1024 / 1024).toFixed(1) : "?";
      console.log(`Model size: ${sizeMB} MB`);

      const buffer = await response.arrayBuffer();
      fs.writeFileSync(dest, Buffer.from(buffer));
      console.log(`Model saved: ${dest}`);

      res.json({ success: true, model: { name: safeName, path: dest } });
    } catch (err: any) {
      console.error("Model download error:", err?.message);
      res.status(500).json({ error: `Download failed: ${err?.message ?? "Unknown error"}` });
    }
  });

  // ── Training proxy routes ─────────────────────────────────────────────────

  // Upload a dataset zip to Python for extraction
  // Write dataset zip directly to disk — avoids holding large files in memory
  const datasetDiskStorage = multer.diskStorage({
    destination: (_req, _file, cb) => {
      const dir = path.join(process.cwd(), "datasets", "_incoming");
      fs.mkdirSync(dir, { recursive: true });
      cb(null, dir);
    },
    filename: (_req, file, cb) => cb(null, `upload_${Date.now()}_${file.originalname}`),
  });
  const datasetUpload = multer({ storage: datasetDiskStorage, limits: { fileSize: 10 * 1024 * 1024 * 1024 } });
  app.post("/api/train/upload-dataset", datasetUpload.single("file"), async (req: Request, res: Response) => {
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });
    const zipPath = req.file.path;
    try {
      // Tell Python the path of the zip on disk — no re-upload needed
      const r = await fetch("http://localhost:5000/train/upload-dataset-path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ zip_path: zipPath, filename: req.file.originalname }),
      });
      const result = await r.json();
      // Clean up zip after Python extracts it
      try { fs.unlinkSync(zipPath); } catch {}
      res.status(r.status).json(result);
    } catch (err: any) {
      try { fs.unlinkSync(zipPath); } catch {}
      res.status(503).json({ error: "Python backend not reachable. Make sure python app.py is running." });
    }
  });

  // Proxy all other /api/train/* routes directly to Python
  for (const [method, paths] of [
    ["GET",  ["/api/train/status", "/api/train/datasets", "/api/train/progress"]],
    ["POST", ["/api/train/start", "/api/train/cancel"]],
  ] as const) {
    for (const p of paths) {
      const pythonPath = p.replace("/api", "");
      if (method === "GET") {
        app.get(p, async (req: Request, res: Response) => {
          try {
            const r = await fetch(`http://localhost:5000${pythonPath}`);
            // SSE passthrough
            if (r.headers.get("content-type")?.includes("text/event-stream")) {
              res.setHeader("Content-Type", "text/event-stream");
              res.setHeader("Cache-Control", "no-cache");
              res.setHeader("X-Accel-Buffering", "no");
              const reader = r.body?.getReader();
              if (!reader) return res.end();
              const pump = async () => {
                while (true) {
                  const { done, value } = await reader.read();
                  if (done) { res.end(); break; }
                  res.write(value);
                }
              };
              pump().catch(() => res.end());
            } else {
              res.status(r.status).json(await r.json());
            }
          } catch { res.status(503).json({ error: "Python backend not reachable" }); }
        });
      } else {
        app.post(p, async (req: Request, res: Response) => {
          try {
            const r = await fetch(`http://localhost:5000${pythonPath}`, {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify(req.body),
            });
            res.status(r.status).json(await r.json());
          } catch { res.status(503).json({ error: "Python backend not reachable" }); }
        });
      }
    }
  }

  // Read a local file as base64 — used by export to avoid Python dependency
  app.get("/api/read-file", (req: Request, res: Response) => {
    const filePath = req.query.path as string;
    if (!filePath) return res.status(400).json({ error: "Missing path" });
    const ext = path.extname(filePath).toLowerCase();
    const mimeMap: Record<string, string> = {
      ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
      ".png": "image/png", ".webp": "image/webp",
    };
    if (!mimeMap[ext]) return res.status(400).json({ error: "Not an image" });
    if (!fs.existsSync(filePath)) return res.status(404).json({ error: "File not found" });
    try {
      const b64 = fs.readFileSync(filePath).toString("base64");
      res.json({ base64: b64, mime: mimeMap[ext] });
    } catch (err: any) {
      res.status(500).json({ error: err?.message ?? "Read failed" });
    }
  });

  // Proxy screenshot scraping (SSE stream)
  app.post("/api/screenshot-scrape", async (req: Request, res: Response) => {
    try {
      const r = await fetch("http://localhost:5000/screenshot-scrape", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      });
      if (!r.ok) { res.status(r.status).json(await r.json()); return; }
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("X-Accel-Buffering", "no");
      const reader = r.body?.getReader();
      if (!reader) return res.end();
      const pump = async () => { while (true) { const { done, value } = await reader.read(); if (done) { res.end(); break; } res.write(value); } };
      pump().catch(() => res.end());
    } catch { res.status(503).json({ error: "Python backend not reachable" }); }
  });

  // Proxy YouTube frame extraction (SSE stream)
  app.post("/api/youtube-frames", async (req: Request, res: Response) => {
    try {
      const r = await fetch("http://localhost:5000/youtube-frames", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      });
      if (!r.ok) { res.status(r.status).json(await r.json()); return; }
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("X-Accel-Buffering", "no");
      const reader = r.body?.getReader();
      if (!reader) return res.end();
      const pump = async () => {
        while (true) {
          const { done, value } = await reader.read();
          if (done) { res.end(); break; }
          res.write(value);
        }
      };
      pump().catch(() => res.end());
    } catch { res.status(503).json({ error: "Python backend not reachable" }); }
  });

  // Proxy image scraping (SSE stream)
  app.post("/api/scrape-images", async (req: Request, res: Response) => {
    try {
      const r = await fetch("http://localhost:5000/scrape-images", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      });
      if (!r.ok) { res.status(r.status).json(await r.json()); return; }
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("X-Accel-Buffering", "no");
      const reader = r.body?.getReader();
      if (!reader) return res.end();
      const pump = async () => {
        while (true) {
          const { done, value } = await reader.read();
          if (done) { res.end(); break; }
          res.write(value);
        }
      };
      pump().catch(() => res.end());
    } catch { res.status(503).json({ error: "Python backend not reachable" }); }
  });

  // Proxy folder scanning and local image serving to Python
  app.post("/api/scan-folder", async (req: Request, res: Response) => {
    try {
      const r = await fetch("http://localhost:5000/scan-folder", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      });
      res.status(r.status).json(await r.json());
    } catch { res.status(503).json({ error: "Python backend not reachable" }); }
  });

  app.get("/api/image", async (req: Request, res: Response) => {
    try {
      const imgPath = req.query.path as string;
      const r = await fetch(`http://localhost:5000/image?path=${encodeURIComponent(imgPath)}`);
      if (!r.ok) return res.status(r.status).json(await r.json());
      const buf = await r.arrayBuffer();
      const ct = r.headers.get("content-type") || "image/jpeg";
      res.setHeader("Content-Type", ct);
      res.setHeader("Cache-Control", "public, max-age=3600");
      res.send(Buffer.from(buf));
    } catch { res.status(503).json({ error: "Python backend not reachable" }); }
  });

  // Serve uploaded images
  app.use("/uploads", express.static(path.join(__dirname, "uploads")));

  // ── Dev / Prod ───────────────────────────────────────────────────────────────
  if (process.env.NODE_ENV !== "production") {
    // Vite MUST be registered before express.static(cwd). If Express runs first it
    // serves .tsx files as application/octet-stream which browsers reject as modules.
    const vite = await createViteServer({ server: { middlewareMode: true }, appType: "spa" });
    app.use(vite.middlewares);
    // .bat launcher files — served after Vite so source files are never intercepted
    app.use(express.static(process.cwd(), { dotfiles: "ignore", index: false }));
  } else {
    const dist = path.join(process.cwd(), "dist");
    app.use(express.static(dist));
    app.get("*", (_req, res) => res.sendFile(path.join(dist, "index.html")));
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`\n🌊 Reef AI running → http://localhost:${PORT}\n`);
  });
}

startServer();