import { GoogleGenAI, Type } from "@google/genai";
import { Detection } from "../types";

// SECURITY: always construct the client from the provided key — never from a
// module-level constant that might have been baked in at build time.
function makeClient(apiKey: string) {
  return new GoogleGenAI({ apiKey });
}

function getColorForLabel(label: string): string {
  const lower = label.toLowerCase();
  if (lower.includes("cot") || lower.includes("starfish")) return "#ef4444";
  if (lower.includes("clam"))                               return "#3b82f6";
  if (lower.includes("bleach"))                             return "#f97316";
  return "#10b981";
}

export async function detectObjects(
  base64Image: string,
  customApiKey?: string
): Promise<Detection[]> {
  const apiKey = customApiKey;
  if (!apiKey || apiKey.trim() === "" || apiKey.includes("TODO")) {
    throw new Error("MISSING_API_KEY");
  }

  const ai = makeClient(apiKey);

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: [
        {
          parts: [
            {
              // FIX: use named object fields instead of a positional array so
              // coordinate order is unambiguous regardless of model behaviour.
              text: `Detect Crown-of-Thorns Starfish (COTS), coral bleaching, and giant clams in this image.
Return a JSON array. Each element must have:
  "label"      — string, the detected class name
  "confidence" — number 0–1
  "bbox"       — object with keys xmin, ymin, xmax, ymax, each a number 0–1000 (normalised to image size)`,
            },
            {
              inlineData: {
                mimeType: "image/jpeg",
                data: base64Image.split(",")[1] || base64Image,
              },
            },
          ],
        },
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              label:      { type: Type.STRING },
              confidence: { type: Type.NUMBER },
              bbox: {
                type: Type.OBJECT,
                properties: {
                  xmin: { type: Type.NUMBER },
                  ymin: { type: Type.NUMBER },
                  xmax: { type: Type.NUMBER },
                  ymax: { type: Type.NUMBER },
                },
                required: ["xmin", "ymin", "xmax", "ymax"],
              },
            },
            required: ["label", "confidence", "bbox"],
          },
        },
      },
    });

    const results = JSON.parse(response.text || "[]");
    return results.map((r: any, index: number) => ({
      id:    `gemini-${Date.now()}-${index}`,
      label: r.label,
      confidence: r.confidence,
      // Named fields — no positional ambiguity
      bbox: [r.bbox.xmin, r.bbox.ymin, r.bbox.xmax, r.bbox.ymax] as [number, number, number, number],
      color: getColorForLabel(r.label),
    }));
  } catch (error) {
    console.error("Gemini detection error:", error);
    if ((error as Error).message === "MISSING_API_KEY") throw error;
    return [];
  }
}
