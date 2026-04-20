export interface Detection {
  id: string;
  label: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] in normalized coordinates (0-1000)
  color?: string;
  isFalsePositive?: boolean;
}

export interface Annotation {
  id: string;
  label: string;
  bbox: [number, number, number, number];
  type: 'missed_target' | 'false_positive';
  color?: string;
}

export interface ImageData {
  id: string;
  name: string;
  url: string;
  detections: Detection[];
  annotations: Annotation[];
  status: 'pending' | 'processing' | 'completed' | 'error';
  isAnnotated?: boolean;
}

export interface VideoFrameCapture {
  id: string;
  timestamp: number;
  thumbnailUrl: string;
  fullImageUrl: string;
  annotations: Annotation[];
}
