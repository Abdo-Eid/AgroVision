export type LegendEntry = {
  id: number;
  name: string;
  color: string;
};

export type StatsRow = {
  id: number;
  name: string;
  pixels: number;
  percent: number;
  confidence: number;
};

export type InferResponse = {
  overlayImage?: string | null;
  stats: StatsRow[];
  legend: LegendEntry[];
  meta: {
    runtimeMs: number;
    isMock: boolean;
  };
};

const apiBase = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
export const mockEnabled = import.meta.env.VITE_USE_MOCK === "1";

const mockLegend: LegendEntry[] = [
  { id: 0, name: "Background", color: "#23323a" },
  { id: 1, name: "Wheat", color: "#e4c15f" },
  { id: 2, name: "Rice", color: "#7bd4c0" },
  { id: 3, name: "Maize", color: "#e36b5d" },
  { id: 4, name: "Sugarcane", color: "#8ab07d" },
  { id: 5, name: "Potato", color: "#d79a6a" },
  { id: 6, name: "Berseem", color: "#72b37e" }
];

const mockStats: StatsRow[] = [
  { id: 1, name: "Wheat", pixels: 24211, percent: 34.2, confidence: 0.82 },
  { id: 2, name: "Rice", pixels: 14433, percent: 20.4, confidence: 0.74 },
  { id: 3, name: "Maize", pixels: 11201, percent: 15.8, confidence: 0.71 },
  { id: 4, name: "Sugarcane", pixels: 8088, percent: 11.4, confidence: 0.69 },
  { id: 5, name: "Potato", pixels: 6551, percent: 9.2, confidence: 0.77 },
  { id: 6, name: "Berseem", pixels: 5958, percent: 8.4, confidence: 0.73 }
];

export async function fetchLegend(): Promise<LegendEntry[]> {
  if (mockEnabled) {
    return mockLegend;
  }
  const response = await fetch(`${apiBase}/api/legend`);
  if (!response.ok) {
    throw new Error("Legend request failed.");
  }
  const data = (await response.json()) as LegendEntry[];
  return data;
}

type InferPayload = {
  viewport: {
    center: { lat: number; lng: number };
    zoom: number;
    tileCount: number;
    bounds?: {
      minLat: number;
      minLon: number;
      maxLat: number;
      maxLon: number;
    };
  };
  options: {
    includeConfidence: boolean;
  };
};

export async function runInference(payload: InferPayload): Promise<InferResponse> {
  if (mockEnabled) {
    return {
      overlayImage: null,
      stats: mockStats,
      legend: mockLegend,
      meta: {
        runtimeMs: 2360,
        isMock: true
      }
    };
  }
  const response = await fetch(`${apiBase}/api/infer`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const message = data?.message ?? "Inference request failed.";
    throw new Error(message);
  }
  const overlayImage = data?.overlayImage ?? data?.overlay_image ?? null;
  const meta = data?.meta ?? {};
  return {
    overlayImage,
    stats: data?.stats ?? data?.stats_table ?? [],
    legend: data?.legend ?? [],
    meta: {
      runtimeMs: meta.runtimeMs ?? data?.runtime_ms ?? 0,
      isMock: meta.isMock ?? false
    }
  };
}
