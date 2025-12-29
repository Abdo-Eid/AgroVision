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
    overlayBounds?: {
      minLat: number;
      minLon: number;
      maxLat: number;
      maxLon: number;
    };
  };
};

const apiBase = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function fetchLegend(): Promise<LegendEntry[]> {
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
      overlayBounds: meta.overlayBounds ?? data?.overlay_bounds ?? undefined
    }
  };
}
