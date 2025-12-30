import { useEffect, useMemo, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";
import {
  fetchLegend,
  runInference,
  type LegendEntry,
  type StatsRow
} from "@/lib/api";
import { Download, LocateFixed, Play, Sparkles } from "lucide-react";
import maplibregl from "maplibre-gl";
import osmtogeojson from "osmtogeojson";

const TILE_LIMIT = 9;
const MIN_AGRI_ZOOM = 8;
const defaultCenter = { lat: 30.84, lng: 31.02 };
const MAX_ZOOM = 17;
const METERS_PER_PIXEL = 10;
const CHIP_SIZE_PX = 256;
const EARTH_RADIUS_M = 6378137;
const OVERLAY_SOURCE_ID = "analysis-overlay";
const OVERLAY_LAYER_ID = "analysis-overlay-layer";

function formatPercent(value: number) {
  return `${value.toFixed(1)}%`;
}

type ViewBounds = {
  minLat: number;
  minLon: number;
  maxLat: number;
  maxLon: number;
};

export default function App() {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const farmlandFetchTimeout = useRef<number | null>(null);
  const farmlandAbort = useRef<AbortController | null>(null);
  const [zoom, setZoom] = useState(6);
  const [opacity, setOpacity] = useState(65);
  const [legend, setLegend] = useState<LegendEntry[]>([]);
  const [stats, setStats] = useState<StatsRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [academicView, setAcademicView] = useState(false);
  const [runtimeMs, setRuntimeMs] = useState(0);
  const [overlayImage, setOverlayImage] = useState<string | null>(null);
  const [maskImage, setMaskImage] = useState<string | null>(null);
  const [exporting, setExporting] = useState<"png" | "mask" | "csv" | null>(null);
  const [agriStatus, setAgriStatus] = useState<"idle" | "loading" | "ready" | "error" | "zoom">(
    "idle"
  );
  const [center, setCenter] = useState(defaultCenter);
  const [tileCount, setTileCount] = useState(1);
  const [viewportBounds, setViewportBounds] = useState<ViewBounds | null>(null);
  const [overlayBounds, setOverlayBounds] = useState<ViewBounds | null>(null);

  const activeOverlayImage =
    opacity === 100 ? maskImage ?? overlayImage : overlayImage;

  const dominantCrop = useMemo(() => {
    if (!stats.length) {
      return "No results yet";
    }
    const sorted = [...stats].sort((a, b) => b.percent - a.percent);
    return `${sorted[0].name} (${formatPercent(sorted[0].percent)})`;
  }, [stats]);

  useEffect(() => {
    fetchLegend()
      .then(setLegend)
      .catch(() => {
        setLegend([]);
      });
  }, []);

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) {
      return;
    }

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: {
        version: 8,
        sources: {
          satellite: {
            type: "raster",
            tiles: [
              "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            ],
            tileSize: 256,
            attribution:
              "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
          },
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors"
          }
        },
        layers: [
          {
            id: "satellite-base",
            type: "raster",
            source: "satellite"
          },
          {
            id: "osm-overlay",
            type: "raster",
            source: "osm",
            paint: {
              "raster-opacity": 0.25,
              "raster-contrast": 0.05,
              "raster-saturation": -0.1
            }
          }
        ]
      },
      center: [defaultCenter.lng, defaultCenter.lat],
      zoom,
      maxZoom: MAX_ZOOM
    });

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");

    const updateBaseBlend = (currentZoom: number) => {
      const minZoom = 6;
      const maxZoom = 12;
      const t = Math.min(1, Math.max(0, (currentZoom - minZoom) / (maxZoom - minZoom)));
      const satOpacity = 0.35 + 0.65 * t;
      const osmOpacity = 0.75 - 0.55 * t;
      if (map.getLayer("satellite-base")) {
        map.setPaintProperty("satellite-base", "raster-opacity", satOpacity);
      }
      if (map.getLayer("osm-overlay")) {
        map.setPaintProperty("osm-overlay", "raster-opacity", osmOpacity);
      }
    };

    map.on("load", () => {
      map.dragPan.enable();
      map.addSource("farmland", {
        type: "geojson",
        data: {
          type: "FeatureCollection",
          features: []
        }
      });
      map.addLayer({
        id: "farmland-fill",
        type: "fill",
        source: "farmland",
        paint: {
          "fill-color": "#6fbf73",
          "fill-opacity": 0
        }
      });
      updateBaseBlend(map.getZoom());
      updateFromMap();
    });

    const toMercatorX = (lon: number) => (lon * Math.PI * EARTH_RADIUS_M) / 180;
    const toMercatorY = (lat: number) => {
      const clamped = Math.max(-85.05112878, Math.min(85.05112878, lat));
      const rad = (clamped * Math.PI) / 180;
      return EARTH_RADIUS_M * Math.log(Math.tan(Math.PI / 4 + rad / 2));
    };

    const updateFromMap = () => {
      const nextZoom = Math.round(map.getZoom());
      setZoom(nextZoom);
      const nextCenter = map.getCenter();
      setCenter({ lat: nextCenter.lat, lng: nextCenter.lng });
      const bounds = map.getBounds();
      const west = bounds.getWest();
      const east = bounds.getEast();
      const south = bounds.getSouth();
      const north = bounds.getNorth();
      setViewportBounds({
        minLat: south,
        minLon: west,
        maxLat: north,
        maxLon: east
      });
      const widthMeters = Math.abs(toMercatorX(east) - toMercatorX(west));
      const heightMeters = Math.abs(toMercatorY(north) - toMercatorY(south));
      const chipMeters = METERS_PER_PIXEL * CHIP_SIZE_PX;
      const tilesX = Math.max(1, Math.ceil(widthMeters / chipMeters));
      const tilesY = Math.max(1, Math.ceil(heightMeters / chipMeters));
      setTileCount(tilesX * tilesY);
      requestFarmland(map, nextZoom);
    };

    map.on("zoom", () => {
      updateBaseBlend(map.getZoom());
    });
    map.on("moveend", updateFromMap);
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }
    if (Math.round(map.getZoom()) !== zoom) {
      map.setZoom(zoom);
    }
  }, [zoom]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }
    if (!activeOverlayImage || !overlayBounds) {
      if (map.getLayer(OVERLAY_LAYER_ID)) {
        map.removeLayer(OVERLAY_LAYER_ID);
      }
      if (map.getSource(OVERLAY_SOURCE_ID)) {
        map.removeSource(OVERLAY_SOURCE_ID);
      }
      return;
    }
    const applyOverlay = () => {
      const coordinates: [number, number][] = [
        [overlayBounds.minLon, overlayBounds.maxLat],
        [overlayBounds.maxLon, overlayBounds.maxLat],
        [overlayBounds.maxLon, overlayBounds.minLat],
        [overlayBounds.minLon, overlayBounds.minLat]
      ];
      const source = map.getSource(OVERLAY_SOURCE_ID) as maplibregl.ImageSource | undefined;
      if (source) {
        source.updateImage({ url: activeOverlayImage, coordinates });
      } else {
        map.addSource(OVERLAY_SOURCE_ID, {
          type: "image",
          url: activeOverlayImage,
          coordinates
        });
        map.addLayer({
          id: OVERLAY_LAYER_ID,
          type: "raster",
          source: OVERLAY_SOURCE_ID,
          paint: {
            "raster-opacity": opacity / 100
          }
        });
      }
    };
    if (!map.isStyleLoaded()) {
      map.once("load", applyOverlay);
      return;
    }
    applyOverlay();
  }, [activeOverlayImage, overlayBounds]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }
    if (map.getLayer(OVERLAY_LAYER_ID)) {
      map.setPaintProperty(OVERLAY_LAYER_ID, "raster-opacity", opacity / 100);
    }
  }, [opacity]);

  const requestFarmland = (map: maplibregl.Map, zoomLevel: number) => {
    if (zoomLevel < MIN_AGRI_ZOOM) {
      setAgriStatus("zoom");
      const source = map.getSource("farmland") as maplibregl.GeoJSONSource | undefined;
      source?.setData({ type: "FeatureCollection", features: [] });
      return;
    }

    const bounds = map.getBounds();
    const south = bounds.getSouth();
    const west = bounds.getWest();
    const north = bounds.getNorth();
    const east = bounds.getEast();
    if (Math.abs(east - west) > 6 || Math.abs(north - south) > 6) {
      setAgriStatus("zoom");
      return;
    }

    if (farmlandFetchTimeout.current) {
      window.clearTimeout(farmlandFetchTimeout.current);
    }

    farmlandFetchTimeout.current = window.setTimeout(async () => {
      if (farmlandAbort.current) {
        farmlandAbort.current.abort();
      }
      const controller = new AbortController();
      farmlandAbort.current = controller;
      setAgriStatus("loading");
      try {
        const query = `[out:json][timeout:25];
(
  way["landuse"~"farmland|farm|orchard|vineyard"](${south},${west},${north},${east});
  relation["landuse"~"farmland|farm|orchard|vineyard"](${south},${west},${north},${east});
);
out geom;`;
        const response = await fetch("https://overpass-api.de/api/interpreter", {
          method: "POST",
          body: query,
          signal: controller.signal
        });
        if (!response.ok) {
          throw new Error("Overpass failed.");
        }
        const data = await response.json();
        const geojson = osmtogeojson(data) as GeoJSON.FeatureCollection;
        const source = map.getSource("farmland") as maplibregl.GeoJSONSource | undefined;
        source?.setData(geojson);
        setAgriStatus("ready");
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          setAgriStatus("error");
        }
      }
    }, 650);
  };

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    const requestBounds =
      viewportBounds ??
      (() => {
        const map = mapRef.current;
        if (!map) {
          return null;
        }
        const bounds = map.getBounds();
        return {
          minLat: bounds.getSouth(),
          minLon: bounds.getWest(),
          maxLat: bounds.getNorth(),
          maxLon: bounds.getEast()
        };
      })();
    try {
      const response = await runInference({
        viewport: {
          center,
          zoom,
          tileCount,
          bounds: requestBounds ?? undefined
        },
        options: {
          includeConfidence: true
        }
      });
      setLegend(response.legend);
      setStats(response.stats);
      setRuntimeMs(response.meta.runtimeMs);
      setOverlayImage(response.overlayImage ?? null);
      setMaskImage(response.maskImage ?? null);
      const resolvedBounds = response.meta.overlayBounds ?? requestBounds ?? null;
      if (resolvedBounds) {
        setOverlayBounds(resolvedBounds);
      }
    } catch (err) {
      setError("Run Analysis failed. Check backend status.");
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async (format: "png" | "mask" | "csv") => {
    setExporting(format);
    setError(null);
    try {
      if (format === "png" || format === "mask") {
        const imageData = format === "mask" ? maskImage : overlayImage;
        if (!imageData) {
          throw new Error(
            format === "mask"
              ? "No mask-only image available."
              : "No overlay image available."
          );
        }
        const base64 = imageData.split(",")[1];
        if (!base64) {
          throw new Error("Overlay image is invalid.");
        }
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) {
          bytes[i] = binary.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: "image/png" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download =
          format === "mask"
            ? "agrovision-mask-only.png"
            : "agrovision-overlay.png";
        anchor.click();
        URL.revokeObjectURL(url);
        return;
      }

      if (stats.length === 0) {
        throw new Error("No statistics available.");
      }
      const headers = ["id", "name", "pixels", "percent", "confidence"];
      const rows = stats.map((row) =>
        [
          row.id,
          `"${row.name.replaceAll('"', '""')}"`,
          row.pixels,
          row.percent.toFixed(3),
          row.confidence.toFixed(3)
        ].join(",")
      );
      const csv = [headers.join(","), ...rows].join("\n");
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "agrovision-stats.csv";
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Export failed. Try again.";
      setError(message);
    } finally {
      setExporting(null);
    }
  };

  const overLimit = tileCount > TILE_LIMIT;

  return (
    <div className="noise min-h-screen px-4 py-8 text-mist md:px-10">
      <div className="mx-auto flex max-w-6xl flex-col gap-8">
        <header className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
          <div className="space-y-3">
            <Badge>AgroVision Crop Mapper</Badge>
            <h1 className="font-display text-4xl font-semibold text-white md:text-5xl">
              Interactive Crop Mapping
            </h1>
            <p className="max-w-xl text-sm text-mist/70 md:text-base">
              Map-first interface for rapid crop type overlays, confidence-aware stats,
              and export-ready summaries.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="rounded-full border border-white/15 bg-white/10 px-4 py-2 text-xs uppercase tracking-[0.25em] text-mist/70">
              Demo region: Nile Delta
            </div>
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
          <Card className="relative overflow-hidden">
            <CardHeader>
              <CardTitle>AOI Preview</CardTitle>
              <CardDescription>
                Pan/zoom to define a viewport. Overlay appears after analysis.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative h-[420px] overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-moss/40 via-ink/40 to-ink/90">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(240,194,90,0.22),_transparent_50%)]" />
                <div ref={mapContainerRef} className="absolute inset-0" />
                <div className="pointer-events-none absolute inset-10 rounded-2xl border border-white/70 bg-white/0 shadow-[0_0_0_1px_rgba(255,255,255,0.25),0_0_25px_rgba(240,194,90,0.15)]" />
              </div>
            </CardContent>
            <CardFooter className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-3">
                <span className="text-xs uppercase tracking-[0.3em] text-mist/60">
                  Overlay opacity
                </span>
                <Slider
                  value={[opacity]}
                  onValueChange={(value) => setOpacity(value[0])}
                  max={100}
                  min={0}
                  step={1}
                  className="w-40"
                />
              </div>
              <span className="text-xs text-mist/70">{opacity}%</span>
              <div className="flex items-center gap-3">
                <LocateFixed className="h-4 w-4 text-canary" />
                <span className="text-xs uppercase tracking-[0.3em] text-mist/60">
                  coordinates
                </span>
                <span className="text-xs text-mist/70">
                  {center.lat.toFixed(4)}, {center.lng.toFixed(4)}
                </span>
              </div>
            </CardFooter>
          </Card>

          <div className="flex flex-col gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Run Analysis</CardTitle>
                <CardDescription>
                  Zoom in until the viewport is under the {TILE_LIMIT}-tile limit.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-mist/70">
                    <span>Zoom</span>
                    <span>Level {zoom}</span>
                  </div>
                  <Slider
                    value={[zoom]}
                    onValueChange={(value) => setZoom(value[0])}
                    min={3}
                    max={MAX_ZOOM}
                    step={1}
                  />
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-mist/80">
                  Tile estimate:{" "}
                  <span className={overLimit ? "text-red-300" : "text-canary"}>
                    {tileCount} tiles
                  </span>
                  {overLimit && (
                    <span className="ml-2 text-xs text-red-200">
                      Zoom in to continue.
                    </span>
                  )}
                </div>
                <Button
                  onClick={handleRun}
                  disabled={loading || overLimit}
                  className="w-full"
                >
                  <Play className="h-4 w-4" />
                  {loading ? "Running analysis..." : "Run Analysis"}
                </Button>
                {error && (
                  <div className="rounded-2xl border border-red-300/40 bg-red-500/10 px-4 py-3 text-sm text-red-100">
                    {error}
                  </div>
                )}
              </CardContent>
              <CardFooter className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Switch
                    checked={academicView}
                    onCheckedChange={(value) => setAcademicView(value)}
                  />
                  <span className="text-xs uppercase tracking-[0.2em] text-mist/60">
                    Academic view
                  </span>
                </div>
                {runtimeMs > 0 && (
                  <span className="text-xs text-mist/60">
                    {runtimeMs} ms inference
                  </span>
                )}
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Legend</CardTitle>
                <CardDescription>Crop classes and palette.</CardDescription>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-3 text-sm">
                {legend.length === 0 ? (
                  <p className="col-span-2 text-mist/60">
                    Legend loads from backend.
                  </p>
                ) : (
                  legend.map((item) => (
                    <div key={item.id} className="flex items-center gap-3">
                      <span
                        className="h-3 w-3 rounded-full"
                        style={{ backgroundColor: item.color }}
                      />
                      <span>{item.name}</span>
                    </div>
                  ))
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
          <Card>
            <CardHeader>
              <CardTitle>Area Statistics</CardTitle>
              <CardDescription>Per-class coverage and mean confidence.</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Class</TableHead>
                    <TableHead>Pixels</TableHead>
                    <TableHead>%</TableHead>
                    <TableHead>Confidence</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {stats.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={4} className="text-mist/60">
                        Run analysis to populate stats.
                      </TableCell>
                    </TableRow>
                  ) : (
                    stats.map((row) => (
                      <TableRow key={row.id}>
                        <TableCell>{row.name}</TableCell>
                        <TableCell>{row.pixels.toLocaleString()}</TableCell>
                        <TableCell>{formatPercent(row.percent)}</TableCell>
                        <TableCell>{row.confidence.toFixed(2)}</TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </CardContent>
            <CardFooter className="justify-between">
              <span className="text-xs uppercase tracking-[0.2em] text-mist/60">
                Dominant crop
              </span>
              <span className="text-sm text-white">{dominantCrop}</span>
            </CardFooter>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Exports</CardTitle>
              <CardDescription>Download overlay PNG or CSV summary.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                variant="outline"
                className="w-full justify-between"
                onClick={() => handleExport("png")}
                disabled={exporting !== null}
              >
                <span>Download overlay PNG</span>
                <Download className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                className="w-full justify-between"
                onClick={() => handleExport("mask")}
                disabled={exporting !== null}
              >
                <span>Download mask-only PNG</span>
                <Download className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                className="w-full justify-between"
                onClick={() => handleExport("csv")}
                disabled={exporting !== null}
              >
                <span>Download CSV summary</span>
                <Download className="h-4 w-4" />
              </Button>
              {academicView && (
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-xs text-mist/70">
                  <p>RGB composite: Sentinel-2 true color (B4, B3, B2).</p>
                  <p>Tile count: {tileCount} (max {TILE_LIMIT}).</p>
                  <p>Normalization: min-max per band, cached in config.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
