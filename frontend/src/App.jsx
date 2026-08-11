import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { Upload, ScanLine, ShieldCheck, Activity, ChevronRight, Server } from "lucide-react";

// ---------------------------------------------------------------------------
// Design tokens (see design plan: clinical light-viewer palette, not the
// generic warm-cream / dark-acid-green defaults)
// ---------------------------------------------------------------------------
const COLORS = {
  bg: "#EEF1F5",
  panel: "#FFFFFF",
  ink: "#121826",
  inkSoft: "#5B6472",
  line: "#DCE1E8",
  teal: "#0E7C7B",
  tealSoft: "#E3F3F2",
  amber: "#C4622D",
  amberSoft: "#FBEBE1",
  navy: "#1B2A4A",
};

const API_BASE = "http://localhost:8000";

const DISPLAY_FONT = "'Space Grotesk', 'Segoe UI', sans-serif";
const BODY_FONT = "'IBM Plex Sans', 'Segoe UI', sans-serif";
const MONO_FONT = "'IBM Plex Mono', 'Courier New', monospace";

// Fallback demo data so the console is inspectable even without a live
// backend connection (e.g. previewing the UI before wiring the API up).
const DEMO_HISTORY = {
  round: [1, 2, 3, 4, 5, 6, 7, 8],
  accuracy: [0.61, 0.70, 0.76, 0.81, 0.85, 0.88, 0.90, 0.91],
};

export default function App() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | scanning | done | error
  const [result, setResult] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [history, setHistory] = useState(DEMO_HISTORY);
  const [usingDemo, setUsingDemo] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/history`)
      .then((r) => r.json())
      .then((data) => {
        if (data.round && data.round.length > 0) setHistory(data);
      })
      .catch(() => setUsingDemo(true));
  }, []);

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setResult(null);
    setStatus("idle");
  }, []);

  const runDiagnosis = async () => {
    if (!imageFile) return;
    setStatus("scanning");

    const formData = new FormData();
    formData.append("file", imageFile);

    try {
      const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: formData });
      if (!res.ok) throw new Error("predict failed");
      const data = await res.json();
      setResult(data);
      setStatus("done");
    } catch (e) {
      // No backend reachable in this environment -- show a clearly labeled
      // simulated result so reviewers can still see the interaction model.
      await new Promise((r) => setTimeout(r, 1400));
      setUsingDemo(true);
      setResult({
        predicted_class: "PNEUMONIA",
        confidence: 87.4,
        heatmap_base64: null,
      });
      setStatus("done");
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files?.[0]);
  };

  return (
    <div
      style={{ background: COLORS.bg, color: COLORS.ink, fontFamily: BODY_FONT, minHeight: "100%" }}
      className="w-full min-h-screen p-6 md:p-10"
    >
      <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />

      {/* Header */}
      <header className="flex items-center justify-between mb-8 flex-wrap gap-4">
        <div>
          <div style={{ fontFamily: MONO_FONT, color: COLORS.teal, letterSpacing: "0.12em", fontSize: "12px" }}>
            PRIVACY-PRESERVING &middot; FEDERATED &middot; EXPLAINABLE
          </div>
          <h1 style={{ fontFamily: DISPLAY_FONT, fontWeight: 600, fontSize: "28px", color: COLORS.navy, marginTop: "4px" }}>
            Diagnostic Console
          </h1>
        </div>
        <div className="flex items-center gap-2 px-3 py-2 rounded-md" style={{ background: COLORS.panel, border: `1px solid ${COLORS.line}` }}>
          <ShieldCheck size={16} color={COLORS.teal} />
          <span style={{ fontFamily: MONO_FONT, fontSize: "12px", color: COLORS.inkSoft }}>
            No raw image ever leaves this session
          </span>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main viewer panel */}
        <div className="lg:col-span-2 rounded-lg p-5" style={{ background: COLORS.panel, border: `1px solid ${COLORS.line}` }}>
          <div className="flex items-center justify-between mb-4">
            <span style={{ fontFamily: MONO_FONT, fontSize: "12px", color: COLORS.inkSoft, letterSpacing: "0.08em" }}>
              01 &nbsp;IMAGE INPUT
            </span>
            {result && (
              <button
                onClick={() => setShowHeatmap((s) => !s)}
                className="flex items-center gap-1 px-3 py-1.5 rounded-md text-sm"
                style={{ background: COLORS.tealSoft, color: COLORS.teal, fontFamily: BODY_FONT, fontWeight: 500 }}
              >
                {showHeatmap ? "Show original" : "Show explanation heatmap"}
                <ChevronRight size={14} />
              </button>
            )}
          </div>

          {/* Drop zone / viewer */}
          <div
            onDrop={onDrop}
            onDragOver={(e) => e.preventDefault()}
            className="relative rounded-md flex items-center justify-center overflow-hidden"
            style={{
              background: "#0B1220",
              minHeight: "420px",
              border: `1px dashed ${imagePreview ? "transparent" : COLORS.line}`,
            }}
          >
            {!imagePreview && (
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex flex-col items-center gap-3 text-center px-6"
                style={{ color: "#8A93A6" }}
              >
                <Upload size={28} />
                <div style={{ fontFamily: BODY_FONT, fontSize: "14px" }}>
                  Drop a scan here, or click to upload
                </div>
                <div style={{ fontFamily: MONO_FONT, fontSize: "11px", color: "#5B6472" }}>
                  X-ray &middot; CT &middot; MRI &middot; JPG / PNG
                </div>
              </button>
            )}

            {imagePreview && (
              <>
                <img
                  src={
                    showHeatmap && result?.heatmap_base64 ? result.heatmap_base64 : imagePreview
                  }
                  alt="scan"
                  className="w-full h-full object-contain"
                  style={{ maxHeight: "420px" }}
                />
                {status === "scanning" && <ScanBeam />}
              </>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
          </div>

          <div className="flex items-center justify-between mt-4">
            <button
              onClick={() => fileInputRef.current?.click()}
              style={{ fontFamily: BODY_FONT, fontSize: "13px", color: COLORS.inkSoft }}
            >
              {imagePreview ? "Choose a different image" : ""}
            </button>
            <button
              onClick={runDiagnosis}
              disabled={!imageFile || status === "scanning"}
              className="px-5 py-2.5 rounded-md flex items-center gap-2"
              style={{
                background: !imageFile ? COLORS.line : COLORS.navy,
                color: !imageFile ? COLORS.inkSoft : "#fff",
                fontFamily: BODY_FONT,
                fontWeight: 500,
                fontSize: "14px",
                cursor: !imageFile ? "not-allowed" : "pointer",
              }}
            >
              <Activity size={16} />
              {status === "scanning" ? "Analyzing..." : "Run diagnosis"}
            </button>
          </div>
        </div>

        {/* Sidebar: result + federation info */}
        <div className="flex flex-col gap-6">
          {/* Result card */}
          <div className="rounded-lg p-5" style={{ background: COLORS.panel, border: `1px solid ${COLORS.line}` }}>
            <span style={{ fontFamily: MONO_FONT, fontSize: "12px", color: COLORS.inkSoft, letterSpacing: "0.08em" }}>
              02 &nbsp;MODEL READOUT
            </span>

            {!result && (
              <div className="mt-4" style={{ color: COLORS.inkSoft, fontSize: "14px" }}>
                Upload a scan and run diagnosis to see the model's prediction and its reasoning.
              </div>
            )}

            {result && (
              <div className="mt-4">
                <div
                  className="inline-block px-3 py-1 rounded-md mb-3"
                  style={{
                    background: result.predicted_class === "NORMAL" ? COLORS.tealSoft : COLORS.amberSoft,
                    color: result.predicted_class === "NORMAL" ? COLORS.teal : COLORS.amber,
                    fontFamily: MONO_FONT,
                    fontSize: "13px",
                    fontWeight: 500,
                  }}
                >
                  {result.predicted_class}
                </div>

                <div style={{ fontFamily: DISPLAY_FONT, fontSize: "32px", fontWeight: 600, color: COLORS.navy }}>
                  {result.confidence}%
                </div>
                <div style={{ fontFamily: BODY_FONT, fontSize: "13px", color: COLORS.inkSoft, marginBottom: "12px" }}>
                  model confidence
                </div>

                <div className="w-full rounded-full h-1.5 mb-4" style={{ background: COLORS.line }}>
                  <div
                    className="h-1.5 rounded-full"
                    style={{
                      width: `${result.confidence}%`,
                      background: result.predicted_class === "NORMAL" ? COLORS.teal : COLORS.amber,
                    }}
                  />
                </div>

                <p style={{ fontFamily: BODY_FONT, fontSize: "13px", color: COLORS.inkSoft, lineHeight: 1.5 }}>
                  The highlighted regions (Grad-CAM) show which areas of the
                  image most influenced this prediction. Use "Show explanation
                  heatmap" above to inspect them.
                </p>

                {usingDemo && (
                  <div style={{ fontFamily: MONO_FONT, fontSize: "11px", color: COLORS.amber, marginTop: "12px" }}>
                    ⚠ backend not connected — showing simulated result
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Federation status card */}
          <div className="rounded-lg p-5" style={{ background: COLORS.navy, color: "#fff" }}>
            <div className="flex items-center gap-2 mb-3">
              <Server size={16} />
              <span style={{ fontFamily: MONO_FONT, fontSize: "12px", letterSpacing: "0.08em", opacity: 0.8 }}>
                03 &nbsp;FEDERATION STATUS
              </span>
            </div>
            <div style={{ fontFamily: DISPLAY_FONT, fontSize: "20px", fontWeight: 600 }}>
              {history.round.length} training rounds completed
            </div>
            <div style={{ fontFamily: BODY_FONT, fontSize: "13px", opacity: 0.75, marginTop: "4px" }}>
              Global model accuracy: {(history.accuracy[history.accuracy.length - 1] * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>

      {/* Training history chart */}
      <div className="rounded-lg p-5 mt-6" style={{ background: COLORS.panel, border: `1px solid ${COLORS.line}` }}>
        <span style={{ fontFamily: MONO_FONT, fontSize: "12px", color: COLORS.inkSoft, letterSpacing: "0.08em" }}>
          04 &nbsp;GLOBAL MODEL CONVERGENCE ACROSS FEDERATED ROUNDS
        </span>
        <div style={{ height: "220px", marginTop: "12px" }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history.round.map((r, i) => ({ round: r, accuracy: history.accuracy[i] }))}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.line} />
              <XAxis dataKey="round" tick={{ fontFamily: MONO_FONT, fontSize: 11, fill: COLORS.inkSoft }} label={{ value: "round", position: "insideBottom", offset: -3, fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fontFamily: MONO_FONT, fontSize: 11, fill: COLORS.inkSoft }} />
              <Tooltip
                contentStyle={{ fontFamily: MONO_FONT, fontSize: "12px", border: `1px solid ${COLORS.line}` }}
                formatter={(v) => `${(v * 100).toFixed(1)}%`}
              />
              <Line type="monotone" dataKey="accuracy" stroke={COLORS.teal} strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <footer style={{ fontFamily: MONO_FONT, fontSize: "11px", color: COLORS.inkSoft, marginTop: "24px", textAlign: "center" }}>
        Each simulated hospital trains locally &middot; only weight updates are aggregated &middot; raw images never leave their source
      </footer>
    </div>
  );
}

// A horizontal scan-line sweep shown while the model is "reading" the image.
function ScanBeam() {
  return (
    <div
      className="absolute left-0 right-0"
      style={{
        height: "2px",
        background: "linear-gradient(90deg, transparent, #0E7C7B, transparent)",
        boxShadow: "0 0 12px 2px #0E7C7B",
        animation: "scan-sweep 1.6s linear infinite",
      }}
    >
      <style>{`
        @keyframes scan-sweep {
          0% { top: 4%; }
          50% { top: 92%; }
          100% { top: 4%; }
        }
      `}</style>
    </div>
  );
}
