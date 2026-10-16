import { useRef, useState } from "react";
import type { Track, TrackPoint } from "./generated/track";
import type { WeatherPacket } from "./generated/weather";
import type { TimingReport } from "./generated/timing";
import {
  boundedJson,
  readTiming,
  readTrack,
  readWeather,
  utcMillis,
} from "./supportEvidence";

const number = (value: number | null | undefined, unit: string) =>
  value == null ? "Not measured" : `${value.toFixed(2)} ${unit}`;
const measurementLabels: Record<string, string> = {
  heading_deg: "hull heading",
  wind_from_deg: "wind FROM direction",
  wind_speed_m_s: "true wind speed",
  current_east_m_s: "eastward current",
  current_north_m_s: "northward current",
  speed_water_m_s: "through-water speed",
  device_speed_over_ground_m_s: "device ground speed",
};
function TrackMap({ track, selected }: { track: Track; selected: number }) {
  const accepted = track.points.filter((point) => point.accepted);
  if (!accepted.length)
    return <p>No accepted GPS fixes; excluded rows remain below.</p>;
  const xs = accepted.map((p) => p.east_m!),
    ys = accepted.map((p) => p.north_m!);
  const minX = Math.min(...xs),
    minY = Math.min(...ys);
  const scale =
    300 / Math.max(1, Math.max(...xs) - minX, Math.max(...ys) - minY);
  const xy = (point: TrackPoint) => [
    30 + (point.east_m! - minX) * scale,
    330 - (point.north_m! - minY) * scale,
  ];
  const segments = new Map<number, TrackPoint[]>();
  for (const point of accepted) {
    const group = segments.get(point.segment!) ?? [];
    group.push(point);
    segments.set(point.segment!, group);
  }
  const point = track.points[selected];
  return (
    <svg
      viewBox="0 0 380 370"
      role="img"
      aria-label="Observed GPS trajectory, disconnected at gaps and excluded fixes"
    >
      <rect width="380" height="370" fill="#eef3ed" />
      {[...segments].map(([segment, group]) => (
        <polyline
          key={segment}
          data-testid="observed-segment"
          points={group.map((p) => xy(p).join(",")).join(" ")}
          fill="none"
          stroke="#17776e"
          strokeWidth="2.5"
        />
      ))}
      {point?.accepted && (
        <circle cx={xy(point)[0]} cy={xy(point)[1]} r="5" fill="#af663f" />
      )}
      <text x="20" y="20" fill="#173f46">
        N ↑ · East → · Local metres
      </text>
      <text x="20" y="356" fill="#173f46">
        Each path ends at a quality break
      </text>
    </svg>
  );
}

function WeatherView({ weather }: { weather: WeatherPacket }) {
  const [decision, setDecision] = useState(
    weather.available_utc ?? weather.points[0]?.valid_utc ?? "",
  );
  let time = NaN,
    error = "";
  try {
    time = utcMillis(decision);
  } catch {
    error = "Enter an ISO timestamp with Z or a UTC offset.";
  }
  const context = weather.request.kind !== "single_run";
  const available =
    !context &&
    weather.available_utc != null &&
    time >= utcMillis(weather.available_utc);
  const initial = weather.request.run_initialisation_utc;
  const points = available
    ? weather.points.filter(
        (p) =>
          utcMillis(p.valid_utc) >= time &&
          (!initial || utcMillis(p.valid_utc) >= utcMillis(initial)) &&
          !p.flags?.length &&
          p.wind_speed_m_s != null &&
          p.wind_from_deg != null,
      )
    : context
      ? weather.points
      : [];
  return (
    <article
      className="support-weather"
      aria-labelledby="weather-support-title"
    >
      <h3 id="weather-support-title">Weather availability</h3>
      <p>
        {weather.source_evidence === "synthetic_fixture"
          ? "Synthetic API fixture"
          : "Downloaded broad weather context"}{" "}
        · {weather.request.kind} · {weather.request.model}
      </p>
      <dl className="support-metadata">
        <dt>Initialisation</dt>
        <dd>{initial ?? "Context product; no individual run"}</dd>
        <dt>Publication</dt>
        <dd>
          {weather.available_utc ?? "Not a decision-time forecast"} (
          {weather.availability})
        </dd>
        <dt>Retrieved</dt>
        <dd>{weather.retrieved_utc}</dd>
        <dt>Model version</dt>
        <dd>{weather.request.model_version ?? "Not supplied by API"}</dd>
        <dt>Requested / returned latitude</dt>
        <dd>
          {weather.request.latitude_deg} / {weather.returned_latitude_deg}°
        </dd>
        <dt>Requested / returned longitude</dt>
        <dd>
          {weather.request.longitude_deg} / {weather.returned_longitude_deg}°
        </dd>
        <dt>Declared resolution</dt>
        <dd>
          {weather.request.declared_spatial_resolution_m / 1000} km /{" "}
          {(weather.request.declared_temporal_resolution_s ?? 3600) / 3600} h
        </dd>
      </dl>
      <label>
        Historical decision UTC
        <input
          aria-label="Historical decision UTC"
          value={decision}
          onChange={(e) => setDecision(e.target.value)}
        />
      </label>
      {error ? (
        <p role="alert">{error}</p>
      ) : (
        <p role="status">
          {context
            ? "Context only; unavailable to a historical policy."
            : available
              ? "Run available under the declared publication evidence."
              : "Run not yet available at this decision."}
        </p>
      )}
      {weather.availability === "assumed_lag" && (
        <p>
          Availability is conditional on an assumed lag; it is not proof of
          publication.
        </p>
      )}
      <div className="table-scroll">
        <table>
          <caption>
            {context ? "Retrospective context" : "Available forecast values"}
          </caption>
          <thead>
            <tr>
              <th>Valid UTC</th>
              <th>Wind speed</th>
              <th>Wind FROM</th>
              <th>Quality</th>
            </tr>
          </thead>
          <tbody>
            {points.slice(0, 24).map((p) => (
              <tr key={p.valid_utc}>
                <td>{p.valid_utc}</td>
                <td>{number(p.wind_speed_m_s, "m/s")}</td>
                <td>{number(p.wind_from_deg, "°")}</td>
                <td>{p.flags?.join(", ") || "Valid"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        {weather.limits} {weather.acknowledgement}
      </p>
    </article>
  );
}

export function MeasurementView() {
  const [track, setTrack] = useState<Track | null>(null);
  const [weather, setWeather] = useState<WeatherPacket | null>(null);
  const [timing, setTiming] = useState<TimingReport | null>(null);
  const [selected, setSelected] = useState(0),
    [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const request = useRef(0);
  const loopback = ["127.0.0.1", "localhost", "[::1]"].includes(
    window.location.hostname,
  );
  async function loadFixture() {
    const id = ++request.current;
    setBusy(true);
    setError("");
    try {
      const [t, w, c] = await Promise.all(
        ["track", "weather", "timing"].map(async (name) =>
          boundedJson(await fetch(`/support/${name}.json`)),
        ),
      );
      const nextTrack = readTrack(t),
        nextWeather = readWeather(w),
        nextTiming = readTiming(c);
      if (!nextTiming.source_hashes.includes(nextTrack.source_sha256))
        throw new Error(
          "Displayed track is not a source of this timing fixture report.",
        );
      if (
        nextTrack.evidence !== "synthetic_fixture" ||
        nextWeather.source_evidence !== "synthetic_fixture" ||
        nextTiming.evidence !== "synthetic_fixture"
      )
        throw new Error(
          "Demo support files must be labelled synthetic fixtures.",
        );
      if (id === request.current) {
        setTrack(nextTrack);
        setWeather(nextWeather);
        setTiming(nextTiming);
        setSelected(0);
      }
    } catch (reason) {
      if (id === request.current)
        setError(
          reason instanceof Error ? reason.message : "Fixture load failed.",
        );
    } finally {
      if (id === request.current) setBusy(false);
    }
  }
  async function importFile(file: File | undefined, kind: "track" | "weather") {
    if (!file) return;
    const id = ++request.current;
    setBusy(true);
    setError("");
    try {
      if (file.size > 5_000_000) throw new Error("File exceeds 5 MB.");
      let data: unknown;
      if (file.name.toLowerCase().endsWith(".json"))
        data = JSON.parse(await file.text()) as unknown;
      else if (kind === "track" && /\.(csv|gpx)$/i.test(file.name)) {
        if (!loopback)
          throw new Error(
            "Raw tracks require the private local importer. Process locally, then open its JSON here.",
          );
        const format = file.name.toLowerCase().endsWith(".gpx") ? "gpx" : "csv";
        let response: Response;
        try {
          response = await fetch(
            `http://127.0.0.1:8787/tracks/import?format=${format}`,
            { method: "POST", body: file },
          );
        } catch {
          throw new Error(
            "Start sailing-support serve locally on port 8787, then import again.",
          );
        }
        if (!response.ok)
          throw new Error(
            `Local importer rejected this file (${response.status}). Check its UTC/SI format.`,
          );
        const result = (await response.json()) as { track: unknown };
        data = result.track;
      } else
        throw new Error("Choose a processed JSON packet or a CSV/GPX track.");
      if (kind === "track") {
        const next = readTrack(data);
        if (id === request.current) {
          setTrack(next);
          setTiming(null);
          setSelected(0);
        }
      } else {
        const next = readWeather(data);
        if (id === request.current) setWeather(next);
      }
    } catch (reason) {
      if (id === request.current)
        setError(reason instanceof Error ? reason.message : "Import failed.");
    } finally {
      if (id === request.current) setBusy(false);
    }
  }
  function download() {
    if (!track) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(track, null, 2)], { type: "application/json" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "private-track.json";
    anchor.click();
    URL.revokeObjectURL(url);
  }
  const point = track?.points[selected];
  const flagged = track?.points.filter((p) => p.flags?.length) ?? [];
  return (
    <section
      className="workspace support-section"
      aria-labelledby="measurement-title"
    >
      <div className="section-heading">
        <div>
          <p className="eyebrow">05 / MEASUREMENT SUPPORT</p>
          <h2 id="measurement-title">From a track to defensible evidence</h2>
        </div>
        <span className="chip">Local and private</span>
      </div>
      <p>
        Inspect observed GPS and instruments, retained quality flags, and
        forecast publication times. Track import does not publish your file.
        Timing fixtures do not calibrate a real boat.
      </p>
      <div className="support-controls">
        <button onClick={() => void loadFixture()} disabled={busy}>
          Load synthetic support fixture
        </button>
        <label>
          Open track
          <input
            type="file"
            aria-label="Open track"
            disabled={busy}
            accept=".csv,.gpx,.json"
            onChange={(e) => void importFile(e.target.files?.[0], "track")}
          />
        </label>
        <label>
          Open processed weather
          <input
            type="file"
            aria-label="Open processed weather"
            disabled={busy}
            accept=".json"
            onChange={(e) => void importFile(e.target.files?.[0], "weather")}
          />
        </label>
      </div>
      <p>
        CSV/GPX uses <code>sailing-support serve</code> on your computer.
        Processed JSON stays in this browser. The public application has no
        research or cloud job submission.
      </p>
      {busy && <p role="status">Validating support evidence…</p>}
      {error && <p role="alert">{error}</p>}
      {track && (
        <>
          <h3>
            {track.evidence === "synthetic_fixture"
              ? "Synthetic track fixture"
              : "Imported track · unvalidated measurement"}
          </h3>
          <p>
            {track.points.filter((p) => p.accepted).length} accepted /{" "}
            {track.points.filter((p) => !p.accepted).length} excluded fixes ·{" "}
            {
              new Set(
                track.points.filter((p) => p.accepted).map((p) => p.segment),
              ).size
            }{" "}
            connected segments
          </p>
          <div className="support-layout">
            <div>
              <TrackMap track={track} selected={selected} />
              <label>
                Source row
                <input
                  type="range"
                  aria-label="Source row"
                  min="0"
                  max={Math.max(0, track.points.length - 1)}
                  value={selected}
                  onChange={(e) => setSelected(Number(e.target.value))}
                />
              </label>
            </div>
            <article>
              <h3>Observed row {point?.row}</h3>
              <p>
                {point?.timestamp_utc ?? "Invalid clock"} ·{" "}
                {point?.accepted ? "Accepted" : "Excluded"}
              </p>
              <dl className="support-metadata">
                <dt>Heading (hull)</dt>
                <dd>{number(point?.heading_deg, "°")}</dd>
                <dt>Wind FROM</dt>
                <dd>{number(point?.wind_from_deg, "°")}</dd>
                <dt>Wind speed</dt>
                <dd>{number(point?.wind_speed_m_s, "m/s")}</dd>
                <dt>Instrumented through-water speed</dt>
                <dd>{number(point?.speed_water_m_s, "m/s")}</dd>
                <dt>GPS-derived ground speed</dt>
                <dd>
                  {point?.derived_speed_over_ground_m_s == null
                    ? "No continuous speed estimate"
                    : number(point.derived_speed_over_ground_m_s, "m/s")}
                </dd>
                <dt>Device ground speed</dt>
                <dd>{number(point?.device_speed_over_ground_m_s, "m/s")}</dd>
                <dt>Current east / north</dt>
                <dd>
                  {number(point?.current_east_m_s, "m/s")} /{" "}
                  {number(point?.current_north_m_s, "m/s")}
                </dd>
                <dt>Quality</dt>
                <dd>{point?.flags?.join(", ") || "Valid"}</dd>
              </dl>
              <p>
                Measured:{" "}
                {track.measured
                  .map((name) => measurementLabels[name] ?? name)
                  .join(", ")}
                .
              </p>
              <p>
                Derived: {track.derived?.join(", ")}. Assumed:{" "}
                {track.assumed?.join(", ")}.
              </p>
              <p className="support-hash">
                Raw source SHA-256: <code>{track.source_sha256}</code>
              </p>
              <button onClick={download}>
                Download processed private track
              </button>
            </article>
          </div>
          <div className="table-scroll">
            <table>
              <caption>
                Retained quality flags · {flagged.length} rows (first 100 shown)
              </caption>
              <thead>
                <tr>
                  <th>Row</th>
                  <th>UTC</th>
                  <th>Included</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {flagged.slice(0, 100).map((p) => (
                  <tr key={p.row}>
                    <td>{p.row}</td>
                    <td>{p.timestamp_utc ?? "Invalid"}</td>
                    <td>{p.accepted ? "Yes" : "No"}</td>
                    <td>{p.flags?.join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {timing && (
        <article>
          <h3>Held-out synthetic timing</h3>
          <p>
            {timing.training_days.length} training UTC days /{" "}
            {timing.holdout_days.length} held-out days ·{" "}
            {timing.transitions.length} instrument-supported transitions. Fit:{" "}
            {timing.fitted_mean_s.toFixed(1)} s; held-out absolute error:{" "}
            {timing.holdout_mae_s.toFixed(1)} s (day-bootstrap 95%:{" "}
            {timing.holdout_mae_day_bootstrap_95_s
              .map((v) => v.toFixed(1))
              .join("–")}{" "}
            s).
          </p>
          <p>
            Fitted quantity: {timing.quantity}. Unsupported:{" "}
            {timing.unsupported?.join(", ")}. {timing.limitations}
          </p>
        </article>
      )}
      {weather && <WeatherView key={weather.source_sha256} weather={weather} />}
    </section>
  );
}
