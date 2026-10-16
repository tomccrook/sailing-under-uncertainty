import { useEffect, useState } from "react";
import { Course } from "./Course";
import { DecisionInspector } from "./DecisionInspector";
import {
  degrees,
  indexAt,
  loadCatalogue,
  loadRecording,
  timeLabel,
} from "./evidence";
import type { LoadedRecording } from "./evidence";
import type { Catalogue } from "./generated/catalogue";

export function PairedReplay({
  onAvailable,
}: {
  onAvailable: (available: boolean) => void;
}) {
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null);
  const [scenarioId, setScenarioId] = useState("");
  const [variant, setVariant] = useState("B3_risk");
  const [comparisonVariant, setComparisonVariant] = useState("B1_belief");
  const [loaded, setLoaded] = useState<{
    primary: LoadedRecording;
    comparison: LoadedRecording;
  } | null>(null);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(20);
  const [truth, setTruth] = useState(false);
  useEffect(() => {
    const abort = new AbortController();
    loadCatalogue(abort.signal)
      .then((value) => {
        if (!value || abort.signal.aborted) return;
        setCatalogue(value);
        setScenarioId(value.default_scenario_id);
        onAvailable(true);
      })
      .catch((reason: unknown) => {
        if (!abort.signal.aborted)
          setError(
            reason instanceof Error ? reason.message : "Catalogue unavailable.",
          );
      });
    return () => abort.abort();
  }, [onAvailable]);
  const scenario = catalogue?.scenarios.find(
    (s) => s.scenario_id === scenarioId,
  );
  const selected = scenario?.recordings.find((r) => r.variant === variant);
  const comparison = scenario?.recordings.find(
    (r) => r.variant === comparisonVariant,
  );
  useEffect(() => {
    if (!scenario || !selected || !comparison) return;
    const abort = new AbortController();
    Promise.all([
      loadRecording(selected, scenario, abort.signal),
      loadRecording(comparison, scenario, abort.signal),
    ])
      .then(([primary, other]) => {
        if (!abort.signal.aborted) setLoaded({ primary, comparison: other });
      })
      .catch((reason: unknown) => {
        if (!abort.signal.aborted)
          setError(
            reason instanceof Error ? reason.message : "Replay unavailable.",
          );
      });
    return () => abort.abort();
  }, [scenario, selected, comparison]);
  const current =
    loaded?.primary.recording.run_id === selected?.run_id &&
    loaded?.comparison.recording.run_id === comparison?.run_id
      ? loaded
      : null;
  const duration = Math.max(
    selected?.elapsed_s ?? 0,
    comparison?.elapsed_s ?? 0,
  );
  useEffect(() => {
    if (!playing || !current) return;
    let previous: number | undefined,
      frame = 0;
    const tick = (now: number) => {
      if (previous !== undefined)
        setElapsed((v) =>
          Math.min(duration, v + ((now - previous!) * rate) / 1000),
        );
      previous = now;
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, current, duration, rate]);
  useEffect(() => {
    if (playing && elapsed >= duration) setPlaying(false);
  }, [elapsed, duration, playing]);
  const resetSelection = () => {
    setPlaying(false);
    setElapsed(0);
    setError("");
  };
  if (!catalogue) return error ? <p role="status">{error}</p> : null;
  return (
    <section className="workspace" aria-labelledby="paired-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">01 / SAME WORLD, DIFFERENT DECISIONS</p>
          <h2 id="paired-title">Compare recorded policies</h2>
        </div>
        <span className="chip">Locked · synthetic diagnostics</span>
      </div>
      <div className="comparison-controls">
        <label>
          Recorded world
          <select
            value={scenarioId}
            onChange={(e) => {
              resetSelection();
              setScenarioId(e.target.value);
            }}
          >
            {catalogue.scenarios.map((s) => (
              <option key={s.scenario_id} value={s.scenario_id}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Inspect policy
          <select
            value={variant}
            onChange={(e) => {
              resetSelection();
              setVariant(e.target.value);
            }}
          >
            {scenario?.recordings.map((r) => (
              <option key={r.variant} value={r.variant}>
                {r.variant.replaceAll("_", " ")}
              </option>
            ))}
          </select>
        </label>
        <label>
          Compare with
          <select
            value={comparisonVariant}
            onChange={(e) => {
              resetSelection();
              setComparisonVariant(e.target.value);
            }}
          >
            {scenario?.recordings.map((r) => (
              <option key={r.variant} value={r.variant}>
                {r.variant.replaceAll("_", " ")}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="fineprint">
        {scenario?.scenario_id} · {scenario?.slice.replaceAll("_", " ")} ·{" "}
        {scenario?.selection_reason}. Selected after evaluation for diagnosis;
        aggregate results below determine performance claims.
      </p>
      {error ? (
        <p role="alert">{error}</p>
      ) : !current ? (
        <p role="status">Verifying paired recordings…</p>
      ) : (
        <>
          <div className="replay-layout">
            <div className="map-panel">
              <div className="map-meta">
                <span>
                  Solid / boat: inspected · dashed / diamond: comparison
                </span>
                <span>
                  {truth
                    ? "DIAGNOSTIC TRUTH · recorded offline"
                    : "POLICY INFORMATION · observed local wind"}
                </span>
              </div>
              <Course
                replay={current.primary.replay}
                index={indexAt(current.primary.replay.samples, elapsed)}
                truth={truth}
                comparison={current.comparison.replay}
                elapsed={elapsed}
              />
              <div className="map-footer">
                <span>East / north · metres</span>
                <span>
                  Failed or finished boats hold their terminal position
                </span>
              </div>
            </div>
            <aside className="inspector">
              <p className="eyebrow">SHARED ELAPSED CLOCK</p>
              <h3>At {timeLabel(elapsed)}</h3>
              <dl>
                {[current.primary, current.comparison].map((r, i) => {
                  const s =
                    r.replay.samples[indexAt(r.replay.samples, elapsed)]!;
                  return (
                    <div key={i}>
                      <dt>
                        {i === 0
                          ? "Inspected · solid boat"
                          : "Comparison · dashed diamond"}
                        <br />
                        {r.recording.variant.replaceAll("_", " ")}
                      </dt>
                      <dd>
                        {elapsed >= r.recording.elapsed_s
                          ? r.recording.status.replaceAll("_", " ")
                          : "sailing"}
                        <small>
                          {" "}
                          ·{" "}
                          {s.position_xy_m
                            .map((v) => v.toFixed(0))
                            .join(" / ")}{" "}
                          m · {degrees(s.heading_rad)}°
                        </small>
                      </dd>
                    </div>
                  );
                })}
              </dl>
              <label className="truth-toggle">
                <input
                  type="checkbox"
                  checked={truth}
                  onChange={(e) => setTruth(e.target.checked)}
                />{" "}
                Show diagnostic truth
              </label>
              <p className="fineprint">
                The toggle changes the displayed recorded wind vector. It never
                changes an action or episode. Candidate scores always use
                available policy information.
              </p>
            </aside>
          </div>
          <div className="transport">
            <button
              className="play-button"
              onClick={() => {
                if (elapsed >= duration) setElapsed(0);
                setPlaying((p) => !p);
              }}
            >
              {playing ? "Pause" : "Play"}
            </button>
            <button className="reset-button" onClick={resetSelection}>
              Reset
            </button>
            <label className="scrubber">
              Replay time
              <input
                aria-label="Replay time"
                type="range"
                min="0"
                max={duration}
                step="0.1"
                value={elapsed}
                onChange={(e) => setElapsed(Number(e.target.value))}
              />
            </label>
            <output aria-label="Elapsed replay time">
              {timeLabel(elapsed)} / {timeLabel(duration)}
            </output>
            <label className="speed-label">
              Speed
              <select
                aria-label="Playback speed"
                value={rate}
                onChange={(e) => setRate(Number(e.target.value))}
              >
                {[1, 5, 20, 60].map((r) => (
                  <option key={r} value={r}>
                    {r}×
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="table-scroll">
            <table>
              <caption>
                Recorded outcomes on this one world; failures score 2400 s,
                elapsed failure time is not a finish.
              </caption>
              <thead>
                <tr>
                  <th>Policy</th>
                  <th>Outcome</th>
                  <th>Elapsed</th>
                  <th>Cost</th>
                  <th>Distance</th>
                  <th>Tacks</th>
                </tr>
              </thead>
              <tbody>
                {[current.primary, current.comparison].map((r, i) => (
                  <tr key={i}>
                    <th scope="row">
                      {r.recording.variant.replaceAll("_", " ")}
                    </th>
                    <td>{r.recording.status.replaceAll("_", " ")}</td>
                    <td>{r.recording.elapsed_s.toFixed(2)} s</td>
                    <td>{r.recording.cost_s.toFixed(2)} s</td>
                    <td>{r.replay.summary.distance_m.toFixed(1)} m</td>
                    <td>{r.replay.summary.tack_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <DecisionInspector
            details={current.primary.details}
            elapsed={Math.min(elapsed, current.primary.recording.elapsed_s)}
          />
          <div className="provenance">
            <p>
              World <code>{scenario?.world_hash.slice(0, 12)}</code> · paired
              configuration <code>{scenario?.pairing_hash.slice(0, 12)}</code> ·
              inspected run <code>{current.primary.recording.run_id}</code>
            </p>
            <div className="download-links">
              <a href={current.primary.recording.replay.url} download>
                Download recorded replay ↗
              </a>
              <a href={current.primary.recording.details.url} download>
                Decisions &amp; configuration ↗
              </a>
              <a href={current.primary.recording.manifest.url} download>
                Run manifest ↗
              </a>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
