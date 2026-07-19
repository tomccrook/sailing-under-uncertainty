import { useEffect, useState } from "react";
import Ajv from "ajv";
import schema from "../replay.schema.json";
import { Course } from "./Course";
import { PairedReplay } from "./PairedReplay";
import { EvidenceContext } from "./EvidenceContext";
import { Tradeoffs } from "./Tradeoffs";
import { Results } from "./Results";
import { LearningResults } from "./LearningResults";
import { MeasurementView } from "./MeasurementView";
import { InvestigationView } from "./InvestigationView";
import { ReplicationView } from "./ReplicationView";
import { ComputeFollowupView, RecordedValidationView } from "./FollowupView";
import type { ReplayBundle } from "./generated/replay";

const validate = new Ajv({ strict: false }).compile<ReplayBundle>(schema);
const degrees = (angle: number) =>
  (((angle * 180) / Math.PI + 360) % 360).toFixed(1);
const timeLabel = (seconds: number) =>
  `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

export default function App() {
  const [hasCatalogue, setHasCatalogue] = useState(false);
  const [replay, setReplay] = useState<ReplayBundle | null>(null);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(20);
  const [truth, setTruth] = useState(false);
  useEffect(() => {
    const abort = new AbortController();
    fetch("/replay.json", { signal: abort.signal })
      .then(async (response) => {
        if (!response.ok)
          throw new Error(
            "No recorded replay is available. Run uv run sail-lab demo from the repository root.",
          );
        const data: unknown = await response.json();
        if (!validate(data) || data.samples.length === 0)
          throw new Error(
            "Replay is missing or has an unsupported schema. Generate a fresh local demo.",
          );
        setReplay(data);
      })
      .catch((reason: unknown) => {
        if (!abort.signal.aborted)
          setError(
            reason instanceof Error
              ? reason.message
              : "Replay could not be loaded.",
          );
      });
    return () => abort.abort();
  }, []);
  useEffect(() => {
    if (!playing || !replay) return;
    let frame = 0;
    let previous: number | undefined;
    const tick = (timestamp: number) => {
      if (previous !== undefined)
        setElapsed((value) =>
          Math.min(
            replay.summary.elapsed_s,
            value + ((timestamp - previous!) / 1000) * rate,
          ),
        );
      previous = timestamp;
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, rate, replay]);
  useEffect(() => {
    if (replay && elapsed >= replay.summary.elapsed_s) setPlaying(false);
  }, [elapsed, replay]);
  let index = 0;
  if (replay) {
    let lo = 0,
      hi = replay.samples.length - 1;
    while (lo <= hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (replay.samples[mid]!.elapsed_s <= elapsed) {
        index = mid;
        lo = mid + 1;
      } else hi = mid - 1;
    }
  }
  const sample = replay?.samples[index];
  const togglePlay = () => {
    if (replay && elapsed >= replay.summary.elapsed_s) setElapsed(0);
    setPlaying((value) => !value);
  };
  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="wordmark" href="#overview">
          <span className="brand-mark">◬</span> SAIL / LAB
        </a>
        <span className="top-context">Tom Crook · Research notebook</span>
        <span className="status-dot">Research · Recorded evidence</span>
      </header>
      <main id="overview">
        <section className="intro">
          <p className="eyebrow">SEQUENTIAL DECISIONS · WIND · RISK</p>
          <h1>
            Sailing under
            <br />
            <em>uncertainty.</em>
          </h1>
          <p className="lede">
            Does accounting for uncertain wind improve the route — or just add
            complexity?
          </p>
          <p className="intro-note">
            A sailing-tactics study by Tom Crook. Start with a validated model,
            compare strong baselines, and let the evidence decide.
          </p>
          <div className="evidence">
            <span>SYNTHETIC EVIDENCE</span>
            <p>
              In the locked nominal study, selected belief PPO was slower than
              the tuned heuristic. Risk planning showed a conditional benefit
              over deterministic planning; superiority over the strong heuristic
              remains unresolved.{" "}
              <a href="#tradeoffs-title">
                Inspect the evidence and uncertainty ↓
              </a>
            </p>
          </div>
        </section>
        <PairedReplay onAvailable={setHasCatalogue} />
        {!hasCatalogue && (
          <section className="workspace" aria-labelledby="replay-title">
            <div className="section-heading">
              <div>
                <p className="eyebrow">01 / THE RECORDED LEG</p>
                <h2 id="replay-title">A recorded route, step by step</h2>
              </div>
              <span className="chip">
                {replay?.world_family ?? "constant"} wind ·{" "}
                {replay?.stage ?? "fixture"}
              </span>
            </div>
            {error ? (
              <div className="empty-state" role="alert">
                <h3>Replay unavailable</h3>
                <p>{error}</p>
              </div>
            ) : !replay ? (
              <p role="status">Loading recorded evidence…</p>
            ) : (
              <>
                <div className="replay-layout">
                  <div className="map-panel">
                    <div className="map-meta">
                      <span>
                        {(
                          replay.policy_label ??
                          replay.summary.policy_id ??
                          "polar_heuristic"
                        )
                          .replaceAll("_", " ")
                          .toUpperCase()}{" "}
                        <i className="legend-line" />
                      </span>
                      <span>
                        {truth
                          ? "DIAGNOSTIC TRUTH · recorded offline"
                          : `POLICY INFORMATION · ${sample?.observation.information_set ?? "perfect_local_known_current"}`}
                      </span>
                    </div>
                    <Course replay={replay} index={index} truth={truth} />
                    <div className="map-footer">
                      <span>Local east / north frame · metres</span>
                      <span>Dashed line: complete recorded route</span>
                    </div>
                  </div>
                  <aside
                    className="inspector"
                    aria-labelledby="inspector-title"
                  >
                    <p className="eyebrow">RECORDED STATE</p>
                    <h3 id="inspector-title">
                      At {timeLabel(sample?.elapsed_s ?? 0)}
                    </h3>
                    <dl>
                      <div>
                        <dt>Heading clockwise from north</dt>
                        <dd>{degrees(sample?.heading_rad ?? 0)}°</dd>
                      </div>
                      <div>
                        <dt>Speed through water</dt>
                        <dd>
                          {(sample?.speed_water_m_s ?? 0).toFixed(2)}{" "}
                          <small>m/s</small>
                        </dd>
                      </div>
                      <div>
                        <dt>East / north position</dt>
                        <dd>
                          {sample?.position_xy_m
                            .map((v) => v.toFixed(0))
                            .join(" / ")}{" "}
                          <small>m</small>
                        </dd>
                      </div>
                      <div>
                        <dt>Completed tacks</dt>
                        <dd>{sample?.tack_count}</dd>
                      </div>
                      <div>
                        <dt>Manoeuvre</dt>
                        <dd>
                          {sample?.manoeuvre_active
                            ? "Turning · latched"
                            : "Sailing"}
                        </dd>
                      </div>
                      <div>
                        <dt>Wind measurement age</dt>
                        <dd>
                          {(
                            (sample?.elapsed_s ?? 0) -
                            (sample?.observation.measurement_time_s ?? 0)
                          ).toFixed(1)}{" "}
                          <small>s</small>
                        </dd>
                      </div>
                      <div>
                        <dt>Instrument status</dt>
                        <dd>
                          {sample?.observation.measurement_valid
                            ? "Available"
                            : "Missing / stale"}
                          {sample?.observation.latest_sensor_dropout
                            ? " · dropout"
                            : ""}
                        </dd>
                      </div>
                      {sample?.observation.belief && (
                        <div>
                          <dt>Filtered air speed / uncertainty</dt>
                          <dd>
                            {Math.hypot(
                              ...sample.observation.belief.mean_air_m_s,
                            ).toFixed(2)}{" "}
                            m/s
                            <small>
                              {" "}
                              · vector σ{" "}
                              {Math.sqrt(
                                sample.observation.belief
                                  .covariance_m2_s2[0][0] +
                                  sample.observation.belief
                                    .covariance_m2_s2[1][1],
                              ).toFixed(2)}{" "}
                              m/s
                            </small>
                          </dd>
                        </div>
                      )}
                      <div>
                        <dt>Available forecast</dt>
                        <dd>
                          {sample?.observation.forecast
                            ? `Issued ${timeLabel(sample.observation.forecast.issued_s)} · ${sample.observation.forecast.horizon_s}s horizon`
                            : "None delivered"}
                        </dd>
                      </div>
                      {replay.policy_parameters &&
                        replay.policy_parameters.name !== "polar_heuristic" && (
                          <div>
                            <dt>Planning horizon / samples / risk weight</dt>
                            <dd>
                              {replay.policy_parameters.planning_horizon_s ??
                                60}
                              s /{" "}
                              {replay.policy_parameters.name ===
                              "deterministic_mpc"
                                ? 1
                                : (replay.policy_parameters.scenario_count ??
                                  8)}{" "}
                              / {replay.policy_parameters.risk_lambda ?? 0.25}
                            </dd>
                          </div>
                        )}
                      <div>
                        <dt>Requested / executed target</dt>
                        <dd>
                          {degrees(sample?.requested_heading_rad ?? 0)}° /{" "}
                          {degrees(sample?.executed_target_rad ?? 0)}°
                        </dd>
                      </div>
                    </dl>
                    <label className="truth-toggle">
                      <input
                        type="checkbox"
                        checked={truth}
                        onChange={(event) => setTruth(event.target.checked)}
                      />{" "}
                      Show diagnostic truth
                    </label>
                    <p className="fineprint">
                      Synthetic instruments supply wind, position, heading and
                      water speed; uniform current is assumed known. Truth is an
                      offline diagnostic. A missing measurement uses the
                      declared prior.
                    </p>
                  </aside>
                </div>
                <div className="transport">
                  <button className="play-button" onClick={togglePlay}>
                    {playing ? "Pause" : "Play"}
                  </button>
                  <button
                    className="reset-button"
                    onClick={() => {
                      setPlaying(false);
                      setElapsed(0);
                    }}
                  >
                    Reset
                  </button>
                  <label className="scrubber">
                    Replay time
                    <input
                      aria-label="Replay time"
                      type="range"
                      min="0"
                      max={replay.summary.elapsed_s}
                      step="0.1"
                      value={elapsed}
                      onChange={(event) =>
                        setElapsed(Number(event.target.value))
                      }
                    />
                  </label>
                  <output aria-label="Elapsed replay time">
                    {timeLabel(elapsed)} / {timeLabel(replay.summary.elapsed_s)}
                  </output>
                  <label className="speed-label">
                    Speed
                    <select
                      aria-label="Playback speed"
                      value={rate}
                      onChange={(event) => setRate(Number(event.target.value))}
                    >
                      {[1, 5, 20, 60].map((speed) => (
                        <option key={speed} value={speed}>
                          {speed}×
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className="metrics">
                  <div>
                    <span>Recorded outcome</span>
                    <strong className="outcome">
                      {replay.summary.status.replaceAll("_", " ")}
                    </strong>
                  </div>
                  <div>
                    <span>Elapsed time · synthetic</span>
                    <strong>
                      {replay.summary.elapsed_s.toFixed(2)} <small>s</small>
                    </strong>
                  </div>
                  <div>
                    <span>Distance over ground</span>
                    <strong>
                      {replay.summary.distance_m.toFixed(1)} <small>m</small>
                    </strong>
                  </div>
                  <div>
                    <span>Completed tacks</span>
                    <strong>{replay.summary.tack_count}</strong>
                  </div>
                </div>
                <div className="provenance">
                  <p>
                    Run <code>{replay.run_id}</code> · {replay.samples.length}{" "}
                    recorded samples · {replay.summary.decisions} decisions
                  </p>
                  <a href="/replay.json" download="sailing-replay.json">
                    Download recorded replay ↗
                  </a>
                </div>
              </>
            )}
          </section>
        )}
        <Tradeoffs />
        <EvidenceContext />
        <Results />
        <LearningResults />
        <MeasurementView />
        <InvestigationView />
        <ReplicationView />
        <ComputeFollowupView />
        <RecordedValidationView />
        <section className="method-grid">
          <article>
            <p className="eyebrow">THE BASELINE</p>
            <h3>A polar-informed route</h3>
            <p>
              The heuristic uses the illustrative polar’s best upwind velocity,
              target geometry, a commitment rule and finite manoeuvres. Baseline
              parameters are fitted on development worlds; final tests remain
              separate.
            </p>
          </article>
          <article>
            <p className="eyebrow">THE REFERENCE</p>
            <h3>471.405 seconds, ideally</h3>
            <p>
              The separate 45° / 3 m/s analytical fixture has no turn or
              acceleration losses and a point target. It checks geometry; it
              does not predict a real dinghy’s finish time.
            </p>
          </article>
          <article>
            <p className="eyebrow">THE RESEARCH QUESTION</p>
            <h3>What survives uncertainty?</h3>
            <p>
              Frozen stochastic worlds include noisy measurements and forecast
              errors. Locked comparisons retain failures and paired uncertainty.
              PPO has not beaten the tuned heuristic. Separate work and
              new-process studies test planning; strong-baseline comparisons and
              real boat physics retain explicit limits.
            </p>
          </article>
        </section>
        <footer>
          Tom Crook · MSc Computational Finance{" "}
          <span>
            Local, reproducible research · Uncalibrated synthetic model
          </span>
        </footer>
      </main>
    </div>
  );
}
