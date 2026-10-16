import type { Details, Observation } from "./generated/details";
import { degrees, indexAt, timeLabel } from "./evidence";

function WindBelief({ observation }: { observation: Observation }) {
  const belief = observation.belief;
  if (!belief) return null;
  const [[a, b], [, d]] = belief.covariance_m2_s2;
  const delta = Math.hypot(a - d, 2 * b);
  const major = Math.sqrt(Math.max(0, (a + d + delta) / 2) * 5.991);
  const minor = Math.sqrt(Math.max(0, (a + d - delta) / 2) * 5.991);
  const angle = (-Math.atan2(2 * b, a - d) * 90) / Math.PI;
  const raw = observation.wind_air_m_s;
  const mean = belief.mean_air_m_s;
  const extent = Math.max(
    3,
    ...raw.map(Math.abs),
    ...mean.map((v) => Math.abs(v) + major),
  );
  const scale = 90 / extent;
  const xy = ([x, y]: [number, number]) => [150 + x * scale, 120 - y * scale];
  const [cx, cy] = xy(mean);
  return (
    <figure className="belief-figure">
      <svg
        viewBox="0 0 300 245"
        role="img"
        aria-label="Available wind estimate and Gaussian covariance ellipse"
      >
        <path d="M40 120H260M150 20V220" stroke="#ccd9d6" />
        <ellipse
          cx={cx}
          cy={cy}
          rx={major * scale}
          ry={minor * scale}
          transform={`rotate(${angle} ${cx} ${cy})`}
          fill="#14776e18"
          stroke="#14776e"
        />
        <line
          x1="150"
          y1="120"
          x2={cx}
          y2={cy}
          stroke="#14776e"
          strokeWidth="2"
        />
        <circle cx={cx} cy={cy} r="4" fill="#14776e" />
        <path
          d={`M150 120L${xy(raw).join(" ")}`}
          stroke="#a65532"
          strokeWidth="2"
          strokeDasharray="5 3"
        />
        <rect
          x={xy(raw)[0]! - 3}
          y={xy(raw)[1]! - 3}
          width="6"
          height="6"
          fill="#a65532"
        />
        <text x="153" y="17">
          north air velocity
        </text>
        <text x="155" y="238">
          east → · ±{extent.toFixed(1)} m/s
        </text>
      </svg>
      <figcaption>
        Solid / circle: belief. Dashed / square: held instrument. Ellipse:
        nominal 95% Gaussian vector region; empirical coverage is tested
        separately.
      </figcaption>
    </figure>
  );
}

export function DecisionInspector({
  details,
  elapsed,
}: {
  details: Details;
  elapsed: number;
}) {
  const decisionIndex = indexAt(details.decisions, elapsed);
  const decision = details.decisions[decisionIndex]!;
  const observation = decision.observation;
  const planning = decision.planning;
  const probability = details.probabilities?.[decisionIndex];
  const forecast = observation.forecast;
  const points =
    forecast?.points.filter((p) => p.valid_s >= decision.elapsed_s) ?? [];
  return (
    <section className="decision-panel" aria-labelledby="decision-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">THE AVAILABLE INFORMATION</p>
          <h3 id="decision-title">
            Decision at {timeLabel(decision.elapsed_s)}
          </h3>
        </div>
        <span className="chip">
          Recorded · {(decision.compute_s * 1000).toFixed(2)} ms
        </span>
      </div>
      <p className="fineprint">
        Most recent decision at or before the shared replay clock. Displaying{" "}
        {details.variant.replaceAll("_", " ")}; subsequent observations are
        excluded from this snapshot.
      </p>
      <div className="decision-information">
        <dl>
          <div>
            <dt>Wind measurement age / availability</dt>
            <dd>
              {(decision.elapsed_s - observation.measurement_time_s).toFixed(1)}{" "}
              s ·{" "}
              {observation.measurement_valid ? "available" : "missing / stale"}
              {observation.latest_sensor_dropout ? " · dropout" : ""}
            </dd>
          </div>
          <div>
            <dt>Requested / executed heading</dt>
            <dd>
              {degrees(decision.action.target_heading_rad)}° /{" "}
              {degrees(decision.executed_target_rad)}°
            </dd>
          </div>
          <div>
            <dt>Manoeuvre remaining</dt>
            <dd>{(observation.manoeuvre_remaining_s ?? 0).toFixed(1)} s</dd>
          </div>
          <div>
            <dt>Known uniform current · east / north</dt>
            <dd>
              {observation.known_current_m_s
                .map((v) => v.toFixed(2))
                .join(" / ")}{" "}
              m/s
            </dd>
          </div>
          <div>
            <dt>Available forecast</dt>
            <dd>
              {forecast
                ? `Issued ${timeLabel(forecast.issued_s)} · delivered ${timeLabel(forecast.available_s)} · ${forecast.horizon_s}s horizon`
                : "None delivered at this decision"}
            </dd>
          </div>
        </dl>
        <WindBelief observation={observation} />
      </div>
      {planning ? (
        <>
          <p>
            {planning.horizon_s} s horizon · {planning.sample_count} conditional
            samples · one shared immediate action, followed by{" "}
            {planning.recourse.replaceAll("-", " ")} recourse. Objective: (1 −
            λ) × mean + λ × CVaR95, λ ={" "}
            {details.config.policy?.name === "scenario_mpc"
              ? (details.config.policy.risk_lambda ?? 0.25)
              : 0}
            .
          </p>
          <div className="table-scroll">
            <table>
              <caption>
                Conditional planning predictions (seconds of episode cost).
                These are estimates, not realised counterfactual outcomes.
              </caption>
              <thead>
                <tr>
                  <th>Candidate heading</th>
                  <th>Mean</th>
                  <th>CVaR95</th>
                  <th>Objective</th>
                  <th>Recorded choice</th>
                </tr>
              </thead>
              <tbody>
                {planning.candidates.map((c, i) => (
                  <tr
                    key={i}
                    className={
                      i === planning.selected_index ? "selected-row" : ""
                    }
                  >
                    <th scope="row">{degrees(c.heading_rad)}°</th>
                    <td>{c.mean_s.toFixed(1)}</td>
                    <td>{c.cvar_s.toFixed(1)}</td>
                    <td>{c.objective_s.toFixed(1)}</td>
                    <td>
                      {i === planning.selected_index
                        ? "Selected minimum"
                        : "Alternative"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="fineprint">
            {planning.sample_count} samples place only{" "}
            {(planning.sample_count * 0.05).toFixed(2)} effective outcomes in
            the worst 5%. This planning objective has substantial Monte Carlo
            uncertainty; final evaluation uses independent worlds. Conditional
            paths: <code>{planning.conditional_paths_hash.slice(0, 12)}</code>.
          </p>
        </>
      ) : probability ? (
        <>
          <p>
            The frozen PPO model requests its maximum-probability categorical
            heading. Finite manoeuvre execution is shown above. These
            probabilities describe its distribution on this public observation;
            they are not predicted success probabilities or a rationale for
            learned intent.
          </p>
          <div className="table-scroll">
            <table>
              <caption>
                Measured categorical distribution · model{" "}
                {details.model_hash?.slice(0, 12)}
              </caption>
              <thead>
                <tr>
                  <th>Action</th>
                  <th>Heading</th>
                  <th>Policy probability</th>
                  <th>Recorded choice</th>
                </tr>
              </thead>
              <tbody>
                {probability.probabilities.map((p, i) => (
                  <tr
                    key={i}
                    className={
                      i === probability.selected_index ? "selected-row" : ""
                    }
                  >
                    <th scope="row">{i + 1}</th>
                    <td>{degrees(probability.headings_rad[i]!)}°</td>
                    <td>{(p * 100).toFixed(2)}%</td>
                    <td>
                      {i === probability.selected_index
                        ? "Requested maximum"
                        : "Alternative"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="fineprint">
            Belief features plus present observations. No local sensitivity
            experiment is supplied.
          </p>
        </>
      ) : (
        <p>
          The recorded heuristic uses the synthetic polar’s upwind VMG, target
          and layline geometry, boundary guard and commitment rule. It supplies
          a heading command; no Monte Carlo candidate scores are produced.
        </p>
      )}
      {points.length > 0 && (
        <details>
          <summary>Available forecast means and uncertainty</summary>
          <div className="table-scroll">
            <table>
              <caption>
                Public packet only. Direction is wind FROM; σ is a forecast
                model quantity.
              </caption>
              <thead>
                <tr>
                  <th>Valid elapsed time</th>
                  <th>From</th>
                  <th>Speed</th>
                  <th>Direction σ</th>
                  <th>Speed σ</th>
                </tr>
              </thead>
              <tbody>
                {points.map((p) => (
                  <tr key={p.valid_s}>
                    <th scope="row">{timeLabel(p.valid_s)}</th>
                    <td>{degrees(p.wind_from_rad)}°</td>
                    <td>{p.speed_m_s.toFixed(2)} m/s</td>
                    <td>
                      {((p.direction_std_rad * 180) / Math.PI).toFixed(1)}°
                    </td>
                    <td>{p.speed_std_m_s.toFixed(2)} m/s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </section>
  );
}
