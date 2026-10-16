import { useEffect, useState } from "react";
import Ajv, { type ValidateFunction } from "ajv";
import computeSchema from "../compute-followup.schema.json";
import empiricalSchema from "../empirical.schema.json";
import type {
  ComputePublication,
  WorkCohort,
} from "./generated/compute-followup";
import type { EmpiricalPublication } from "./generated/empirical";

const ajv = new Ajv({ strict: false });
const validateCompute = ajv.compile<ComputePublication>(computeSchema);
const validateEmpirical = ajv.compile<EmpiricalPublication>(empiricalSchema);
const names: Record<string, string> = {
  B1_belief: "Tuned B1 belief",
  B2_matched: "Deterministic B2 (work target)",
  B3_mean_matched: "B3 expected cost (2 paths)",
  B3_risk_matched: "B3 risk (2 paths)",
  B3_risk_original: "Original B3 risk (8 paths; unmatched)",
  nominal: "Nominal mixture (primary)",
  new_family: "New-process mixture (primary)",
  gust_front: "New family: gust fronts",
  persistent_trend: "New family: persistent trends",
  oscillating: "Nominal: oscillating",
  ou: "Nominal: OU",
  regime: "Nominal: regimes",
  spatial: "Nominal: spatial",
};
const num = (x: number, digits = 2) =>
  x.toLocaleString("en-GB", { maximumFractionDigits: digits });
const difference = (x: number, ci: [number, number], digits = 2) =>
  `${x >= 0 ? "+" : ""}${x.toFixed(digits)} [${ci[0].toFixed(digits)}, ${ci[1].toFixed(digits)}]`;
const sameSet = (values: string[], expected: string[]) =>
  values.length === expected.length &&
  new Set(values).size === values.length &&
  expected.every((v) => values.includes(v));

function validCohort(g: WorkCohort): boolean {
  const policies = g.policies;
  const variants = Object.keys(names).filter((v) => v.startsWith("B"));
  if (
    !sameSet(
      policies.map((p) => p.name),
      variants,
    ) ||
    !sameSet(
      g.contrasts.map((d) => d.reference),
      variants.filter((v) => v !== "B3_risk_matched"),
    ) ||
    Math.abs(g.tail_effective_worlds - g.worlds * 0.05) > 1e-9
  )
    return false;
  if (
    policies.some(
      (p) =>
        p.finished + p.boundary_violation + p.deadline_exceeded !== g.worlds ||
        p.recursive_step_calls < p.steps ||
        (p.finished === g.worlds
          ? p.zero_failure_upper95 == null ||
            Math.abs(p.zero_failure_upper95 - (1 - 0.05 ** (1 / g.worlds))) >
              1e-10
          : p.zero_failure_upper95 != null),
    )
  )
    return false;
  if (
    g.contrasts.some(
      (d) =>
        d.mean_difference_95[0] > d.mean_difference_95[1] ||
        d.cvar95_difference_95[0] > d.cvar95_difference_95[1],
    )
  )
    return false;
  const b2 = policies.find((p) => p.name === "B2_matched")!;
  const matched =
    b2.steps_per_decision > 0 &&
    ["B3_mean_matched", "B3_risk_matched"].every(
      (v) =>
        Math.abs(
          policies.find((p) => p.name === v)!.steps_per_decision /
            b2.steps_per_decision -
            1,
        ) <= g.matching_tolerance,
    );
  return matched === g.work_matching_passed;
}
function validCompute(v: ComputePublication): boolean {
  if (
    v.episodes !== 5 * v.worlds ||
    !sameSet(
      v.cohorts.map((g) => g.name),
      ["nominal", "new_family"],
    ) ||
    !sameSet(
      v.families.map((g) => g.name),
      [
        "oscillating",
        "ou",
        "regime",
        "spatial",
        "gust_front",
        "persistent_trend",
      ],
    ) ||
    v.cohorts.reduce((n, g) => n + g.worlds, 0) !== v.worlds
  )
    return false;
  return (
    [...v.cohorts, ...v.families].every(validCohort) &&
    v.cohorts.every(
      (g) =>
        v.families
          .filter((f) =>
            g.name === "new_family"
              ? ["gust_front", "persistent_trend"].includes(f.name)
              : !["gust_front", "persistent_trend"].includes(f.name),
          )
          .reduce((n, f) => n + f.worlds, 0) === g.worlds,
    )
  );
}
function validEmpirical(v: EmpiricalPublication): boolean {
  if (
    v.raw_gps_rows !==
      v.excluded_gps_rows + v.accepted_gps_rows + v.rejected_gps_rows ||
    v.train_voyages > v.registered_train_identifiers ||
    v.heldout_voyages > v.registered_heldout_identifiers ||
    !sameSet(
      v.predictors.map((p) => p.name),
      ["AR1", "persistence", "climatology"],
    ) ||
    v.predictors.some((p) => p.heldout_pairs !== v.wind_heldout_pairs) ||
    v.vector_rmse_difference95[0] > v.vector_rmse_difference95[1] ||
    v.projection_gate_passed !== v.projection_heldout_p99_relative <= 0.01
  )
    return false;
  const scores = Object.fromEntries(
    v.predictors.map((p) => [p.name, p.vector_rmse_m_s]),
  );
  const ar1 = scores.AR1;
  const persistence = scores.persistence;
  return (
    ar1 !== undefined &&
    persistence !== undefined &&
    Math.abs(ar1 - persistence - v.vector_rmse_difference_m_s) < 1e-10
  );
}
function useEvidence<T>(
  url: string,
  validate: ValidateFunction<T>,
  consistent: (v: T) => boolean,
) {
  const [value, setValue] = useState<T | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    fetch(url, { signal: controller.signal })
      .then(async (response) => {
        if (response.status === 404) return;
        if (!response.ok) throw new Error("Evidence request failed");
        // Vite/static SPA hosts may serve index.html for unpublished optional evidence.
        if (response.headers.get("content-type")?.includes("text/html")) return;
        const text = await response.text();
        if (new TextEncoder().encode(text).length > 2_000_000)
          throw new Error("Evidence exceeds size bound");
        const data: unknown = JSON.parse(text);
        if (!validate(data) || !consistent(data))
          throw new Error("Incomplete or inconsistent evidence");
        setValue(data);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setError(
            reason instanceof Error ? reason.message : "Invalid evidence",
          );
      });
    return () => controller.abort();
  }, [url, validate, consistent]);
  return { value, error };
}

export function ComputeFollowupView() {
  const { value, error } = useEvidence(
    "/compute-followup.json",
    validateCompute,
    validCompute,
  );
  const [selected, setSelected] = useState("nominal");
  const group = value
    ? [...value.cohorts, ...value.families].find((g) => g.name === selected)
    : undefined;
  const b2 = group?.policies.find((p) => p.name === "B2_matched");
  return (
    <section
      aria-label="Measured work and new wind families"
      className="results-panel followup-panel"
    >
      <p className="eyebrow">08 / Work and generalisation</p>
      <h2>Does the result survive comparable work and unseen winds?</h2>
      {error ? (
        <p role="alert">{error}</p>
      ) : !value ? (
        <p>No complete measured-work study is published yet.</p>
      ) : (
        group && (
          <>
            <p>
              Locked synthetic follow-up: {num(value.worlds)} paired worlds and
              all {num(value.episodes)} episodes. Validation selected a{" "}
              {value.horizon_s} s deterministic horizon; matched B3 uses two 60
              s paths. Original eight-path risk MPC remains an unmatched
              reference.
            </p>
            <label className="replication-select">
              Work comparison cohort
              <select
                value={selected}
                onChange={(e) => setSelected(e.target.value)}
              >
                {[...value.cohorts, ...value.families].map((g) => (
                  <option key={g.name} value={g.name}>
                    {names[g.name]}
                  </option>
                ))}
              </select>
            </label>
            <p>
              {num(group.worlds)} independent worlds ·{" "}
              {num(group.tail_effective_worlds)} effective CVaR95 tail worlds.
            </p>
            {group.tail_effective_worlds < 30 && (
              <p className="error">
                Small family tail: estimates remain sensitive to individual
                worlds.
              </p>
            )}
            <p>
              <strong>
                {group.work_matching_passed
                  ? "Measured integration-work match passes."
                  : "Measured integration-work match fails."}
              </strong>{" "}
              Both two-path B3 variants must fall within ±
              {num(group.matching_tolerance * 100)}% of B2’s average actual
              integration requests per decision. This is a cohort average, not
              exact equality at every state, equal runtime or equal episode
              work.
            </p>
            {["nominal", "new_family"].includes(group.name) && (
              <figure className="paired-figure">
                <div
                  className="scientific-figure-scroll"
                  role="region"
                  aria-label="Scrollable paired scientific figure"
                  tabIndex={0}
                >
                  <img
                    src={`/compute-followup/${group.name}.svg`}
                    alt={`Two-path B3 risk mean and CVaR95 paired differences for ${names[group.name]}`}
                    loading="lazy"
                  />
                </div>
                <figcaption>
                  Registered pooled cohort; pointwise paired 95% intervals.
                  Lower cost favours two-path risk. All failures are retained.
                  Family tables below do not inherit this pooled precision.
                  Scroll horizontally to inspect the figure on narrow screens.
                </figcaption>
              </figure>
            )}
            <div className="table-scroll">
              <table>
                <caption>
                  Two-path B3 risk minus reference: lower favours risk.
                  Pointwise paired 95% intervals; failures cost 2400 s.
                </caption>
                <thead>
                  <tr>
                    <th>Reference</th>
                    <th>Mean difference (s)</th>
                    <th>CVaR95 difference (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {group.contrasts.map((d) => (
                    <tr key={d.reference}>
                      <th>{names[d.reference]}</th>
                      <td>
                        {difference(d.mean_difference_s, d.mean_difference_95)}
                      </td>
                      <td>
                        {difference(
                          d.cvar95_difference_s,
                          d.cvar95_difference_95,
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="table-scroll">
              <table>
                <caption>
                  All outcomes and actual model work retained. Finish / boundary
                  / deadline counts sum to the paired denominator.
                </caption>
                <thead>
                  <tr>
                    <th>Controller</th>
                    <th>Mean / CVaR95 (s)</th>
                    <th>Finish / boundary / deadline</th>
                    <th>Outer steps / decision</th>
                    <th>Work ratio to B2</th>
                    <th>Decision p95 (ms)</th>
                  </tr>
                </thead>
                <tbody>
                  {group.policies.map((p) => (
                    <tr key={p.name}>
                      <th>{names[p.name]}</th>
                      <td>
                        {num(p.mean_cost_s)} / {num(p.cvar95_s)}
                      </td>
                      <td>
                        {p.finished} / {p.boundary_violation} /{" "}
                        {p.deadline_exceeded}
                      </td>
                      <td>{num(p.steps_per_decision)}</td>
                      <td>
                        {b2 &&
                          num(p.steps_per_decision / b2.steps_per_decision, 3)}
                      </td>
                      <td>{num(1000 * p.decision_latency_p95_s, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>
              Two samples make planning CVaR95 the maximum sampled cost. Nominal
              and new-family mixtures are separately registered; gust fronts and
              persistent random trends were absent from tuning. Results do not
              establish real sailing effectiveness.
            </p>
            <p>
              The prior eight-path risk versus B1 and eight-path mean-B3
              comparisons retain their unresolved conclusion. This new two-path
              objective ablation is a different comparison; intervals crossing
              zero do not prove equality.
            </p>
            <details className="followup-evidence">
              <summary>Registered sources and full evidence</summary>
              <dl>
                <dt>Lock SHA256</dt>
                <dd>{value.lock_sha256}</dd>
                <dt>Report SHA256</dt>
                <dd>{value.report_sha256}</dd>
                <dt>Complete episodes SHA256</dt>
                <dd>{value.episodes_sha256}</dd>
              </dl>
              <p>
                {num(value.wall_s)} s measured local evaluation. Latency
                includes metering and concurrent workers.
              </p>
              <p>
                <a href="/compute-followup/episodes.parquet" download>
                  Download every work-study episode
                </a>{" "}
                ·{" "}
                <a href="/compute-followup/report.json" download>
                  Download work report and penalty sensitivities
                </a>{" "}
                ·{" "}
                <a href="/compute-followup/registration.json" download>
                  Download work registration
                </a>{" "}
                ·{" "}
                <a
                  href={`/compute-followup/${["gust_front", "persistent_trend", "new_family"].includes(group.name) ? "new_family" : "nominal"}.pdf`}
                  download
                >
                  Download paired figure PDF
                </a>
              </p>
              {value.limitations.map((text) => (
                <p key={text}>{text}</p>
              ))}
            </details>
          </>
        )
      )}
    </section>
  );
}

export function RecordedValidationView() {
  const { value, error } = useEvidence(
    "/empirical.json",
    validateEmpirical,
    validEmpirical,
  );
  return (
    <section
      aria-label="Recorded component validation"
      className="results-panel followup-panel"
    >
      <p className="eyebrow">09 / Recorded components</p>
      <h2>What do real observations actually validate?</h2>
      {error ? (
        <p role="alert">{error}</p>
      ) : !value ? (
        <p>No complete recorded-data validation is published yet.</p>
      ) : (
        <>
          <p>
            Actual public sailing GPS and buoy observations, separate from
            synthetic policy results. No boat polar or tack dynamics is
            empirically calibrated.
          </p>
          <p>
            <strong>{num(value.raw_gps_rows)} source GPS rows:</strong>{" "}
            {num(value.accepted_gps_rows)} accepted,{" "}
            {num(value.rejected_gps_rows)} quality-rejected,{" "}
            {num(value.excluded_gps_rows)} excluded for provenance/clock
            reasons. Every source row remains accounted for.
          </p>
          <p>
            {value.train_voyages} training / {value.heldout_voyages} held-out
            voyages with primary GPS. Registration contains{" "}
            {value.registered_train_identifiers} /{" "}
            {value.registered_heldout_identifiers} source identifiers before
            provenance exclusions. Reconstructed points and GPS gaps do not
            become continuous observed paths.
          </p>
          <p>
            <strong>
              {value.projection_gate_passed
                ? "Held-out projection consistency passes."
                : "Held-out projection consistency fails."}
            </strong>{" "}
            p99 relative speed difference against WGS84 geodesic distances is{" "}
            {num(value.projection_heldout_p99_relative * 100, 6)}%, on{" "}
            {num(value.projection_heldout_pairs)} intervals. Both estimates use
            the same GPS; this is numerical consistency, not independent GPS
            accuracy.
          </p>
          <figure>
            <img
              src="/empirical/observed-track.svg"
              alt="Recorded GPS path from the latest held-out voyage, with gaps disconnected"
              loading="lazy"
            />
            <figcaption>
              Greer Jarrett’s latest held-out voyage, first day/chunk;
              post-evaluation diagnostic. Clock assumed CET. This is an observed
              track, not a policy counterfactual.
            </figcaption>
          </figure>
          <p>
            NOAA buoy 44013: {value.wind_train_days} training days,{" "}
            {value.wind_heldout_days} later held-out days and{" "}
            {num(value.wind_heldout_pairs)} paired ten-minute transitions. Fits
            and predictive covariances use training observations only.
          </p>
          <div className="table-scroll">
            <table>
              <caption>
                Held-out observed-wind prediction. Same buoy supplies targets;
                retrospective observations are not historical forecasts.
              </caption>
              <thead>
                <tr>
                  <th>Predictor</th>
                  <th>Vector RMSE (m/s)</th>
                  <th>Speed RMSE (m/s)</th>
                  <th>Direction RMSE (°)</th>
                  <th>95% coverage</th>
                  <th>Gaussian NLL</th>
                </tr>
              </thead>
              <tbody>
                {value.predictors.map((p) => (
                  <tr key={p.name}>
                    <th>{p.name}</th>
                    <td>{num(p.vector_rmse_m_s, 4)}</td>
                    <td>{num(p.speed_rmse_m_s, 4)}</td>
                    <td>
                      {p.direction_rmse_deg_at_speed_ge_0_5 == null
                        ? "Unavailable"
                        : num(p.direction_rmse_deg_at_speed_ge_0_5)}
                    </td>
                    <td>{num(p.coverage95 * 100)}%</td>
                    <td>{num(p.gaussian_nll_mean, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            AR(1) minus persistence vector RMSE:{" "}
            {difference(
              value.vector_rmse_difference_m_s,
              value.vector_rmse_difference95,
              5,
            )}{" "}
            m/s, pointwise 95% whole-day bootstrap. Direction error excludes
            observed speeds below 0.5 m/s. Day dependence and this single
            site/season limit inference.
          </p>
          <details className="followup-evidence">
            <summary>Recorded sources, limitations and downloads</summary>
            <dl>
              <dt>Registration SHA256</dt>
              <dd>{value.registration_sha256}</dd>
              <dt>Report SHA256</dt>
              <dd>{value.report_sha256}</dd>
            </dl>
            <p>
              <a href="/empirical/report.json" download>
                Download recorded-component report
              </a>{" "}
              ·{" "}
              <a href="/empirical/gps-processed-points.parquet" download>
                Download processed public GPS
              </a>{" "}
              ·{" "}
              <a href="/empirical/wind-heldout-predictions.parquet" download>
                Download every held-out wind prediction
              </a>{" "}
              ·{" "}
              <a href="/empirical/wind-validation.pdf" download>
                Download wind figure PDF
              </a>{" "}
              ·{" "}
              <a href="/empirical/LICENSE-DATA.md" download>
                Download attribution and data licence
              </a>
            </p>
            <p>
              <a href="https://zenodo.org/records/17340505">
                Published sailing source
              </a>{" "}
              ·{" "}
              <a href="https://www.ndbc.noaa.gov/data/historical/stdmet/44013h2025.txt.gz">
                NOAA wind source
              </a>
            </p>
            {Object.entries(value.attribution).map(([key, text]) => (
              <p key={key}>{text}</p>
            ))}
            {value.limitations.map((text) => (
              <p key={text}>{text}</p>
            ))}
          </details>
        </>
      )}
    </section>
  );
}
