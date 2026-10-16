import { useEffect, useState } from "react";
import Ajv from "ajv";
import schema from "../replication.schema.json";
import type {
  ReplicationPublication,
  CohortResult,
} from "./generated/replication";

const validate = new Ajv({ strict: false }).compile<ReplicationPublication>(
  schema,
);
const variants = ["B1_belief", "B2_mean", "B2_budget", "B3_mean", "B3_risk"];
const names: Record<string, string> = {
  B1_belief: "Tuned B1 belief",
  B2_mean: "Original B2 (60 s)",
  B2_budget: "Allowance-selected B2",
  B3_mean: "B3 expected cost",
  B3_risk: "B3 risk",
  nominal: "Nominal mixture (primary)",
  sensor_delay: "Sensor delay: 12 s",
  delay_tack: "Delay 12 s + minimum tack 16 s",
  oscillating: "Nominal: oscillating",
  ou: "Nominal: OU",
  regime: "Nominal: regimes",
  spatial: "Nominal: spatial",
};
const number = (value: number) =>
  value.toLocaleString("en-GB", { maximumFractionDigits: 2 });
const difference = (point: number, bounds: [number, number]) =>
  `${point >= 0 ? "+" : ""}${point.toFixed(2)} [${bounds[0].toFixed(2)}, ${bounds[1].toFixed(2)}] s`;
function complete(group: CohortResult) {
  return (
    group.policies.length === 5 &&
    new Set(group.policies.map((p) => p.name)).size === 5 &&
    variants.every((v) => group.policies.some((p) => p.name === v)) &&
    group.contrasts.length === 4 &&
    new Set(group.contrasts.map((c) => c.reference)).size === 4 &&
    variants
      .slice(0, 4)
      .every((v) => group.contrasts.some((c) => c.reference === v)) &&
    Math.abs(group.tail_effective_worlds - 0.05 * group.worlds) < 1e-9 &&
    group.policies.every(
      (p) =>
        p.worlds === group.worlds &&
        p.finished + p.deadline_exceeded + p.boundary_violation === p.worlds &&
        p.model_steps <= p.allocated_model_step_ceilings &&
        p.model_steps <= p.model_step_calls_including_splits &&
        (p.finished === p.worlds
          ? p.zero_failure_upper95 !== null &&
            Math.abs(p.zero_failure_upper95 - (1 - 0.05 ** (1 / p.worlds))) <
              1e-10
          : p.zero_failure_upper95 === null),
    ) &&
    group.contrasts.every(
      (c) =>
        c.mean_difference_95[0] <= c.mean_difference_95[1] &&
        c.cvar95_difference_95[0] <= c.cvar95_difference_95[1],
    )
  );
}

export function ReplicationView() {
  const [data, setData] = useState<ReplicationPublication | null>(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState("nominal");
  useEffect(() => {
    const abort = new AbortController();
    fetch("/replication.json", { signal: abort.signal })
      .then(async (response) => {
        if (!response.ok)
          throw new Error(
            "No complete independent replication is published yet.",
          );
        const raw = await response.text();
        if (raw.length > 2_000_000)
          throw new Error("Replication evidence exceeds the size limit.");
        const value: unknown = JSON.parse(raw);
        if (!validate(value))
          throw new Error("Replication evidence has an unsupported schema.");
        const nominal = value.cohorts.find((c) => c.name === "nominal");
        if (
          value.cohorts.length !== 3 ||
          new Set(value.cohorts.map((c) => c.name)).size !== 3 ||
          !["nominal", "sensor_delay", "delay_tack"].every((n) =>
            value.cohorts.some((c) => c.name === n),
          ) ||
          value.nominal_families.length !== 4 ||
          new Set(value.nominal_families.map((c) => c.name)).size !== 4 ||
          !["oscillating", "ou", "regime", "spatial"].every((n) =>
            value.nominal_families.some((c) => c.name === n),
          ) ||
          [...value.cohorts, ...value.nominal_families].some(
            (c) => !complete(c),
          ) ||
          value.total_worlds !==
            value.cohorts.reduce((sum, c) => sum + c.worlds, 0) ||
          value.retained_episodes !== 5 * value.total_worlds ||
          !nominal ||
          nominal.worlds !==
            value.nominal_families.reduce((sum, c) => sum + c.worlds, 0) ||
          value.precision_target_met !== nominal.worlds >= 1024
        ) {
          throw new Error(
            "Replication evidence has missing or inconsistent paired denominators.",
          );
        }
        setData(value);
      })
      .catch((reason: unknown) => {
        if (!abort.signal.aborted)
          setError(
            reason instanceof Error
              ? reason.message
              : "Replication could not load.",
          );
      });
    return () => abort.abort();
  }, []);
  const group =
    data &&
    [...data.cohorts, ...data.nominal_families].find(
      (c) => c.name === selected,
    );
  return (
    <section className="panel" aria-label="Independent planner replication">
      <p className="eyebrow">07 / Independent replication</p>
      <h2>
        Does the planner result survive more worlds and a stronger baseline?
      </h2>
      {!data || !group ? (
        <p role="status">{error || "Loading complete replication evidence…"}</p>
      ) : (
        <>
          <p>
            Independent synthetic follow-up: {number(data.total_worlds)} paired
            worlds and all {number(data.retained_episodes)} episodes. The
            original M5 study remains separate. No outcomes selected these
            sample sizes.
          </p>
          <p>
            Validation selected a {data.selected_deterministic_horizon_s} s
            deterministic horizon from 60/120/240/480 s. B3 uses eight 60 s
            paths. Both have a 480 simulated-second allowance per immediate
            heading; actual work and runtime differ. Latency includes the
            counter and concurrent CPU workers.
          </p>
          <label className="replication-select">
            Replication cohort
            <select
              value={selected}
              onChange={(event) => setSelected(event.target.value)}
            >
              {[...data.cohorts, ...data.nominal_families].map((c) => (
                <option key={c.name} value={c.name}>
                  {names[c.name]}
                </option>
              ))}
            </select>
          </label>
          <p>
            {number(group.worlds)} independent worlds ·{" "}
            {number(group.tail_effective_worlds)} effective CVaR95 tail worlds.
            {group.tail_effective_worlds < 30 &&
              " Small tail cohort: conclusions remain exploratory."}
            {!data.precision_target_met &&
              " The nominal replication sample target was not reached."}
          </p>
          <div className="table-scroll">
            <table>
              <caption>
                B3 risk minus reference: lower cost favours B3 risk. Pointwise
                paired 95% intervals.
              </caption>
              <thead>
                <tr>
                  <th>Reference</th>
                  <th>Mean cost difference</th>
                  <th>CVaR95 cost difference</th>
                </tr>
              </thead>
              <tbody>
                {group.contrasts.map((c) => (
                  <tr key={c.reference}>
                    <th scope="row">{names[c.reference]}</th>
                    <td>
                      {difference(c.mean_difference_s, c.mean_difference_95)}
                    </td>
                    <td>
                      {difference(
                        c.cvar95_difference_s,
                        c.cvar95_difference_95,
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
                All controllers and failures retained; failed episodes cost 2400
                s.
              </caption>
              <thead>
                <tr>
                  <th>Controller</th>
                  <th>Mean / CVaR95 s</th>
                  <th>Finished</th>
                  <th>Boundary / deadline</th>
                  <th>Decision p95 ms</th>
                  <th>Outer steps / ceiling</th>
                  <th>Step calls incl. splits</th>
                </tr>
              </thead>
              <tbody>
                {group.policies.map((p) => (
                  <tr key={p.name}>
                    <th scope="row">{names[p.name]}</th>
                    <td>
                      {number(p.mean_cost_s)} / {number(p.cvar95_s)}
                    </td>
                    <td>
                      {p.finished} / {p.worlds}
                    </td>
                    <td>
                      {p.boundary_violation} / {p.deadline_exceeded}
                    </td>
                    <td>{number(1000 * p.decision_latency_p95_s)}</td>
                    <td>
                      {number(p.model_steps)} /{" "}
                      {number(p.allocated_model_step_ceilings)}
                    </td>
                    <td>{number(p.model_step_calls_including_splits)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {group.policies
            .filter((p) => p.zero_failure_upper95 !== null)
            .map((p) => (
              <p className="quiet" key={p.name}>
                {names[p.name]}: no observed failures; one-sided 95%
                failure-probability upper bound{" "}
                {number(100 * (p.zero_failure_upper95 ?? 0))}%.
              </p>
            ))}
          <p>{data.limitations.join(" ")}</p>
          <p>
            CVaR95 can saturate at the 2400 s failure cost when failures occupy
            the whole upper tail. A narrow interval on that plateau does not
            estimate how long unfinished routes would have taken.
          </p>
          <p className="quiet">
            {number(data.evaluation_wall_s)} s with {data.workers} CPU workers ·{" "}
            {number(data.bootstrap_count)} draws, seed {data.analysis_seed},
            whole worlds resampled within families. CVaR uses fractional tail
            mass.
          </p>
          <details>
            <summary>Source identities and full evidence</summary>
            <dl className="replication-sources">
              <dt>New lock</dt>
              <dd className="mono">{data.lock_sha256}</dd>
              <dt>Original M5 lock</dt>
              <dd className="mono">{data.original_lock_sha256}</dd>
              <dt>Frozen numerical source</dt>
              <dd className="mono">{data.numerical_source_sha256}</dd>
              <dt>Complete episode table SHA-256</dt>
              <dd className="mono">{data.episodes_sha256}</dd>
            </dl>
            <p>
              <a href={data.episodes_url} download>
                Download all replication episodes (Parquet)
              </a>{" "}
              ·{" "}
              <a href={data.report_url} download>
                Download report and penalty sensitivity
              </a>
              {" · "}
              <a href="/replication/registration.json" download>
                Frozen policies and source versions
              </a>
              {" · "}
              <a href="/replication/scenarios.json" download>
                Registered scenario configurations
              </a>
            </p>
            {data.cohorts.some((c) => c.name === selected) && (
              <p>
                <a href={`/replication/${selected}.svg`} download>
                  Download paired comparison figure (SVG)
                </a>{" "}
                ·{" "}
                <a href={`/replication/${selected}.pdf`} download>
                  PDF
                </a>
              </p>
            )}
          </details>
        </>
      )}
    </section>
  );
}
