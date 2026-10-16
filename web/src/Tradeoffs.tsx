import { useEffect, useState } from "react";
import Ajv from "ajv";

type Policy = {
  worlds: number;
  completions: number;
  mean_cost_s: number;
  cvar95_s: number;
  paired_mean_difference_s: number;
  paired_mean_difference_95: [number, number];
  decision_latency_p95_s: number;
};
type Group = {
  training_seeds: number;
  mean_cost_s: number;
  cvar95_mixture_s: number;
  paired_mean_difference_s: number;
  hierarchical_paired_mean_95: [number, number];
};
type Evidence = {
  stage: "locked_learning";
  lock_hash: string;
  slices: {
    nominal: {
      independent_worlds: number;
      policies: Record<string, Policy>;
      learned_groups: Record<string, Group>;
    };
  };
  classical_cohort: {
    independent_worlds: number;
    policies: Record<string, Policy>;
    paired_vs_B2: { policies: Record<string, Policy> };
  };
  compute: {
    wall_s: number;
    workers: number;
    device: "cpu";
    latency_condition: string;
  };
};
const ciSchema = {
  type: "array",
  minItems: 2,
  maxItems: 2,
  items: { type: "number" },
};
const policySchema = {
  type: "object",
  required: [
    "worlds",
    "completions",
    "mean_cost_s",
    "cvar95_s",
    "paired_mean_difference_s",
    "paired_mean_difference_95",
    "decision_latency_p95_s",
  ],
  properties: {
    worlds: { type: "integer", minimum: 1 },
    completions: { type: "integer", minimum: 0 },
    mean_cost_s: { type: "number", minimum: 0 },
    cvar95_s: { type: "number", minimum: 0 },
    paired_mean_difference_s: { type: "number" },
    paired_mean_difference_95: ciSchema,
    decision_latency_p95_s: { type: "number", minimum: 0 },
  },
};
const policiesSchema = {
  type: "object",
  minProperties: 1,
  additionalProperties: policySchema,
};
const groupSchema = {
  type: "object",
  required: [
    "training_seeds",
    "mean_cost_s",
    "cvar95_mixture_s",
    "paired_mean_difference_s",
    "hierarchical_paired_mean_95",
  ],
  properties: {
    training_seeds: { type: "integer", minimum: 1 },
    mean_cost_s: { type: "number", minimum: 0 },
    cvar95_mixture_s: { type: "number", minimum: 0 },
    paired_mean_difference_s: { type: "number" },
    hierarchical_paired_mean_95: ciSchema,
  },
};
const validate = new Ajv().compile<Evidence>({
  type: "object",
  required: ["stage", "lock_hash", "slices", "classical_cohort", "compute"],
  properties: {
    stage: { const: "locked_learning" },
    lock_hash: { type: "string", pattern: "^[a-f0-9]{64}$" },
    slices: {
      type: "object",
      required: ["nominal"],
      properties: {
        nominal: {
          type: "object",
          required: ["independent_worlds", "policies", "learned_groups"],
          properties: {
            independent_worlds: { type: "integer", minimum: 1 },
            policies: policiesSchema,
            learned_groups: {
              type: "object",
              minProperties: 1,
              additionalProperties: groupSchema,
            },
          },
        },
      },
    },
    classical_cohort: {
      type: "object",
      required: ["independent_worlds", "policies", "paired_vs_B2"],
      properties: {
        independent_worlds: { type: "integer", minimum: 1 },
        policies: policiesSchema,
        paired_vs_B2: {
          type: "object",
          required: ["policies"],
          properties: { policies: policiesSchema },
        },
      },
    },
    compute: {
      type: "object",
      required: ["wall_s", "workers", "device", "latency_condition"],
      properties: {
        wall_s: { type: "number", minimum: 0 },
        workers: { type: "integer", minimum: 1 },
        device: { const: "cpu" },
        latency_condition: { type: "string" },
      },
    },
  },
});

type Point = { name: string; mean: number; tail: number };
function Frontier({ points, worlds }: { points: Point[]; worlds: number }) {
  const [selectedName, setSelectedName] = useState("");
  const [compact, setCompact] = useState(
    () => window.matchMedia("(max-width: 650px)").matches,
  );
  useEffect(() => {
    const media = window.matchMedia("(max-width: 650px)");
    const update = () => setCompact(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  const width = compact ? 360 : 610;
  const left = compact ? 65 : 80;
  const span = width - left - 50;
  const selected = points.find((p) => p.name === selectedName) ?? points[0]!;
  const xmin = Math.min(...points.map((p) => p.mean)) * 0.95;
  const xmax = Math.max(...points.map((p) => p.mean)) * 1.05;
  const ymin = Math.min(...points.map((p) => p.tail)) * 0.85;
  const ymax = Math.max(...points.map((p) => p.tail)) * 1.05;
  const x = (value: number) =>
    left + ((value - xmin) / (xmax - xmin || 1)) * span;
  const y = (value: number) =>
    260 - ((value - ymin) / (ymax - ymin || 1)) * 205;
  return (
    <>
      <label className="learning-select">
        Highlight policy
        <select
          value={selected.name}
          onChange={(e) => setSelectedName(e.target.value)}
        >
          {points.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name.replaceAll("_", " ")}
            </option>
          ))}
        </select>
      </label>
      <svg
        className="frontier-chart"
        viewBox={`0 0 ${width} 325`}
        style={{ fontSize: compact ? 14 : 11 }}
        role="img"
        aria-label="Recorded expected and tail cost trade-offs"
      >
        <path
          d={`M${left} 40V260H${width - 50}`}
          fill="none"
          stroke="#80989a"
        />
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <text x={left + span * v} y="278" textAnchor="middle">
              {(xmin + (xmax - xmin) * v).toFixed(0)}
            </text>
            <text x={left - 10} y={260 - 205 * v} textAnchor="end">
              {(ymin + (ymax - ymin) * v).toFixed(0)}
            </text>
          </g>
        ))}
        <text x={width / 2} y="313" textAnchor="middle">
          Mean cost (s) · lower is better
        </text>
        <text x="22" y="155" transform="rotate(-90 22 155)" textAnchor="middle">
          Upper-tail CVaR95 cost (s)
        </text>
        {points.map((p, i) => (
          <g key={p.name}>
            <circle
              cx={x(p.mean)}
              cy={y(p.tail)}
              r={p.name === selected.name ? 8 : 5}
              fill={p.name === selected.name ? "#a65532" : "#14776e"}
            />
            <text x={x(p.mean) + 10} y={y(p.tail) + (i % 2 ? 15 : -10)}>
              {i + 1}
            </text>
            <title>
              {p.name}: mean {p.mean.toFixed(1)}, CVaR95 {p.tail.toFixed(1)}
            </title>
          </g>
        ))}
      </svg>
      <p>
        {selected.name.replaceAll("_", " ")}: mean {selected.mean.toFixed(1)} s,
        CVaR95 {selected.tail.toFixed(1)} s. {worlds} independent worlds.
      </p>
      <p className="fineprint">
        Numbered points follow the selector order. Point estimates, not a fitted
        efficient frontier or a smooth risk-aversion curve. PPO tail cost is the
        equally weighted model/world mixture; confidence intervals and training
        variation are reported separately.
      </p>
    </>
  );
}

export function Tradeoffs() {
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [cohort, setCohort] = useState("nominal");
  const [error, setError] = useState("");
  useEffect(() => {
    const abort = new AbortController();
    fetch("/learning.json", { signal: abort.signal })
      .then(async (r) => {
        if (!r.ok) return;
        const value: unknown = await r.json();
        // Older development summaries remain supported by the existing results panel.
        if (!validate(value)) return;
        const policySets = [
          value.slices.nominal.policies,
          value.classical_cohort.policies,
          value.classical_cohort.paired_vs_B2.policies,
        ];
        if (
          policySets.some((s) =>
            Object.values(s).some(
              (p) =>
                p.completions > p.worlds ||
                p.cvar95_s < p.mean_cost_s ||
                p.paired_mean_difference_95[0] > p.paired_mean_difference_95[1],
            ),
          ) ||
          Object.values(value.slices.nominal.learned_groups).some(
            (p) =>
              p.cvar95_mixture_s < p.mean_cost_s ||
              p.hierarchical_paired_mean_95[0] >
                p.hierarchical_paired_mean_95[1],
          )
        )
          throw new Error("Inconsistent locked trade-off evidence.");
        if (!abort.signal.aborted) setEvidence(value);
      })
      .catch((reason: unknown) => {
        if (!abort.signal.aborted)
          setError(
            reason instanceof Error
              ? reason.message
              : "Trade-offs unavailable.",
          );
      });
    return () => abort.abort();
  }, []);
  if (!evidence) return error ? <p role="status">{error}</p> : null;
  const nominal = evidence.slices.nominal;
  const classical = evidence.classical_cohort;
  const points: Point[] =
    cohort === "nominal"
      ? [
          ...Object.entries(nominal.policies)
            .filter(([name]) => name === "B1_belief")
            .map(([name, p]) => ({
              name,
              mean: p.mean_cost_s,
              tail: p.cvar95_s,
            })),
          ...Object.entries(nominal.learned_groups).map(([name, p]) => ({
            name,
            mean: p.mean_cost_s,
            tail: p.cvar95_mixture_s,
          })),
        ]
      : Object.entries(classical.policies).map(([name, p]) => ({
          name,
          mean: p.mean_cost_s,
          tail: p.cvar95_s,
        }));
  const fixed = nominal.learned_groups.PPO_belief_fixed;
  return (
    <section className="results-panel" aria-labelledby="tradeoffs-title">
      <p className="eyebrow">THE CURRENT ANSWER · M5 LOCKED RESULTS</p>
      <h2 id="tradeoffs-title">Complexity has to earn its cost</h2>
      {fixed && (
        <p>
          Across {nominal.independent_worlds} nominal worlds and{" "}
          {fixed.training_seeds} training seeds, fixed belief PPO costs{" "}
          {Math.abs(fixed.paired_mean_difference_s).toFixed(1)} s{" "}
          {fixed.paired_mean_difference_s >= 0 ? "more" : "less"} than the tuned
          belief heuristic on average (signed difference; pointwise joint 95%
          interval{" "}
          {fixed.hierarchical_paired_mean_95
            .map((v) => v.toFixed(1))
            .join(" to ")}{" "}
          s).{" "}
          {fixed.hierarchical_paired_mean_95[0] > 0
            ? "This experiment supports a negative learned-policy result."
            : "The interval determines whether the difference is resolved."}
        </p>
      )}
      <label className="learning-select">
        Trade-off cohort
        <select value={cohort} onChange={(e) => setCohort(e.target.value)}>
          <option value="nominal">
            Nominal · heuristic and learned groups
          </option>
          <option value="classical">
            Registered classical subset · planning
          </option>
        </select>
      </label>
      <Frontier
        points={points}
        worlds={
          cohort === "nominal"
            ? nominal.independent_worlds
            : classical.independent_worlds
        }
      />
      <div className="table-scroll">
        <table>
          <caption>
            Registered classical subset: {classical.independent_worlds} paired
            worlds, including nominal and shifted conditions. These differ from
            the full nominal learned-policy cohort.
          </caption>
          <thead>
            <tr>
              <th>Policy</th>
              <th>Finished</th>
              <th>Mean cost</th>
              <th>CVaR95</th>
              <th>Paired mean vs B1 / 95% interval</th>
              <th>Paired mean vs B2 / 95% interval</th>
              <th>p95 decision compute</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(classical.policies).map(([name, p]) => {
              const b2 = classical.paired_vs_B2.policies[name];
              return (
                <tr key={name}>
                  <th scope="row">{name.replaceAll("_", " ")}</th>
                  <td>
                    {p.completions}/{p.worlds}
                  </td>
                  <td>{p.mean_cost_s.toFixed(1)} s</td>
                  <td>{p.cvar95_s.toFixed(1)} s</td>
                  <td>
                    {p.paired_mean_difference_s.toFixed(1)} [
                    {p.paired_mean_difference_95
                      .map((v) => v.toFixed(1))
                      .join(", ")}
                    ] s
                  </td>
                  <td>
                    {b2
                      ? `${b2.paired_mean_difference_s.toFixed(1)} [${b2.paired_mean_difference_95.map((v) => v.toFixed(1)).join(", ")}] s`
                      : "Unavailable"}
                  </td>
                  <td>{(p.decision_latency_p95_s * 1000).toFixed(2)} ms</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="fineprint">
        Cost is actual finish time on success and 2400 s on sailing failure.
        CVaR95 averages the worst 5% with fractional tail weights; this subset
        supplies only {(classical.independent_worlds * 0.05).toFixed(1)} tail
        outcomes. Risk planning’s paired advantage against deterministic
        planning is conditional on this registered mixture. Its difference from
        the strong B1 and tail precision remain unresolved. Intervals are
        pointwise, not simultaneous across hypotheses or slices. No
        matched-compute superiority claim is made.
      </p>
      <p>
        Evaluation compute: {evidence.compute.wall_s.toFixed(1)} s with{" "}
        {evidence.compute.workers} local CPU workers. Decision latencies are{" "}
        {evidence.compute.latency_condition}; recording-time latency in the
        inspector is a separate reproduction measurement.
      </p>
      <a href="/learning.json" download>
        Download locked results and intervals ↗
      </a>
    </section>
  );
}
