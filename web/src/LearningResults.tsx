import { useEffect, useState } from "react";
import Ajv from "ajv";

type Group = {
  training_seeds: number;
  independent_worlds: number;
  mean_cost_s: number;
  paired_mean_difference_s: number;
  hierarchical_paired_mean_95: [number, number];
  between_training_sd_mean_cost_s: number | null;
  cvar95_mixture_s: number;
  tail_independent_worlds: number;
};
type Curve = {
  variant: string;
  points: { steps: number; mean_cost_s: number }[];
};
type Outcome = {
  completions: number;
  worlds: number;
  mean_cost_s: number;
  failure_categories?: Record<string, number>;
};
type LearningEvidence = {
  stage: "development_learning" | "locked_learning";
  reference: string;
  learned_groups: Record<string, Group>;
  independent_worlds: number;
  episodes: number;
  curves: Curve[];
  limitations: string[];
  policies?: Record<string, Outcome>;
  slices?: Record<
    string,
    {
      independent_worlds: number;
      learned_groups: Record<string, Group>;
      policies?: Record<string, Outcome>;
    }
  >;
  lock_hash?: string;
};
const groupSchema = {
  type: "object",
  required: [
    "training_seeds",
    "independent_worlds",
    "mean_cost_s",
    "paired_mean_difference_s",
    "hierarchical_paired_mean_95",
    "between_training_sd_mean_cost_s",
    "cvar95_mixture_s",
    "tail_independent_worlds",
  ],
  properties: {
    training_seeds: { type: "integer", minimum: 1 },
    independent_worlds: { type: "integer", minimum: 1 },
    mean_cost_s: { type: "number", minimum: 0 },
    paired_mean_difference_s: { type: "number" },
    hierarchical_paired_mean_95: {
      type: "array",
      minItems: 2,
      maxItems: 2,
      items: { type: "number" },
    },
    between_training_sd_mean_cost_s: { type: ["number", "null"], minimum: 0 },
    cvar95_mixture_s: { type: "number", minimum: 0 },
    tail_independent_worlds: { type: "number", minimum: 0 },
  },
};
const groupsSchema = {
  type: "object",
  minProperties: 1,
  additionalProperties: groupSchema,
};
const outcomesSchema = {
  type: "object",
  additionalProperties: {
    type: "object",
    required: ["worlds", "completions", "mean_cost_s"],
    properties: {
      worlds: { type: "integer", minimum: 1 },
      completions: { type: "integer", minimum: 0 },
      mean_cost_s: { type: "number", minimum: 0 },
      failure_categories: {
        type: "object",
        additionalProperties: { type: "integer", minimum: 0 },
      },
    },
  },
};
const validate = new Ajv({ allowUnionTypes: true }).compile<LearningEvidence>({
  type: "object",
  required: [
    "stage",
    "reference",
    "learned_groups",
    "independent_worlds",
    "episodes",
    "curves",
    "limitations",
  ],
  properties: {
    stage: { enum: ["development_learning", "locked_learning"] },
    reference: { type: "string" },
    independent_worlds: { type: "integer", minimum: 1 },
    episodes: { type: "integer", minimum: 1 },
    learned_groups: groupsSchema,
    policies: outcomesSchema,
    lock_hash: { type: "string", pattern: "^[a-f0-9]{64}$" },
    limitations: { type: "array", items: { type: "string" } },
    curves: {
      type: "array",
      items: {
        type: "object",
        required: ["variant", "points"],
        properties: {
          variant: { type: "string" },
          points: {
            type: "array",
            minItems: 1,
            items: {
              type: "object",
              required: ["steps", "mean_cost_s"],
              properties: {
                steps: { type: "integer", minimum: 1 },
                mean_cost_s: { type: "number", minimum: 0 },
              },
            },
          },
        },
      },
    },
    slices: {
      type: "object",
      additionalProperties: {
        type: "object",
        required: ["independent_worlds", "learned_groups"],
        properties: {
          independent_worlds: { type: "integer", minimum: 1 },
          learned_groups: groupsSchema,
          policies: outcomesSchema,
        },
      },
    },
  },
});

export function LearningResults() {
  const [evidence, setEvidence] = useState<LearningEvidence | null>(null);
  const [slice, setSlice] = useState("all");
  const [curveIndex, setCurveIndex] = useState(0);
  const [message, setMessage] = useState(
    "No precomputed learning study loaded.",
  );
  useEffect(() => {
    const abort = new AbortController();
    fetch("/learning.json", { signal: abort.signal })
      .then(async (response) => {
        if (!response.ok) return;
        const value: unknown = await response.json();
        if (!validate(value))
          throw new Error("Invalid learning evidence contract.");
        const groups = [
          value.learned_groups,
          ...Object.values(value.slices ?? {}).map((s) => s.learned_groups),
        ];
        const outcomeSets = [
          value.policies,
          ...Object.values(value.slices ?? {}).map((s) => s.policies),
        ];
        if (
          (value.stage === "locked_learning" && !value.lock_hash) ||
          groups.some((g) =>
            Object.values(g).some(
              (r) =>
                r.hierarchical_paired_mean_95[0] >
                  r.hierarchical_paired_mean_95[1] ||
                r.cvar95_mixture_s + 1e-8 < r.mean_cost_s,
            ),
          ) ||
          outcomeSets.some((set) =>
            Object.values(set ?? {}).some(
              (p) =>
                p.completions > p.worlds ||
                (p.failure_categories !== undefined &&
                  Object.values(p.failure_categories).reduce(
                    (n, count) => n + count,
                    0,
                  ) !==
                    p.worlds - p.completions),
            ),
          ) ||
          value.curves.some((c) =>
            c.points.some((p, i) => p.steps <= (c.points[i - 1]?.steps ?? 0)),
          )
        )
          throw new Error("Inconsistent learning evidence or missing lock.");
        setEvidence(value);
      })
      .catch((error: unknown) => {
        if (!abort.signal.aborted)
          setMessage(
            error instanceof Error
              ? error.message
              : "Learning evidence unavailable.",
          );
      });
    return () => abort.abort();
  }, []);
  const selected = slice === "all" ? evidence : evidence?.slices?.[slice];
  const curve = evidence?.curves[curveIndex];
  const outcomes = selected?.policies ?? {};
  return (
    <section className="results-panel" aria-labelledby="learning-title">
      <p className="eyebrow">03 / MEMORY &amp; LEARNED DECISIONS</p>
      <h2 id="learning-title">Can a learned policy justify its complexity?</h2>
      {evidence ? (
        <>
          <p>
            {evidence.stage === "locked_learning"
              ? "Locked evaluation"
              : "Development validation"}
            {": "}
            {evidence.independent_worlds} independent worlds,{" "}
            {evidence.episodes} episodes. Each interval resamples training seeds
            and paired whole worlds. Lower cost is better; failures remain in
            the score at 2400 s. Ordinary PPO optimises expected cost.
          </p>
          {evidence.slices && (
            <label className="learning-select">
              Evaluation slice
              <select value={slice} onChange={(e) => setSlice(e.target.value)}>
                <option value="all">All registered worlds</option>
                {Object.keys(evidence.slices).map((name) => (
                  <option key={name} value={name}>
                    {name.replaceAll("_", " ")}
                  </option>
                ))}
              </select>
            </label>
          )}
          <div className="table-scroll">
            <table>
              <caption>
                {selected?.independent_worlds} worlds; reference{" "}
                {evidence.reference.replaceAll("_", " ")}
              </caption>
              <thead>
                <tr>
                  <th scope="col">Representation / training</th>
                  <th scope="col">Seeds</th>
                  <th scope="col">Finished / model episodes</th>
                  <th scope="col">Mean cost (s)</th>
                  <th scope="col">Difference / joint 95% interval</th>
                  <th scope="col">Between-training SD (s)</th>
                  <th scope="col">CVaR95 mixture (s)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(selected?.learned_groups ?? {}).map(
                  ([name, r]) => (
                    <tr key={name}>
                      <th scope="row">{name.replaceAll("_", " ")}</th>
                      <td>{r.training_seeds}</td>
                      <td>
                        {Object.entries(outcomes).some(([v]) =>
                          v.startsWith(name + "/seed-"),
                        )
                          ? `${Object.entries(outcomes)
                              .filter(([v]) => v.startsWith(name + "/seed-"))
                              .reduce(
                                (n, [, p]) => n + p.completions,
                                0,
                              )}/${r.training_seeds * r.independent_worlds}`
                          : "—"}
                      </td>
                      <td>{r.mean_cost_s.toFixed(1)}</td>
                      <td>
                        {r.paired_mean_difference_s.toFixed(1)} [
                        {r.hierarchical_paired_mean_95
                          .map((v) => v.toFixed(1))
                          .join(", ")}
                        ]
                      </td>
                      <td>
                        {r.between_training_sd_mean_cost_s?.toFixed(1) ??
                          "one seed"}
                      </td>
                      <td>{r.cvar95_mixture_s.toFixed(1)}</td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
          {Object.keys(outcomes).length > 0 && (
            <details>
              <summary>Completion and failure counts for every model</summary>
              <div className="table-scroll">
                <table>
                  <caption>
                    Individual models and classical references; all sailing
                    outcomes retained
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">Policy / seed</th>
                      <th scope="col">Finished</th>
                      <th scope="col">Mean cost (s)</th>
                      <th scope="col">Failures</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(outcomes).map(([name, p]) => (
                      <tr key={name}>
                        <th scope="row">{name.replaceAll("_", " ")}</th>
                        <td>
                          {p.completions}/{p.worlds}
                        </td>
                        <td>{p.mean_cost_s.toFixed(1)}</td>
                        <td>
                          {Object.entries(p.failure_categories ?? {})
                            .map(
                              ([kind, n]) =>
                                `${kind.replaceAll("_", " ")}: ${n}`,
                            )
                            .join("; ") || "none"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
          {evidence.curves.length > 0 && curve && (
            <>
              <label className="learning-select">
                Training diagnostic
                <select
                  value={curveIndex}
                  onChange={(e) => setCurveIndex(Number(e.target.value))}
                >
                  {evidence.curves.map((c, i) => (
                    <option value={i} key={c.variant}>
                      {c.variant.replaceAll("_", " ")}
                    </option>
                  ))}
                </select>
              </label>
              <svg
                className="learning-curve"
                viewBox="0 0 660 210"
                role="img"
                aria-label={`Training cost curve for ${curve.variant}`}
              >
                <path d="M50 20V170H640" fill="none" stroke="#617b7e" />
                <polyline
                  fill="none"
                  stroke="#14776e"
                  strokeWidth="3"
                  points={curve.points
                    .map(
                      (p) =>
                        `${50 + (p.steps / (curve.points.at(-1)?.steps ?? 1)) * 590},${170 - (p.mean_cost_s / 2400) * 150}`,
                    )
                    .join(" ")}
                />
                <text x="2" y="24">
                  2400 s
                </text>
                <text x="20" y="174">
                  0
                </text>
                <text x="220" y="200">
                  Training decisions · rolling episode cost
                </text>
              </svg>
              <p className="fineprint">
                Training worlds and exploratory actions only. This curve cannot
                establish out-of-sample benefit.
              </p>
            </>
          )}
          <ul className="fineprint">
            {evidence.limitations.map((limit) => (
              <li key={limit}>{limit}</li>
            ))}
          </ul>
          {evidence.lock_hash && (
            <p className="fineprint">
              Policy/scenario lock:{" "}
              <code>{evidence.lock_hash.slice(0, 16)}…</code>
            </p>
          )}
          <a href="/learning.json" download="sailing-learning-results.json">
            Download complete learning results ↗
          </a>
        </>
      ) : (
        <p role="status">{message}</p>
      )}
    </section>
  );
}
