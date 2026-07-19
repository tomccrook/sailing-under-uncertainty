import { useEffect, useState } from "react";
import Ajv from "ajv";

type PolicyResult = {
  worlds: number;
  completions: number;
  mean_cost_s: number;
  cvar95_s: number;
  tail_effective_outcomes: number;
  paired_mean_difference_s: number;
  paired_mean_difference_95: [number, number];
  decision_latency_p95_s: number;
};
type StudyResults = {
  stage: "development_pilot";
  reference: string;
  policies: Record<string, PolicyResult>;
  methodology_amendment?: { pilot_already_open?: boolean };
};
const validate = new Ajv().compile<StudyResults>({
  type: "object",
  required: ["stage", "reference", "policies"],
  properties: {
    stage: { const: "development_pilot" },
    reference: { type: "string" },
    methodology_amendment: {
      type: "object",
      properties: { pilot_already_open: { type: "boolean" } },
    },
    policies: {
      type: "object",
      minProperties: 1,
      additionalProperties: {
        type: "object",
        required: [
          "worlds",
          "completions",
          "mean_cost_s",
          "cvar95_s",
          "tail_effective_outcomes",
          "paired_mean_difference_s",
          "paired_mean_difference_95",
          "decision_latency_p95_s",
        ],
        properties: {
          worlds: { type: "integer", minimum: 1 },
          completions: { type: "integer", minimum: 0 },
          mean_cost_s: { type: "number", minimum: 0 },
          cvar95_s: { type: "number", minimum: 0 },
          tail_effective_outcomes: { type: "number", minimum: 0 },
          paired_mean_difference_s: { type: "number" },
          paired_mean_difference_95: {
            type: "array",
            items: { type: "number" },
            minItems: 2,
            maxItems: 2,
          },
          decision_latency_p95_s: { type: "number", minimum: 0 },
        },
      },
    },
  },
});

export function Results() {
  const [results, setResults] = useState<StudyResults | null>(null);
  const [message, setMessage] = useState(
    "Loading precomputed development comparisons…",
  );
  useEffect(() => {
    const abort = new AbortController();
    fetch("/results.json", { signal: abort.signal })
      .then(async (response) => {
        if (!response.ok) {
          setMessage(
            "No precomputed comparison loaded. The recorded replay remains available.",
          );
          return;
        }
        const value: unknown = await response.json();
        if (
          !validate(value) ||
          !Object.hasOwn(value.policies, value.reference) ||
          Object.values(value.policies).some(
            (r) =>
              r.completions > r.worlds ||
              r.worlds !== value.policies[value.reference]?.worlds ||
              r.cvar95_s + 1e-8 < r.mean_cost_s ||
              r.paired_mean_difference_95[0] > r.paired_mean_difference_95[1],
          )
        ) {
          throw new Error(
            "Comparison artefact has an invalid development-results contract.",
          );
        }
        setResults(value);
      })
      .catch((reason) => {
        if (!abort.signal.aborted)
          setMessage(
            reason instanceof Error
              ? reason.message
              : "Results could not be loaded.",
          );
      });
    return () => abort.abort();
  }, []);
  return (
    <section className="results-panel" aria-labelledby="results-title">
      <p className="eyebrow">02 / PRECOMPUTED DEVELOPMENT EVIDENCE</p>
      <h2 id="results-title">What the paired worlds show</h2>
      {results ? (
        <>
          <p>
            Same frozen wind worlds and instruments; lower cost is better.
            Failed sailing episodes cost 2400 s. Intervals resample paired
            worlds. These are development results from fixed policies.
          </p>
          {results.methodology_amendment?.pilot_already_open && (
            <p className="fineprint">
              B2 mean instruments were added after the first pilot was open. The
              original single-draw B2 remains in the table. This amendment is
              exploratory.
            </p>
          )}
          <div className="table-scroll">
            <table>
              <caption>
                Exploratory comparison; reference{" "}
                {results.reference.replaceAll("_", " ")}
              </caption>
              <thead>
                <tr>
                  <th scope="col">Policy</th>
                  <th scope="col">Finished</th>
                  <th scope="col">Mean cost (s)</th>
                  <th scope="col">Difference / 95% interval (s)</th>
                  <th scope="col">CVaR95 (s)</th>
                  <th scope="col">Compute p95 (ms)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(results.policies).map(([name, r]) => (
                  <tr key={name}>
                    <th scope="row">{name.replaceAll("_", " ")}</th>
                    <td>
                      {r.completions}/{r.worlds}
                    </td>
                    <td>{r.mean_cost_s.toFixed(1)}</td>
                    <td>
                      {r.paired_mean_difference_s.toFixed(1)} [
                      {r.paired_mean_difference_95
                        .map((v) => v.toFixed(1))
                        .join(", ")}
                      ]
                    </td>
                    <td>{r.cvar95_s.toFixed(1)}</td>
                    <td>{(r.decision_latency_p95_s * 1000).toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="fineprint">
            Only{" "}
            {results.policies[
              results.reference
            ]?.tail_effective_outcomes.toFixed(1)}{" "}
            effective outcomes in the 95% tail per policy. Tail estimates are
            unstable. This pilot does not establish the final research claim.
          </p>
          <a href="/results.json" download="sailing-development-results.json">
            Download complete development results ↗
          </a>
        </>
      ) : (
        <p role="status">{message}</p>
      )}
    </section>
  );
}
