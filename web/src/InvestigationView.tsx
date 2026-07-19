import { useEffect, useState } from "react";
import Ajv from "ajv";
import schema from "../investigation.schema.json";
import type { InvestigationCatalogue } from "./generated/investigation";

const validate = new Ajv({ strict: false }).compile<InvestigationCatalogue>(
  schema,
);
const names = {
  fixed_script: "Fixed audit script",
  local_assistant: "Local assistant",
  researcher_critic: "Researcher + critic",
};

export function InvestigationView() {
  const [data, setData] = useState<InvestigationCatalogue | null>(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState("");
  useEffect(() => {
    const abort = new AbortController();
    fetch("/investigations.json", { signal: abort.signal })
      .then(async (response) => {
        if (!response.ok)
          throw new Error("No completed investigation audit is published yet.");
        const raw = await response.text();
        if (raw.length > 2_000_000)
          throw new Error("Investigation evidence exceeds the size limit.");
        const value: unknown = JSON.parse(raw);
        if (!validate(value))
          throw new Error("Investigation evidence has an unsupported schema.");
        const ids = new Set(value.tasks.map((row) => row.task_id));
        const keys = new Set(
          value.tasks.map((row) => `${row.task_id}/${row.strategy}`),
        );
        if (
          value.strategies.length !== 3 ||
          new Set(value.strategies.map((row) => row.name)).size !== 3 ||
          keys.size !== value.tasks.length ||
          value.tasks.length !== ids.size * 3 ||
          value.strategies.some((summary) => {
            const rows = value.tasks.filter(
              (row) => row.strategy === summary.name,
            );
            return (
              rows.length !== summary.tasks ||
              rows.filter((row) => row.accepted_report).length !==
                summary.accepted_reports ||
              rows.filter((row) => row.exact_issue_set).length !==
                summary.exact_issue_sets
            );
          })
        )
          throw new Error(
            "Investigation evidence has missing or inconsistent task denominators.",
          );
        setData(value);
        setSelected(value.tasks[0]?.task_id ?? "");
      })
      .catch((reason: unknown) => {
        if (!abort.signal.aborted)
          setError(
            reason instanceof Error ? reason.message : "Audit could not load.",
          );
      });
    return () => abort.abort();
  }, []);
  const ids = data ? [...new Set(data.tasks.map((row) => row.task_id))] : [];
  return (
    <section
      className="investigation-section"
      aria-label="Investigation assistant audit"
    >
      <div className="section-heading">
        <div>
          <p className="eyebrow">06 / INVESTIGATION AUDIT</p>
          <h2>Does an assistant earn its overhead?</h2>
        </div>
        <span className="evidence-badge">Synthetic audit tasks</span>
      </div>
      <p>
        A bounded local model investigates evidence through private MCP tools.
        These tasks test research judgement and tool use; they do not measure
        sailing performance. The fixed script remains a strong,
        model-independent control.
      </p>
      {error && <p role="status">{error}</p>}
      {data && (
        <>
          <p>
            {data.model_repository} · 4-bit MLX · a sequential critic reuses the
            same model. All {ids.length} tasks remain in each denominator,
            including rejected reports.
          </p>
          <div className="table-scroll">
            <table>
              <caption>
                Frozen held-out investigation cases · exact detection and
                observed cost
              </caption>
              <thead>
                <tr>
                  <th>Workflow</th>
                  <th>Exact issue sets</th>
                  <th>Accepted reports</th>
                  <th>False accusations</th>
                  <th>Missed issues</th>
                  <th>Unsupported structured claims</th>
                  <th>Wall time</th>
                  <th>Generated tokens</th>
                </tr>
              </thead>
              <tbody>
                {data.strategies.map((row) => (
                  <tr key={row.name}>
                    <th scope="row">{names[row.name]}</th>
                    <td>
                      {row.exact_issue_sets} / {row.tasks}
                    </td>
                    <td>
                      {row.accepted_reports} / {row.tasks}
                    </td>
                    <td>{row.false_accusations}</td>
                    <td>{row.missed_issues}</td>
                    <td>{row.unsupported_structured_claims}</td>
                    <td>{row.wall_s.toFixed(1)} s</td>
                    <td>{row.generated_tokens.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            Numerical fields are computed by the evidence tools. “Accepted”
            means the report passed citation and claim checks; it does not prove
            that every relevant issue was found. Free-text explanations remain
            unverified proposals. Paid inference service cost: $0; electricity
            and hardware cost are not estimated.
          </p>
          <label className="investigation-select">
            Inspect a labelled task
            <select
              aria-label="Investigation task"
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
            >
              {ids.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </label>
          <div className="investigation-cases">
            {data.tasks
              .filter((row) => row.task_id === selected)
              .map((row) => (
                <article key={row.strategy}>
                  <h3>{names[row.strategy]}</h3>
                  <p>
                    <strong>
                      {row.status === "succeeded"
                        ? "Report accepted"
                        : "Report rejected / failed"}
                    </strong>{" "}
                    ·{" "}
                    {row.exact_issue_set
                      ? "Exact issue set"
                      : "Detection differs from labels"}
                  </p>
                  <dl>
                    <dt>Known injected issues</dt>
                    <dd>{row.expected_issues.join(", ") || "None"}</dd>
                    <dt>Proposed issues</dt>
                    <dd>
                      {row.predicted_issues.join(", ") ||
                        (row.accepted_report
                          ? "None"
                          : "None / no completed report")}
                    </dd>
                    <dt>Citations</dt>
                    <dd>
                      {row.evidence_links_correct
                        ? "Validated"
                        : "Missing or invalid"}
                    </dd>
                  </dl>
                  {row.error && (
                    <p className="investigation-error">{row.error}</p>
                  )}
                  {row.explanation && (
                    <details>
                      <summary>Unverified model or script explanation</summary>
                      <p>{row.explanation}</p>
                    </details>
                  )}
                </article>
              ))}
          </div>
          <p>
            {data.limits} The fixture author knew how faults were constructed;
            this small suite is not a representative deployment sample. No
            prompt tuning used its evaluation outcomes.
          </p>
          <p className="provenance">
            Model revision <code>{data.model_revision}</code> · MCP{" "}
            {data.protocol_version} · inference and source versions are in the
            download.
          </p>
          <a href={data.evidence_url} download>
            Download full audit records and raw model outputs ↗
          </a>
        </>
      )}
    </section>
  );
}
