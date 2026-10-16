import { useEffect, useState } from "react";
import Ajv from "ajv";
import { checkedJson, loadCatalogue } from "./evidence";
import type { Catalogue } from "./generated/catalogue";

type Estimate = {
  mean: number;
  world_bootstrap_95: [number, number];
  independent_worlds: number;
};
type Calibration = {
  stage: "development";
  duration_s: number;
  burn_in_s: number;
  worlds_per_case: number;
  estimator: Record<
    string,
    {
      coverage: Estimate;
      direction_rmse_deg: Estimate;
      raw_direction_rmse_deg: Estimate;
    }
  >;
  forecast_direction_coverage: Record<string, Estimate>;
};
const estimateSchema = {
  type: "object",
  required: ["mean", "world_bootstrap_95", "independent_worlds"],
  properties: {
    mean: { type: "number" },
    independent_worlds: { type: "integer", minimum: 1 },
    world_bootstrap_95: {
      type: "array",
      minItems: 2,
      maxItems: 2,
      items: { type: "number" },
    },
  },
};
const validate = new Ajv().compile<Calibration>({
  type: "object",
  required: [
    "stage",
    "duration_s",
    "burn_in_s",
    "worlds_per_case",
    "estimator",
    "forecast_direction_coverage",
  ],
  properties: {
    stage: { const: "development" },
    duration_s: { type: "number", minimum: 0 },
    burn_in_s: { type: "number", minimum: 0 },
    worlds_per_case: { type: "integer", minimum: 1 },
    estimator: {
      type: "object",
      minProperties: 1,
      additionalProperties: {
        type: "object",
        required: ["coverage", "direction_rmse_deg", "raw_direction_rmse_deg"],
        properties: {
          coverage: estimateSchema,
          direction_rmse_deg: estimateSchema,
          raw_direction_rmse_deg: estimateSchema,
        },
      },
    },
    forecast_direction_coverage: {
      type: "object",
      minProperties: 1,
      additionalProperties: estimateSchema,
    },
  },
});
const interval = (e: Estimate, percentage = false) => {
  const factor = percentage ? 100 : 1;
  return `${(e.mean * factor).toFixed(1)} [${e.world_bootstrap_95.map((v) => (v * factor).toFixed(1)).join(", ")}]${percentage ? "%" : "°"}`;
};

export function EvidenceContext() {
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null);
  const [calibration, setCalibration] = useState<Calibration | null>(null);
  const [estimatorCase, setEstimatorCase] = useState("ou/noisy");
  const [forecastCase, setForecastCase] = useState("ou/nominal/60s");
  const [error, setError] = useState("");
  useEffect(() => {
    const abort = new AbortController();
    loadCatalogue(abort.signal)
      .then(async (value) => {
        if (!value || abort.signal.aborted) return;
        setCatalogue(value);
        if (!value.calibration) return;
        const report = await checkedJson(value.calibration, abort.signal);
        if (!validate(report) || report.burn_in_s >= report.duration_s)
          throw new Error("Invalid calibration evidence.");
        const estimates = [
          ...Object.values(report.estimator).flatMap((e) => [
            e.coverage,
            e.direction_rmse_deg,
            e.raw_direction_rmse_deg,
          ]),
          ...Object.values(report.forecast_direction_coverage),
        ];
        if (
          estimates.some(
            (e) =>
              e.world_bootstrap_95[0] > e.world_bootstrap_95[1] ||
              e.independent_worlds !== report.worlds_per_case,
          ) ||
          [
            ...Object.values(report.estimator).map((e) => e.coverage),
            ...Object.values(report.forecast_direction_coverage),
          ].some(
            (e) =>
              e.mean < 0 ||
              e.mean > 1 ||
              e.world_bootstrap_95.some((v) => v < 0 || v > 1),
          )
        )
          throw new Error("Inconsistent calibration intervals or counts.");
        if (!abort.signal.aborted) {
          setCalibration(report);
          if (!("ou/noisy" in report.estimator))
            setEstimatorCase(Object.keys(report.estimator)[0]!);
          if (!("ou/nominal/60s" in report.forecast_direction_coverage))
            setForecastCase(
              Object.keys(report.forecast_direction_coverage)[0]!,
            );
        }
      })
      .catch((reason: unknown) => {
        if (!abort.signal.aborted)
          setError(
            reason instanceof Error
              ? reason.message
              : "Calibration unavailable.",
          );
      });
    return () => abort.abort();
  }, []);
  if (!catalogue) return null;
  const estimator = calibration?.estimator[estimatorCase];
  const forecast = calibration?.forecast_direction_coverage[forecastCase];
  return (
    <section className="results-panel" aria-labelledby="calibration-title">
      <p className="eyebrow">WHAT THE UNCERTAINTY CAN SUPPORT</p>
      <h2 id="calibration-title">Is the wind uncertainty calibrated?</h2>
      {error && <p role="status">{error}</p>}
      {calibration && estimator && forecast && (
        <>
          <p>
            Development diagnostics: {calibration.worlds_per_case} independent
            synthetic worlds per case, {calibration.duration_s} s each; first{" "}
            {calibration.burn_in_s} s excluded. Intervals resample whole worlds.
            These are separate from locked policy comparisons.
          </p>
          <div className="calibration-grid">
            <article>
              <label>
                Estimator case
                <select
                  value={estimatorCase}
                  onChange={(e) => setEstimatorCase(e.target.value)}
                >
                  {Object.keys(calibration.estimator).map((k) => (
                    <option key={k} value={k}>
                      {k.replaceAll("/", " · ")}
                    </option>
                  ))}
                </select>
              </label>
              <dl>
                <div>
                  <dt>Raw direction RMSE / 95% interval</dt>
                  <dd>{interval(estimator.raw_direction_rmse_deg)}</dd>
                </div>
                <div>
                  <dt>Filtered direction RMSE / 95% interval</dt>
                  <dd>{interval(estimator.direction_rmse_deg)}</dd>
                </div>
                <div>
                  <dt>Nominal 95% vector-region coverage / interval</dt>
                  <dd>{interval(estimator.coverage, true)}</dd>
                </div>
              </dl>
            </article>
            <article>
              <label>
                Forecast case and lead
                <select
                  value={forecastCase}
                  onChange={(e) => setForecastCase(e.target.value)}
                >
                  {Object.keys(calibration.forecast_direction_coverage).map(
                    (k) => (
                      <option key={k} value={k}>
                        {k.replaceAll("/", " · ")}
                      </option>
                    ),
                  )}
                </select>
              </label>
              <dl>
                <div>
                  <dt>Nominal 95% direction-interval coverage / interval</dt>
                  <dd>{interval(forecast, true)}</dd>
                </div>
                <div>
                  <dt>Independent worlds</dt>
                  <dd>{forecast.independent_worlds}</dd>
                </div>
              </dl>
              <p className="fineprint">
                Forecast calibration and local vector-estimator calibration are
                different quantities. Bias, delay and abrupt changes can
                undermine either model.
              </p>
            </article>
          </div>
        </>
      )}
      <div className="download-links">
        {Object.entries(catalogue.documents).map(([name, asset]) => (
          <a href={asset.url} download key={name}>
            {name.replaceAll("-", " ")} ↗
          </a>
        ))}
        {catalogue.calibration && (
          <a href={catalogue.calibration.url} download>
            Calibration results ↗
          </a>
        )}
        <a href="/catalogue.json" download>
          Evidence catalogue ↗
        </a>
      </div>
      <p className="fineprint">
        Lock <code>{catalogue.lock_hash}</code>. Synthetic polar and weather; no
        empirical boat calibration or financial profitability claim.
      </p>
    </section>
  );
}
