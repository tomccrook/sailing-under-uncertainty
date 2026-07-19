import { createHash } from "node:crypto";
import { test, expect } from "@playwright/test";

// Contract fixtures only; these values are not scientific findings.
const policy = {
  worlds: 100,
  completions: 98,
  mean_cost_s: 600,
  cvar95_s: 1500,
  paired_mean_difference_s: 0,
  paired_mean_difference_95: [0, 0],
  decision_latency_p95_s: 0.1,
};
const group = {
  training_seeds: 5,
  independent_worlds: 100,
  mean_cost_s: 700,
  cvar95_mixture_s: 1700,
  paired_mean_difference_s: 100,
  hierarchical_paired_mean_95: [20, 180],
  between_training_sd_mean_cost_s: 40,
  tail_independent_worlds: 5,
};
const evidence = {
  stage: "locked_learning",
  lock_hash: "a".repeat(64),
  reference: "B1_belief",
  independent_worlds: 100,
  episodes: 600,
  curves: [],
  limitations: ["Contract fixture"],
  learned_groups: { PPO_belief_fixed: group },
  slices: {
    nominal: {
      independent_worlds: 100,
      policies: { B1_belief: policy },
      learned_groups: { PPO_belief_fixed: group },
    },
  },
  classical_cohort: {
    independent_worlds: 100,
    policies: {
      B1_belief: policy,
      B2_mean: { ...policy, mean_cost_s: 620 },
      B3_risk: { ...policy, mean_cost_s: 590 },
    },
    paired_vs_B2: {
      policies: {
        B3_risk: {
          ...policy,
          paired_mean_difference_s: -30,
          paired_mean_difference_95: [-55, -5],
        },
      },
    },
  },
  compute: {
    wall_s: 50,
    workers: 4,
    device: "cpu",
    latency_condition: "contract fixture measurement",
  },
};
test("trade-offs keep cohort counts, failure scores and compute definitions", async ({
  page,
}) => {
  await page.route("**/catalogue.json", (route) =>
    route.fulfill({ status: 404 }),
  );
  await page.route("**/learning.json", (route) =>
    route.fulfill({ json: evidence }),
  );
  await page.goto("/");
  const panel = page.getByRole("region", {
    name: "Complexity has to earn its cost",
  });
  await expect(panel).toContainText("100.0 s more");
  await expect(
    panel.getByRole("img", { name: /Recorded expected and tail/ }),
  ).toBeVisible();
  await panel.getByLabel("Trade-off cohort").selectOption("classical");
  await panel.getByLabel("Highlight policy").selectOption("B3_risk");
  await expect(panel).toContainText("B3 risk: mean 590.0 s, CVaR95 1500.0 s");
  await expect(panel).toContainText("-30.0 [-55.0, -5.0] s");
  await expect(panel).toContainText("5.0 tail outcomes");
  await expect(panel).toContainText("No matched-compute superiority claim");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});

test("trade-offs reject impossible aggregate outcomes", async ({ page }) => {
  await page.route("**/catalogue.json", (route) =>
    route.fulfill({ status: 404 }),
  );
  await page.route("**/learning.json", (route) =>
    route.fulfill({
      json: {
        ...evidence,
        classical_cohort: {
          ...evidence.classical_cohort,
          policies: { B1_belief: { ...policy, completions: 101 } },
        },
      },
    }),
  );
  await page.goto("/");
  await expect(
    page
      .getByRole("status")
      .filter({ hasText: "Inconsistent locked trade-off evidence" }),
  ).toBeVisible();
});

for (const corrupt of [false, true])
  test(`calibration ${corrupt ? "rejects invalid coverage" : "separates filter and forecast coverage"}`, async ({
    page,
  }) => {
    const estimate = {
      mean: 0.9,
      world_bootstrap_95: [0.8, 0.95],
      independent_worlds: 8,
    };
    const calibration = {
      stage: "development",
      duration_s: 600,
      burn_in_s: 60,
      worlds_per_case: 8,
      estimator: {
        "ou/noisy": {
          coverage: corrupt ? { ...estimate, mean: 1.2 } : estimate,
          direction_rmse_deg: {
            ...estimate,
            mean: 3,
            world_bootstrap_95: [2, 4],
          },
          raw_direction_rmse_deg: {
            ...estimate,
            mean: 6,
            world_bootstrap_95: [5, 7],
          },
        },
      },
      forecast_direction_coverage: {
        "ou/nominal/60s": estimate,
        "ou/biased/60s": {
          ...estimate,
          mean: 0.4,
          world_bootstrap_95: [0.3, 0.5],
        },
      },
    };
    const body = JSON.stringify(calibration);
    await page.route("**/catalogue.json", (route) =>
      route.fulfill({
        json: {
          version: 1,
          stage: "locked_diagnostic",
          lock_hash: "a".repeat(64),
          evaluation_manifest_hash: "b".repeat(64),
          default_scenario_id: "fixture",
          documents: {},
          calibration: {
            url: "/demo/calibration.json",
            sha256: createHash("sha256").update(body).digest("hex"),
          },
          scenarios: [
            {
              scenario_id: "fixture",
              label: "Contract fixture",
              selection_reason: "Not research evidence",
              slice: "nominal",
              world_hash: "c".repeat(64),
              pairing_hash: "d".repeat(64),
              recordings: ["B1_belief", "B3_risk"].map((variant) => ({
                variant,
                run_id: variant,
                status: "finished",
                elapsed_s: 500,
                cost_s: 500,
                replay: { url: "/demo/missing.json", sha256: "e".repeat(64) },
                details: { url: "/demo/missing.json", sha256: "e".repeat(64) },
                manifest: { url: "/demo/missing.json", sha256: "e".repeat(64) },
              })),
            },
          ],
        },
      }),
    );
    await page.route("**/demo/calibration.json", (route) =>
      route.fulfill({ contentType: "application/json", body }),
    );
    await page.route("**/demo/missing.json", (route) =>
      route.fulfill({ status: 404 }),
    );
    await page.goto("/");
    const panel = page.getByRole("region", {
      name: "Is the wind uncertainty calibrated?",
    });
    if (corrupt)
      await expect(panel).toContainText(
        "Inconsistent calibration intervals or counts",
      );
    else {
      await expect(panel).toContainText("3.0 [2.0, 4.0]°");
      await panel
        .getByLabel("Forecast case and lead")
        .selectOption("ou/biased/60s");
      await expect(panel).toContainText("40.0 [30.0, 50.0]%");
      await expect(panel).toContainText(
        "separate from locked policy comparisons",
      );
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth > innerWidth,
        ),
      ).toBe(false);
    }
  });
