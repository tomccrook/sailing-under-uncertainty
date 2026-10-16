import { expect, test } from "@playwright/test";

const variants = ["B1_belief", "B2_mean", "B2_budget", "B3_mean", "B3_risk"];
function fixture() {
  const group = (name: string, n: number) => ({
    name,
    worlds: n,
    tail_effective_worlds: 0.05 * n,
    policies: variants.map((name) => ({
      name,
      worlds: n,
      finished: n - 1,
      deadline_exceeded: 0,
      boundary_violation: 1,
      mean_cost_s: 700,
      cvar95_s: 2400,
      decision_latency_mean_s: 0.01,
      decision_latency_p95_s: 0.02,
      model_steps: 40,
      model_step_calls_including_splits: 42,
      allocated_model_step_ceilings: 80,
      model_rollouts: 8,
      zero_failure_upper95: null,
    })),
    contrasts: variants.slice(0, 4).map((reference) => ({
      reference,
      mean_difference_s: -2,
      mean_difference_95: [-4, 1],
      cvar95_difference_s: 0,
      cvar95_difference_95: [-5, 5],
    })),
  });
  return {
    schema_version: 1,
    evidence_level: "independent_locked_synthetic_replication",
    lock_sha256: "0".repeat(64),
    original_lock_sha256: "1".repeat(64),
    numerical_source_sha256: "2".repeat(64),
    method_sha256: "3".repeat(64),
    report_sha256: "4".repeat(64),
    episodes_sha256: "5".repeat(64),
    episodes_url: "/replication/episodes.parquet",
    report_url: "/replication/report.json",
    total_worlds: 16,
    retained_episodes: 80,
    selected_deterministic_horizon_s: 120,
    evaluation_wall_s: 60,
    workers: 4,
    bootstrap_count: 200,
    analysis_seed: 885300,
    precision_target_met: false,
    cohorts: [
      group("nominal", 8),
      group("sensor_delay", 4),
      group("delay_tack", 4),
    ],
    nominal_families: ["oscillating", "ou", "regime", "spatial"].map((f) =>
      group(f, 2),
    ),
    limitations: [
      "Synthetic browser fixture; these numbers are not measured results.",
    ],
  };
}

test("replication retains all controllers and separates weak stress tails", async ({
  page,
}) => {
  await page.route("**/replication.json", (route) =>
    route.fulfill({ json: fixture() }),
  );
  await page.goto("/");
  const panel = page.getByRole("region", {
    name: "Independent planner replication",
  });
  await expect(panel.getByText(/all 80 episodes/)).toBeVisible();
  await expect(panel.getByText(/actual work and runtime differ/)).toBeVisible();
  await expect(
    panel.getByText(/nominal replication sample target was not reached/),
  ).toBeVisible();
  await panel.getByLabel("Replication cohort").selectOption("delay_tack");
  await expect(panel.getByText(/Small tail cohort/)).toBeVisible();
  await expect(
    panel.getByRole("row", { name: /Tuned B1 belief 700/ }),
  ).toBeVisible();
  await panel.getByText("Source identities and full evidence").click();
  await expect(
    panel.getByRole("link", { name: /Download all replication episodes/ }),
  ).toHaveAttribute("href", "/replication/episodes.parquet");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});

test("replication refuses missing paired outcomes", async ({ page }) => {
  const value = fixture();
  value.retained_episodes = 79;
  await page.route("**/replication.json", (route) =>
    route.fulfill({ json: value }),
  );
  await page.goto("/");
  const panel = page.getByRole("region", {
    name: "Independent planner replication",
  });
  await expect(panel.getByRole("status")).toContainText(
    "missing or inconsistent paired denominators",
  );
  await expect(panel.getByLabel("Replication cohort")).toHaveCount(0);
});
