import { expect, test } from "@playwright/test";

const variants = [
  "B1_belief",
  "B2_matched",
  "B3_mean_matched",
  "B3_risk_matched",
  "B3_risk_original",
];
function workFixture() {
  const group = (name: string, worlds: number) => ({
    name,
    worlds,
    tail_effective_worlds: worlds * 0.05,
    matching_tolerance: 0.1,
    work_matching_passed: true,
    policies: variants.map((name) => ({
      name,
      finished: worlds - 1,
      boundary_violation: 1,
      deadline_exceeded: 0,
      mean_cost_s: 1000,
      cvar95_s: 2400,
      steps: 100,
      steps_per_decision: 100,
      recursive_step_calls: 102,
      decision_latency_p95_s: 0.02,
      latency_mean_s: 0.01,
      zero_failure_upper95: null,
    })),
    contrasts: variants
      .filter((name) => name !== "B3_risk_matched")
      .map((reference) => ({
        reference,
        mean_difference_s: -2,
        mean_difference_95: [-4, 1],
        cvar95_difference_s: 0,
        cvar95_difference_95: [-1, 1],
      })),
  });
  return {
    version: "measured-work-publication-v1",
    evidence_level: "locked_synthetic_measured_work_and_new_processes",
    lock_sha256: "0".repeat(64),
    report_sha256: "1".repeat(64),
    episodes_sha256: "2".repeat(64),
    worlds: 2048,
    episodes: 10240,
    horizon_s: 130,
    matched_samples: 2,
    wall_s: 60,
    cohorts: [group("nominal", 1024), group("new_family", 1024)],
    families: ["oscillating", "ou", "regime", "spatial"]
      .map((f) => group(f, 256))
      .concat([group("gust_front", 512), group("persistent_trend", 512)]),
    limitations: ["Synthetic browser fixture; these are not measured results."],
  };
}
function recordedFixture() {
  return {
    version: "recorded-publication-v1",
    evidence_level: "recorded_GPS_and_buoy_component_validation",
    registration_sha256: "0".repeat(64),
    report_sha256: "1".repeat(64),
    raw_gps_rows: 100,
    excluded_gps_rows: 4,
    accepted_gps_rows: 90,
    rejected_gps_rows: 6,
    registered_train_identifiers: 3,
    registered_heldout_identifiers: 2,
    train_voyages: 2,
    heldout_voyages: 2,
    projection_heldout_pairs: 20,
    projection_heldout_p99_relative: 0.0001,
    projection_gate_passed: true,
    wind_train_days: 60,
    wind_heldout_days: 30,
    wind_heldout_pairs: 100,
    predictors: [
      { name: "AR1", score: 0.8 },
      { name: "persistence", score: 0.81 },
      { name: "climatology", score: 7 },
    ].map(({ name, score }) => ({
      name,
      heldout_pairs: 100,
      vector_rmse_m_s: score,
      speed_rmse_m_s: 0.6,
      direction_rmse_deg_at_speed_ge_0_5: 10,
      coverage95: 0.94,
      gaussian_nll_mean: 2,
    })),
    vector_rmse_difference_m_s: -0.01,
    vector_rmse_difference95: [-0.02, 0.01],
    attribution: {
      sailing: "Illustrative fixture",
      wind: "Illustrative fixture",
    },
    limitations: ["Browser contract fixture only."],
    boat_calibrated: false,
  };
}

test("work comparison retains controllers, uncertainty and family tail limits", async ({
  page,
}) => {
  await page.route("**/compute-followup/*.svg", (route) =>
    route.fulfill({
      contentType: "image/svg+xml",
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="100"><text x="10" y="40">Illustrative contract fixture</text></svg>',
    }),
  );
  await page.route("**/compute-followup.json", (route) =>
    route.fulfill({ json: workFixture() }),
  );
  await page.goto("/");
  const panel = page.getByRole("region", {
    name: "Measured work and new wind families",
  });
  await expect(panel.getByText(/all 10,240 episodes/)).toBeVisible();
  await expect(
    panel.getByText("Measured integration-work match passes."),
  ).toBeVisible();
  await expect(
    panel.getByText(/retain their unresolved conclusion/),
  ).toBeVisible();
  await expect(
    panel.getByRole("row", { name: /B3 risk \(2 paths\) 1,000/ }),
  ).toBeVisible();
  await expect(panel.getByRole("img")).toHaveAttribute(
    "src",
    "/compute-followup/nominal.svg",
  );
  await panel.getByRole("img").scrollIntoViewIfNeeded();
  await panel
    .getByRole("img")
    .evaluate((img) => (img as HTMLImageElement).decode());
  const figure = panel.getByRole("region", {
    name: "Scrollable paired scientific figure",
  });
  await expect(figure).toHaveAttribute("tabindex", "0");
  expect(
    await panel.getByRole("img").evaluate((img) => img.clientWidth),
  ).toBeGreaterThanOrEqual(1100);
  if ((page.viewportSize()?.width ?? 1440) < 1100) {
    expect(await figure.evaluate((el) => el.scrollWidth > el.clientWidth)).toBe(
      true,
    );
    await figure.focus();
    await page.keyboard.press("ArrowRight");
    await expect
      .poll(() => figure.evaluate((el) => el.scrollLeft))
      .toBeGreaterThan(0);
  }
  await panel.getByLabel("Work comparison cohort").selectOption("new_family");
  await expect(panel.getByRole("img")).toHaveAttribute(
    "src",
    "/compute-followup/new_family.svg",
  );
  await panel.getByLabel("Work comparison cohort").selectOption("gust_front");
  await expect(panel.getByRole("img")).toHaveCount(0);
  await expect(panel.getByText(/Small family tail/)).toBeVisible();
  await expect(
    panel.getByText(/25.6 effective CVaR95 tail worlds/),
  ).toBeVisible();
  await panel.getByText("Registered sources and full evidence").click();
  await expect(
    panel.getByRole("link", { name: "Download every work-study episode" }),
  ).toHaveAttribute("href", "/compute-followup/episodes.parquet");
  await expect(
    panel.getByRole("link", { name: "Download paired figure PDF" }),
  ).toHaveAttribute("href", "/compute-followup/new_family.pdf");
  await panel.getByLabel("Work comparison cohort").selectOption("ou");
  await expect(
    panel.getByRole("link", { name: "Download paired figure PDF" }),
  ).toHaveAttribute("href", "/compute-followup/nominal.pdf");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});

test("work comparison refuses a false compute-matching claim", async ({
  page,
}) => {
  const value = workFixture();
  value.cohorts[0]!.policies[2]!.steps_per_decision = 200;
  await page.route("**/compute-followup.json", (route) =>
    route.fulfill({ json: value }),
  );
  await page.goto("/");
  const panel = page.getByRole("region", {
    name: "Measured work and new wind families",
  });
  await expect(panel.getByRole("alert")).toContainText("inconsistent evidence");
  await expect(panel.getByLabel("Work comparison cohort")).toHaveCount(0);
});

test("recorded components expose exclusions and limit claims to identified components", async ({
  page,
}) => {
  await page.route("**/empirical.json", (route) =>
    route.fulfill({ json: recordedFixture() }),
  );
  await page.goto("/");
  const panel = page.getByRole("region", {
    name: "Recorded component validation",
  });
  await expect(panel.getByText(/100 source GPS rows/)).toBeVisible();
  await expect(
    panel.getByText(/90 accepted, 6 quality-rejected, 4 excluded/),
  ).toBeVisible();
  await expect(
    panel.getByText(/No boat polar or tack dynamics is empirically calibrated/),
  ).toBeVisible();
  await expect(
    panel.getByText(/numerical consistency, not independent GPS accuracy/),
  ).toBeVisible();
  await expect(
    panel.getByText(/observed track, not a policy counterfactual/),
  ).toBeVisible();
  await expect(panel.getByRole("row", { name: /AR1 0.8/ })).toBeVisible();
  await panel.getByText("Recorded sources, limitations and downloads").click();
  await expect(
    panel.getByRole("link", { name: "Download processed public GPS" }),
  ).toHaveAttribute("href", "/empirical/gps-processed-points.parquet");
  await expect(
    panel.getByRole("link", { name: "Download attribution and data licence" }),
  ).toHaveAttribute("href", "/empirical/LICENSE-DATA.md");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});

test("recorded components refuse an unaccounted source row", async ({
  page,
}) => {
  const value = recordedFixture();
  value.excluded_gps_rows = 3;
  await page.route("**/empirical.json", (route) =>
    route.fulfill({ json: value }),
  );
  await page.goto("/");
  const panel = page.getByRole("region", {
    name: "Recorded component validation",
  });
  await expect(panel.getByRole("alert")).toContainText("inconsistent evidence");
  await expect(panel.getByRole("table")).toHaveCount(0);
});

test("unpublished optional evidence handles a static SPA fallback", async ({
  page,
}) => {
  for (const name of ["compute-followup", "empirical"]) {
    await page.route(`**/${name}.json`, (route) =>
      route.fulfill({
        body: "<!doctype html><html></html>",
        contentType: "text/html",
      }),
    );
  }
  await page.goto("/");
  for (const name of [
    "Measured work and new wind families",
    "Recorded component validation",
  ]) {
    const panel = page.getByRole("region", { name });
    await expect(panel.getByText(/No complete .* published yet/)).toBeVisible();
    await expect(panel.getByRole("alert")).toHaveCount(0);
  }
});
