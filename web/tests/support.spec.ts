import { test, expect } from "@playwright/test";

// Small UI contract fixtures, not observations or research results.
const point = (row: number, time: number, segment: number) => ({
  row,
  timestamp_utc: `2026-01-01T12:00:${String(time).padStart(2, "0")}Z`,
  latitude_deg: 50.7,
  longitude_deg: -1.3,
  east_m: row * 2,
  north_m: row * 3,
  segment,
  accepted: true,
  flags: [],
  heading_deg: null,
  wind_from_deg: null,
  wind_speed_m_s: null,
  current_east_m_s: null,
  current_north_m_s: null,
  speed_water_m_s: null,
  device_speed_over_ground_m_s: null,
  derived_speed_over_ground_m_s: row === 1 ? 3.6 : null,
});
const track = {
  schema_version: "track-v1",
  source_sha256: "a".repeat(64),
  source_format: "csv",
  session_id: "browser-contract-fixture",
  evidence: "synthetic_fixture",
  imported_utc: "2026-01-02T12:00:00Z",
  config: {
    nominal_interval_s: 1,
    gap_s: 10,
    max_ground_speed_m_s: 15,
    max_radius_m: 100000,
    max_bytes: 5000000,
    max_rows: 10000,
  },
  frame: { latitude_deg: 50.7, longitude_deg: -1.3 },
  measured: ["UTC timestamp", "GPS latitude/longitude"],
  points: [
    point(0, 0, 0),
    point(1, 1, 0),
    {
      ...point(2, 1, 0),
      accepted: false,
      segment: null,
      flags: ["duplicate_time"],
    },
    point(3, 2, 1),
    { ...point(4, 20, 2), flags: ["gap_before"] },
  ],
};
const weather = {
  schema_version: "weather-v1",
  source_evidence: "synthetic_fixture",
  request: {
    kind: "single_run",
    latitude_deg: 50.7,
    longitude_deg: -1.3,
    model: "fixture_model",
    declared_spatial_resolution_m: 9000,
    run_initialisation_utc: "2026-01-01T00:00:00Z",
    assumed_publication_lag_s: 28800,
  },
  source_url: "https://single-runs-api.open-meteo.com/v1/forecast",
  source_sha256: "b".repeat(64),
  retrieved_utc: "2026-01-02T00:00:00Z",
  available_utc: "2026-01-01T08:00:00Z",
  availability: "assumed_lag",
  returned_latitude_deg: 50.75,
  returned_longitude_deg: -1.25,
  points: [
    { valid_utc: "2026-01-01T08:00:00Z", wind_speed_m_s: 6, wind_from_deg: 0 },
    { valid_utc: "2026-01-01T09:00:00Z", wind_speed_m_s: 5, wind_from_deg: 10 },
  ],
};
const timing = {
  schema_version: "timing-v1",
  evidence: "synthetic_fixture",
  config: {},
  source_hashes: [track.source_sha256],
  split_sha256: "c".repeat(64),
  training_days: ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
  holdout_days: ["2026-01-05", "2026-01-06"],
  transitions: [],
  fitted_mean_s: 13,
  fitted_mean_day_bootstrap_95_s: [11, 15],
  holdout_mae_s: 6,
  holdout_mae_day_bootstrap_95_s: [5, 7],
};

test.beforeEach(async ({ page }) => {
  await page.route("**/support/track.json", (route) =>
    route.fulfill({ json: track }),
  );
  await page.route("**/support/weather.json", (route) =>
    route.fulfill({ json: weather }),
  );
  await page.route("**/support/timing.json", (route) =>
    route.fulfill({ json: timing }),
  );
  await page.goto("/");
});

test("quality breaks are drawn separately and timing stays labelled synthetic", async ({
  page,
}) => {
  await page
    .getByRole("button", { name: "Load synthetic support fixture" })
    .click();
  await expect(
    page.getByText("4 accepted / 1 excluded fixes · 3 connected segments"),
  ).toBeVisible();
  await expect(page.getByTestId("observed-segment")).toHaveCount(3);
  await expect(
    page.getByRole("heading", { name: "Held-out synthetic timing" }),
  ).toBeVisible();
  await expect(page.getByText("duplicate_time", { exact: true })).toBeVisible();
  const download = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Download processed private track" })
    .click();
  expect((await download).suggestedFilename()).toBe("private-track.json");
});

test("forecast initialisation cannot reveal an unpublished run", async ({
  page,
}) => {
  await page
    .getByRole("button", { name: "Load synthetic support fixture" })
    .click();
  await expect(
    page.getByText("Run available under the declared publication evidence."),
  ).toBeVisible();
  await page
    .getByRole("textbox", { name: "Historical decision UTC" })
    .fill("2026-01-01T07:59:59Z");
  await expect(
    page.getByText("Run not yet available at this decision."),
  ).toBeVisible();
  await expect(
    page
      .getByRole("table")
      .filter({
        has: page.getByText("Available forecast values", { exact: true }),
      })
      .locator("tbody tr"),
  ).toHaveCount(0);
});

test("private processed import stays in browser and does not inherit fixture calibration", async ({
  page,
}) => {
  await page
    .getByRole("button", { name: "Load synthetic support fixture" })
    .click();
  await expect(
    page.getByRole("heading", { name: "Held-out synthetic timing" }),
  ).toBeVisible();
  let imports = 0;
  await page.route("http://127.0.0.1:8787/**", () => {
    imports++;
  });
  await page.getByLabel("Open track", { exact: true }).setInputFiles({
    name: "private.json",
    mimeType: "application/json",
    buffer: Buffer.from(
      JSON.stringify({ ...track, evidence: "unvalidated_measurement" }),
    ),
  });
  await expect(
    page.getByRole("heading", {
      name: "Imported track · unvalidated measurement",
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Held-out synthetic timing" }),
  ).toHaveCount(0);
  expect(imports).toBe(0);
});

test("forged connected paths and retrospective availability are rejected", async ({
  page,
}) => {
  const invalid = {
    ...track,
    points: track.points.map((p) => (p.row === 3 ? { ...p, segment: 0 } : p)),
  };
  await page.getByLabel("Open track", { exact: true }).setInputFiles({
    name: "invalid.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(invalid)),
  });
  await expect(
    page
      .getByRole("alert")
      .filter({ hasText: "Track connects an invalid point" }),
  ).toBeVisible();
  const context = {
    ...weather,
    request: { ...weather.request, kind: "reanalysis" },
  };
  await page.getByLabel("Open processed weather").setInputFiles({
    name: "weather.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(context)),
  });
  await expect(
    page
      .getByRole("alert")
      .filter({ hasText: "Context products cannot be relabelled" }),
  ).toBeVisible();
});
