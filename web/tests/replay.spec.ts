import { test, expect } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.route("**/catalogue.json", (route) =>
    route.fulfill({ status: 404 }),
  );
});

// UI contract fixtures only: these numbers are not experimental measurements.
const learningFixture = {
  stage: "locked_learning",
  reference: "B1_belief",
  independent_worlds: 1000,
  episodes: 7000,
  lock_hash: "a".repeat(64),
  learned_groups: {
    PPO_belief_fixed: {
      training_seeds: 5,
      independent_worlds: 1000,
      mean_cost_s: 700,
      paired_mean_difference_s: 100,
      hierarchical_paired_mean_95: [-10, 210],
      between_training_sd_mean_cost_s: 80,
      cvar95_mixture_s: 2400,
      tail_independent_worlds: 50,
    },
  },
  curves: [
    {
      variant: "PPO_belief_fixed/seed-1",
      points: [
        { steps: 1024, mean_cost_s: 1800 },
        { steps: 2048, mean_cost_s: 1200 },
      ],
    },
  ],
  limitations: [
    "Synthetic evidence; subgroup tail estimates remain uncertain.",
  ],
  slices: {
    nominal: {
      independent_worlds: 1000,
      learned_groups: {
        PPO_belief_fixed: {
          training_seeds: 5,
          independent_worlds: 1000,
          mean_cost_s: 700,
          paired_mean_difference_s: 100,
          hierarchical_paired_mean_95: [-10, 210],
          between_training_sd_mean_cost_s: 80,
          cvar95_mixture_s: 2400,
          tail_independent_worlds: 50,
        },
      },
    },
  },
};

test("learning evidence separates training curves, worlds and training seeds", async ({
  page,
}) => {
  await page.route("**/learning.json", (route) =>
    route.fulfill({ json: learningFixture }),
  );
  await page.goto("/");
  const panel = page.getByRole("region", {
    name: "Can a learned policy justify its complexity?",
  });
  await expect(panel).toContainText("1000 independent worlds, 7000 episodes");
  await panel.getByLabel("Evaluation slice").selectOption("nominal");
  await expect(panel).toContainText("100.0 [-10.0, 210.0]");
  await expect(
    panel.getByRole("img", { name: /Training cost curve/ }),
  ).toBeVisible();
  await expect(panel).toContainText(
    "Training worlds and exploratory actions only",
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});

test("locked evidence without a lock is rejected", async ({ page }) => {
  await page.route("**/learning.json", (route) =>
    route.fulfill({ json: { ...learningFixture, lock_hash: undefined } }),
  );
  await page.goto("/");
  await expect(
    page.getByRole("status").filter({ hasText: "missing lock" }),
  ).toBeVisible();
});

test("learning evidence rejects impossible completion counts", async ({
  page,
}) => {
  await page.route("**/learning.json", (route) =>
    route.fulfill({
      json: {
        ...learningFixture,
        policies: {
          B1_belief: { worlds: 10, completions: 11, mean_cost_s: 600 },
        },
      },
    }),
  );
  await page.goto("/");
  await expect(
    page
      .getByRole("status")
      .filter({ hasText: "Inconsistent learning evidence" }),
  ).toBeVisible();
});

test("recorded replay supports play, pause, scrub, reset, truth and download", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "A recorded route, step by step" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Play", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Pause", exact: true }),
  ).toBeVisible();
  await expect
    .poll(async () =>
      Number(
        await page.getByRole("slider", { name: "Replay time" }).inputValue(),
      ),
    )
    .toBeGreaterThan(1);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await page.getByRole("slider", { name: "Replay time" }).fill("250");
  await expect(page.getByRole("heading", { name: "At 4:10" })).toBeVisible();
  await page.getByRole("checkbox", { name: "Show diagnostic truth" }).check();
  await expect(
    page.getByText("DIAGNOSTIC TRUTH · recorded offline"),
  ).toBeVisible();
  await page.getByRole("slider", { name: "Replay time" }).focus();
  await page.keyboard.press("ArrowRight");
  await expect
    .poll(async () =>
      Number(
        await page.getByRole("slider", { name: "Replay time" }).inputValue(),
      ),
    )
    .toBeGreaterThan(250);
  await page.getByRole("button", { name: "Reset" }).click();
  await expect(page.getByRole("slider", { name: "Replay time" })).toHaveValue(
    "0",
  );
  await expect(
    page.getByRole("link", { name: "Download recorded replay" }),
  ).toHaveAttribute("href", "/replay.json");
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  );
  expect(overflow).toBe(false);
  await page.screenshot({
    path: `test-results/replay-${test.info().project.name}.png`,
    fullPage: true,
  });
});

test("missing results show useful recovery instructions", async ({ page }) => {
  await page.route("**/replay.json", (route) =>
    route.fulfill({ status: 404, body: "Missing" }),
  );
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("uv run sail-lab demo");
});

test("invalid artefacts are rejected", async ({ page }) => {
  await page.route("**/replay.json", (route) =>
    route.fulfill({ json: { samples: [] } }),
  );
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("unsupported schema");
});

// Contract-test values only; this fixture is never exported as research evidence.
test("precomputed development results retain failures and paired intervals", async ({
  page,
}) => {
  const baseline = {
    worlds: 32,
    completions: 30,
    mean_cost_s: 601,
    cvar95_s: 2400,
    tail_effective_outcomes: 1.6,
    paired_mean_difference_s: 0,
    paired_mean_difference_95: [0, 0],
    decision_latency_p95_s: 0.001,
  };
  await page.route("**/results.json", (route) =>
    route.fulfill({
      json: {
        stage: "development_pilot",
        reference: "B1_belief",
        policies: {
          B1_belief: baseline,
          B3_risk: {
            ...baseline,
            completions: 28,
            mean_cost_s: 702,
            paired_mean_difference_s: 101,
            paired_mean_difference_95: [40, 160],
            decision_latency_p95_s: 0.1,
          },
        },
      },
    }),
  );
  await page.goto("/");
  await expect(
    page.getByRole("cell", { name: "30/32", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("cell", { name: "28/32", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("cell", { name: "101.0 [40.0, 160.0]", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Tail estimates are unstable.", { exact: false }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth,
    ),
  ).toBe(false);
});

test("unsupported results stages are rejected without disabling replay", async ({
  page,
}) => {
  await page.route("**/results.json", (route) =>
    route.fulfill({ json: { stage: "final", policies: {} } }),
  );
  await page.goto("/");
  await expect(
    page.getByText(
      "Comparison artefact has an invalid development-results contract.",
    ),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Play", exact: true }),
  ).toBeVisible();
});
