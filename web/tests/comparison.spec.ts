import { createHash } from "node:crypto";
import { test, expect, type Page } from "@playwright/test";
import replayFixture from "./fixtures/replay.json" with { type: "json" };
import detailsFixture from "./fixtures/details.json" with { type: "json" };
import type { Details } from "../src/generated/details";

// Browser contract values only. Never exported as research evidence.
async function installFixture(
  page: Page,
  fault: "none" | "hash" | "future" = "none",
) {
  const files: Record<string, string> = {};
  const asset = (url: string, value: unknown) => {
    const body = JSON.stringify(value);
    files[url] = body;
    return { url, sha256: createHash("sha256").update(body).digest("hex") };
  };
  const scenarios = ["fixture-first", "fixture-second"].map(
    (scenario_id, world) => ({
      scenario_id,
      label: `Contract world ${world + 1}`,
      slice: "nominal",
      selection_reason: "Contract fixture, not research evidence",
      world_hash: detailsFixture.world_hash,
      pairing_hash: "b".repeat(64),
      recordings: ["B3_risk", "B1_belief", "PPO_belief_fixed/seed-1"].map(
        (variant, i) => {
          const run_id = `${scenario_id}-policy-${i}`;
          const replay = structuredClone(replayFixture);
          replay.run_id = run_id;
          replay.policy_label = variant;
          replay.summary.policy_id = variant;
          const details = structuredClone(detailsFixture) as Details;
          details.run_id = run_id;
          details.variant = variant;
          if (i === 1)
            details.decisions.forEach((d) => {
              d.planning = null;
            });
          if (i === 2) {
            details.model_hash = "c".repeat(64);
            details.decisions.forEach((d) => {
              d.planning = null;
            });
            details.probabilities = details.decisions.map((d) => ({
              elapsed_s: d.elapsed_s,
              headings_rad: Array(8).fill(d.action.target_heading_rad),
              probabilities: [0.65, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
              selected_index: 0,
            }));
          }
          if (fault === "future" && i === 0)
            details.decisions[0]!.observation.elapsed_s = 99;
          const r = asset(`/demo/${run_id}/replay.json`, replay);
          if (fault === "hash" && i === 0) files[r.url] = "{}";
          return {
            variant,
            run_id,
            status: replay.summary.status,
            elapsed_s: replay.summary.elapsed_s,
            cost_s: 2400,
            replay: r,
            details: asset(`/demo/${run_id}/details.json`, details),
            manifest: asset(`/demo/${run_id}/manifest.json`, {}),
          };
        },
      ),
    }),
  );
  const catalogue = {
    version: 1,
    stage: "locked_diagnostic",
    lock_hash: "a".repeat(64),
    evaluation_manifest_hash: "d".repeat(64),
    default_scenario_id: "fixture-first",
    scenarios,
    documents: {},
  };
  await page.route("**/catalogue.json", (route) =>
    route.fulfill({ json: catalogue }),
  );
  await page.route("**/demo/**", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: files[new URL(route.request().url()).pathname] ?? "{}",
    }),
  );
}

test("paired replay synchronises clocks and shows actual decision scores", async ({
  page,
}) => {
  await installFixture(page);
  await page.goto("/");
  const panel = page.getByRole("region", { name: "Compare recorded policies" });
  await expect(
    panel.getByText("Selected minimum", { exact: true }),
  ).toBeVisible();
  await expect(
    panel.getByRole("img", { name: /Paired recorded sailing/ }),
  ).toBeVisible();
  await panel.getByRole("slider").fill("15");
  await expect(
    panel.getByRole("heading", { name: "Decision at 0:10" }),
  ).toBeVisible();
  await panel.getByRole("checkbox", { name: "Show diagnostic truth" }).check();
  await expect(panel).toContainText("DIAGNOSTIC TRUTH · recorded offline");
  await panel
    .getByLabel("Inspect policy")
    .selectOption("PPO_belief_fixed/seed-1");
  await expect(panel.getByRole("slider")).toHaveValue("0");
  await expect(panel).toContainText("Requested maximum");
  await expect(
    panel.getByRole("cell", { name: "65.00%", exact: true }),
  ).toBeVisible();
  await expect(panel).toContainText("not predicted success probabilities");
  await panel.getByLabel("Recorded world").selectOption("fixture-second");
  await expect(
    panel.getByRole("link", { name: "Download recorded replay" }),
  ).toHaveAttribute("href", /fixture-second-policy-2/);
  await panel.getByRole("button", { name: "Play", exact: true }).click();
  await expect(
    panel.getByRole("button", { name: "Pause", exact: true }),
  ).toBeVisible();
  await panel.getByRole("button", { name: "Reset" }).click();
  await expect(panel.getByRole("slider")).toHaveValue("0");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});

test("changing worlds aborts stale replay responses", async ({ page }) => {
  await installFixture(page);
  let release!: () => void;
  const delayed = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route(
    "**/demo/fixture-first-policy-0/replay.json",
    async (route) => {
      await delayed;
      await route.fallback();
    },
  );
  await page.goto("/");
  await page.getByLabel("Recorded world").selectOption("fixture-second");
  await expect(
    page.getByRole("link", { name: "Download recorded replay" }),
  ).toHaveAttribute("href", /fixture-second-policy-0/);
  release();
  await expect(
    page.getByRole("heading", { name: "Decision at 0:00" }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Download recorded replay" }),
  ).toHaveAttribute("href", /fixture-second-policy-0/);
});

for (const fault of ["hash", "future"] as const)
  test(`rejects ${fault} corruption in a selected recording`, async ({
    page,
  }) => {
    await installFixture(page, fault);
    await page.goto("/");
    await expect(page.getByRole("alert")).toContainText(
      fault === "hash" ? "hash mismatch" : "observation times",
    );
    await expect(
      page.getByRole("table", { name: /Conditional planning predictions/ }),
    ).toHaveCount(0);
  });
