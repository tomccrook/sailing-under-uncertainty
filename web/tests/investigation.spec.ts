import { test, expect } from "@playwright/test";

const strategyNames = [
  "fixed_script",
  "local_assistant",
  "researcher_critic",
] as const;
const fixture = {
  schema_version: 1,
  evidence_level: "synthetic_audit_fixture",
  model_repository: "local/test-model",
  model_revision: "a".repeat(40),
  source_sha256: "b".repeat(64),
  protocol_version: "2026-07-28",
  strategies: strategyNames.map((name) => ({
    name,
    tasks: 2,
    accepted_reports: name === "fixed_script" ? 2 : 1,
    exact_issue_sets: name === "fixed_script" ? 2 : 1,
    exact_rate_wilson95: [0.3, 0.9],
    false_accusations: name === "fixed_script" ? 0 : 1,
    missed_issues: 0,
    unsupported_structured_claims: name === "fixed_script" ? 0 : 1,
    numerical_correct_reports: name === "fixed_script" ? 2 : 1,
    budget_compliant_tasks: 2,
    wall_s: name === "fixed_script" ? 0.1 : 50,
    model_calls: name === "fixed_script" ? 0 : 4,
    generated_tokens: name === "fixed_script" ? 0 : 300,
    prompt_tokens: 900,
  })),
  tasks: ["evaluation-00", "evaluation-01"].flatMap((task_id) =>
    strategyNames.map((strategy) => {
      const good = strategy === "fixed_script" || task_id === "evaluation-00";
      return {
        task_id,
        strategy,
        status: good ? "succeeded" : "failed",
        expected_issues: [],
        predicted_issues: good ? [] : ["omitted_failures"],
        exact_issue_set: good,
        accepted_report: good,
        evidence_links_correct: good,
        numerical_correct: good,
        error: good ? null : "Unsupported issue accusation",
        explanation: good
          ? "Exploratory comparison only."
          : "Unverified model text.",
      };
    }),
  ),
  paid_service_cost_usd: 0,
  result_sha256: "c".repeat(64),
  evidence_url: "/investigation-evidence.json",
  limits: "Small synthetic fixtures only.",
};

test("retains rejected assistant reports and separates audit from sailing efficacy", async ({
  page,
}) => {
  await page.route("**/investigations.json", (route) =>
    route.fulfill({ json: fixture }),
  );
  await page.goto("/");
  const panel = page.getByRole("region", {
    name: "Investigation assistant audit",
  });
  await expect(
    panel.getByText("Synthetic audit tasks", { exact: true }),
  ).toBeVisible();
  await expect(panel.getByText(/All 2 tasks remain/)).toBeVisible();
  await panel.getByLabel("Investigation task").selectOption("evaluation-01");
  await expect(
    panel.getByText("Report rejected / failed", { exact: true }),
  ).toHaveCount(2);
  await expect(
    panel.getByText("Unsupported issue accusation", { exact: true }),
  ).toHaveCount(2);
  await expect(
    panel.getByRole("link", { name: /Download full audit/ }),
  ).toHaveAttribute("href", "/investigation-evidence.json");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
});

test("rejects missing case denominators instead of showing favourable subset", async ({
  page,
}) => {
  await page.route("**/investigations.json", (route) =>
    route.fulfill({ json: { ...fixture, tasks: fixture.tasks.slice(1) } }),
  );
  await page.goto("/");
  const panel = page.getByRole("region", {
    name: "Investigation assistant audit",
  });
  await expect(panel.getByRole("status")).toContainText(
    "missing or inconsistent task denominators",
  );
  await expect(panel.getByRole("table")).toHaveCount(0);
});
