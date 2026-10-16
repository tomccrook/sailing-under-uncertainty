import Ajv from "ajv";
import catalogueSchema from "../catalogue.schema.json";
import detailsSchema from "../details.schema.json";
import replaySchema from "../replay.schema.json";
import type {
  Asset,
  Catalogue,
  Recording,
  Scenario,
} from "./generated/catalogue";
import type { Details } from "./generated/details";
import type { ReplayBundle } from "./generated/replay";

const ajv = new Ajv({ strict: false });
const catalogueValidator = ajv.compile<Catalogue>(catalogueSchema);
const detailsValidator = ajv.compile<Details>(detailsSchema);
const replayValidator = ajv.compile<ReplayBundle>(replaySchema);

export async function checkedBytes(asset: Asset, signal: AbortSignal) {
  if (!asset.url.startsWith("/demo/") || asset.url.split("/").includes(".."))
    throw new Error("Evidence URL is outside the public catalogue.");
  const response = await fetch(asset.url, { signal });
  if (!response.ok) throw new Error("Recorded evidence file is unavailable.");
  if (Number(response.headers.get("content-length")) > 12_000_000)
    throw new Error("Evidence file exceeds the replay budget.");
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength > 12_000_000)
    throw new Error("Evidence file exceeds the replay budget.");
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const hash = Array.from(new Uint8Array(digest), (v) =>
    v.toString(16).padStart(2, "0"),
  ).join("");
  if (hash !== asset.sha256)
    throw new Error("Evidence hash mismatch. Re-export the catalogue.");
  return bytes;
}

export async function checkedJson(
  asset: Asset,
  signal: AbortSignal,
): Promise<unknown> {
  return JSON.parse(
    new TextDecoder().decode(await checkedBytes(asset, signal)),
  );
}

export async function loadCatalogue(
  signal: AbortSignal,
): Promise<Catalogue | null> {
  const response = await fetch("/catalogue.json", { signal });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error("Diagnostic catalogue is unavailable.");
  // Vite's missing-file HTML fallback is equivalent to an absent optional catalogue.
  if (response.headers.get("content-type")?.includes("text/html")) return null;
  const text = await response.text();
  if (text.length > 200_000)
    throw new Error("Diagnostic catalogue exceeds its size budget.");
  const value: unknown = JSON.parse(text);
  if (!catalogueValidator(value))
    throw new Error("Invalid diagnostic catalogue contract.");
  const ids = value.scenarios.map((s) => s.scenario_id);
  if (
    new Set(ids).size !== ids.length ||
    !ids.includes(value.default_scenario_id) ||
    value.scenarios.some(
      (s) =>
        new Set(s.recordings.map((r) => r.variant)).size !==
        s.recordings.length,
    )
  )
    throw new Error(
      "Diagnostic catalogue has ambiguous scenario or policy identifiers.",
    );
  return value;
}

export type LoadedRecording = {
  recording: Recording;
  replay: ReplayBundle;
  details: Details;
};

export async function loadRecording(
  recording: Recording,
  scenario: Scenario,
  signal: AbortSignal,
): Promise<LoadedRecording> {
  const [replay, details] = await Promise.all([
    checkedJson(recording.replay, signal),
    checkedJson(recording.details, signal),
  ]);
  if (!replayValidator(replay) || !detailsValidator(details))
    throw new Error("Unsupported recorded replay or decision contract.");
  if (
    replay.samples.length === 0 ||
    replay.samples.length > 5000 ||
    replay.run_id !== recording.run_id ||
    details.run_id !== recording.run_id ||
    details.variant !== recording.variant ||
    replay.policy_label !== recording.variant ||
    details.world_hash !== scenario.world_hash ||
    replay.summary.elapsed_s !== recording.elapsed_s ||
    replay.summary.status !== recording.status ||
    replay.stage !== "locked" ||
    replay.samples[0]!.elapsed_s !== 0 ||
    replay.samples.at(-1)!.elapsed_s !== recording.elapsed_s ||
    JSON.stringify(replay.course) !== JSON.stringify(details.config.course) ||
    replay.samples.some(
      (s, i) =>
        s.elapsed_s < (s.observation.measurement_time_s ?? 0) ||
        (i > 0 && s.elapsed_s <= replay.samples[i - 1]!.elapsed_s),
    ) ||
    details.decisions.length !== replay.summary.decisions ||
    details.decisions.some(
      (d, i) =>
        d.elapsed_s !== d.observation.elapsed_s ||
        d.elapsed_s >= recording.elapsed_s ||
        (i > 0 && d.elapsed_s <= details.decisions[i - 1]!.elapsed_s) ||
        (d.observation.forecast?.available_s ?? 0) > d.elapsed_s ||
        d.observation.measurement_time_s > d.elapsed_s ||
        (d.planning &&
          (d.planning.selected_index >= d.planning.candidates.length ||
            Math.abs(
              d.planning.candidates[d.planning.selected_index]!.heading_rad -
                d.action.target_heading_rad,
            ) > 1e-8)),
    )
  )
    throw new Error(
      "Recorded evidence disagrees with the paired catalogue or observation times.",
    );
  return { recording, replay, details };
}

export function indexAt<T extends { elapsed_s: number }>(
  samples: T[],
  elapsed: number,
): number {
  let low = 0,
    high = samples.length - 1,
    index = 0;
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (samples[mid]!.elapsed_s <= elapsed) {
      index = mid;
      low = mid + 1;
    } else high = mid - 1;
  }
  return index;
}

export const degrees = (angle: number) =>
  (((angle * 180) / Math.PI + 360) % 360).toFixed(1);
export const timeLabel = (seconds: number) =>
  `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
