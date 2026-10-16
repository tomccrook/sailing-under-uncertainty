import Ajv from "ajv";
import trackSchema from "../track.schema.json";
import weatherSchema from "../weather.schema.json";
import timingSchema from "../timing.schema.json";
import type { Track } from "./generated/track";
import type { WeatherPacket } from "./generated/weather";
import type { TimingReport } from "./generated/timing";

const ajv = new Ajv({ strict: false, allErrors: true });
export const utcMillis = (value: string) => {
  if (
    !/T.+(?:Z|[+-]\d{2}:\d{2})$/.test(value) ||
    !Number.isFinite(Date.parse(value))
  )
    throw new Error("Timestamp needs a valid explicit UTC offset.");
  return Date.parse(value);
};
ajv.addFormat("date-time", {
  validate: (value: string) => {
    try {
      utcMillis(value);
      return true;
    } catch {
      return false;
    }
  },
});
ajv.addFormat("date", /^\d{4}-\d{2}-\d{2}$/);
const trackValidator = ajv.compile<Track>(trackSchema);
const weatherValidator = ajv.compile<WeatherPacket>(weatherSchema);
const timingValidator = ajv.compile<TimingReport>(timingSchema);

export function readTrack(data: unknown): Track {
  if (
    !trackValidator(data) ||
    data.schema_version !== "track-v1" ||
    data.points.length > 10000
  )
    throw new Error(
      "Unsupported processed track. Use the local support importer.",
    );
  let lastRow = -1,
    lastTime = -Infinity,
    lastSegment = -1;
  let broken = true;
  for (const point of data.points) {
    if (point.row <= lastRow)
      throw new Error("Track rows must retain their source order.");
    lastRow = point.row;
    if (!point.accepted) {
      broken = true;
      continue;
    }
    if (
      point.east_m == null ||
      point.north_m == null ||
      point.segment == null ||
      point.timestamp_utc == null ||
      point.latitude_deg == null ||
      point.longitude_deg == null
    )
      throw new Error("Accepted track point has missing coordinates or time.");
    const time = utcMillis(point.timestamp_utc);
    if (
      time <= lastTime ||
      point.segment < lastSegment ||
      (broken && point.segment <= lastSegment)
    )
      throw new Error("Track connects an invalid point or has a bad clock.");
    if (
      (broken || point.segment !== lastSegment) &&
      point.derived_speed_over_ground_m_s != null
    )
      throw new Error("Track speed bridges a disconnected segment.");
    lastTime = time;
    lastSegment = point.segment;
    broken = false;
  }
  return data;
}

export function readWeather(data: unknown): WeatherPacket {
  if (
    !weatherValidator(data) ||
    data.schema_version !== "weather-v1" ||
    data.points.length > 10000
  )
    throw new Error("Unsupported processed weather packet.");
  const request = data.request;
  if (request.kind === "single_run") {
    if (!request.run_initialisation_utc || !data.available_utc)
      throw new Error("Single run lacks initialisation or publication time.");
    const initial = utcMillis(request.run_initialisation_utc);
    const available = utcMillis(data.available_utc);
    const known = request.known_available_utc;
    const expected = known
      ? utcMillis(known)
      : initial + (request.assumed_publication_lag_s ?? 28800) * 1000;
    if (
      available < initial ||
      available !== expected ||
      data.availability !== (known ? "known" : "assumed_lag") ||
      (known && !request.availability_evidence)
    )
      throw new Error("Invalid forecast publication evidence.");
  } else if (
    data.available_utc !== null ||
    data.availability !== "context_only"
  )
    throw new Error(
      "Context products cannot be relabelled as available historical forecasts.",
    );
  let previous = -Infinity;
  for (const point of data.points) {
    const valid = utcMillis(point.valid_utc);
    if (valid <= previous)
      throw new Error("Weather validity times must increase.");
    previous = valid;
  }
  return data;
}

export function readTiming(data: unknown): TimingReport {
  if (!timingValidator(data) || data.schema_version !== "timing-v1")
    throw new Error("Unsupported timing report.");
  if (
    !data.training_days.length ||
    data.holdout_days.length !== 2 ||
    [...data.training_days].sort().at(-1)! >= [...data.holdout_days].sort()[0]!
  )
    throw new Error("Timing report lacks its chronological group holdout.");
  return data;
}

export async function boundedJson(response: Response): Promise<unknown> {
  if (!response.ok)
    throw new Error(
      "Support file unavailable; generate the labelled fixture locally.",
    );
  const text = await response.text();
  if (new Blob([text]).size > 5_000_000)
    throw new Error("Processed file exceeds 5 MB.");
  return JSON.parse(text) as unknown;
}
