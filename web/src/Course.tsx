import { useEffect, useRef } from "react";
import type { ReplayBundle } from "./generated/replay";
import { indexAt } from "./evidence";

export function Course({
  replay,
  index,
  truth,
  comparison,
  elapsed,
}: {
  replay: ReplayBundle;
  index: number;
  truth: boolean;
  comparison?: ReplayBundle;
  elapsed?: number;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    const draw = () => {
      const context = element.getContext("2d");
      if (!context) return;
      const width = element.clientWidth;
      const height = element.clientHeight;
      const ratio = window.devicePixelRatio || 1;
      element.width = width * ratio;
      element.height = height * ratio;
      context.scale(ratio, ratio);
      context.clearRect(0, 0, width, height);
      const [left, right, bottom, top] = replay.course.bounds_xy_m ?? [
        -600, 600, -100, 1200,
      ];
      const metricScale = Math.min(
        (width - 90) / (right - left),
        (height - 70) / (top - bottom),
      );
      const ox = (width - (right - left) * metricScale) / 2;
      const oy = (height - (top - bottom) * metricScale) / 2;
      const point = ([x, y]: [number, number]): [number, number] => [
        ox + (x - left) * metricScale,
        height - oy - (y - bottom) * metricScale,
      ];
      context.font = "11px ui-monospace, monospace";
      context.lineWidth = 1;
      for (let x = Math.ceil(left / 200) * 200; x <= right; x += 200) {
        const [px, py] = point([x, bottom]);
        context.strokeStyle = "#d5e1e0";
        context.beginPath();
        context.moveTo(px, py);
        context.lineTo(px, point([x, top])[1]);
        context.stroke();
        context.fillStyle = "#617b7e";
        context.fillText(String(x), px - 12, py + 20);
      }
      for (let y = Math.ceil(bottom / 200) * 200; y <= top; y += 200) {
        const [px, py] = point([left, y]);
        context.strokeStyle = "#d5e1e0";
        context.beginPath();
        context.moveTo(px, py);
        context.lineTo(point([right, y])[0], py);
        context.stroke();
        context.fillStyle = "#617b7e";
        context.fillText(String(y), px - 36, py + 4);
      }
      context.setLineDash([4, 5]);
      context.strokeStyle = "#80989a";
      context.strokeRect(
        ox,
        oy,
        (right - left) * metricScale,
        (top - bottom) * metricScale,
      );
      context.setLineDash([]);
      const mark = point(replay.course.target_xy_m ?? [0, 1000]);
      context.beginPath();
      context.arc(
        ...mark,
        Math.max((replay.course.finish_radius_m ?? 10) * metricScale, 4),
        0,
        2 * Math.PI,
      );
      context.strokeStyle = "#bb633e";
      context.lineWidth = 2;
      context.stroke();
      context.fillStyle = "#8a4b31";
      context.textAlign = "center";
      context.fillText(
        `FINISH · ${replay.course.finish_radius_m ?? 10} m region`,
        mark[0],
        mark[1] - 15,
      );
      context.textAlign = "start";
      const start = point(replay.course.start_xy_m ?? [0, 0]);
      context.fillStyle = "#153d45";
      context.beginPath();
      context.arc(...start, 4, 0, 2 * Math.PI);
      context.fill();
      context.fillText("START", start[0] + 12, start[1] + 8);
      context.beginPath();
      replay.samples.forEach((sample, i) => {
        const p = point(sample.position_xy_m);
        if (i === 0) context.moveTo(...p);
        else context.lineTo(...p);
      });
      context.strokeStyle = "#aacac7";
      context.lineWidth = 2;
      context.setLineDash([4, 4]);
      context.stroke();
      context.setLineDash([]);
      context.beginPath();
      replay.samples.slice(0, index + 1).forEach((sample, i) => {
        const p = point(sample.position_xy_m);
        if (i === 0) context.moveTo(...p);
        else context.lineTo(...p);
      });
      context.strokeStyle = "#14776e";
      context.lineWidth = 3;
      context.stroke();
      if (comparison) {
        const otherIndex = indexAt(
          comparison.samples,
          elapsed ?? replay.samples[index]!.elapsed_s,
        );
        context.beginPath();
        comparison.samples.slice(0, otherIndex + 1).forEach((s, i) => {
          const p = point(s.position_xy_m);
          if (i === 0) context.moveTo(...p);
          else context.lineTo(...p);
        });
        context.strokeStyle = "#a65532";
        context.lineWidth = 3;
        context.setLineDash([8, 4]);
        context.stroke();
        context.setLineDash([]);
        const p = point(comparison.samples[otherIndex]!.position_xy_m);
        context.fillStyle = "#a65532";
        context.beginPath();
        context.moveTo(p[0], p[1] - 7);
        context.lineTo(p[0] + 7, p[1]);
        context.lineTo(p[0], p[1] + 7);
        context.lineTo(p[0] - 7, p[1]);
        context.closePath();
        context.fill();
      }
      const sample = replay.samples[index];
      if (sample) {
        const p = point(sample.position_xy_m);
        context.save();
        context.translate(...p);
        context.rotate(sample.heading_rad);
        context.beginPath();
        context.moveTo(0, -12);
        context.lineTo(7, 9);
        context.lineTo(0, 5);
        context.lineTo(-7, 9);
        context.closePath();
        context.fillStyle = "#14776e";
        context.fill();
        context.strokeStyle = "#fff";
        context.lineWidth = 1.5;
        context.stroke();
        context.restore();
        const wind = truth
          ? sample.diagnostic_wind_air_m_s
          : sample.observation.wind_air_m_s;
        const origin: [number, number] = [width - 48, 17];
        const norm = Math.hypot(...wind);
        if (norm > 0) {
          const dx = (wind[0] / norm) * 27,
            dy = (-wind[1] / norm) * 27;
          context.beginPath();
          context.moveTo(...origin);
          context.lineTo(origin[0] + dx, origin[1] + dy);
          context.strokeStyle = "#bb633e";
          context.lineWidth = 2;
          context.stroke();
          const angle = Math.atan2(dy, dx);
          context.beginPath();
          context.moveTo(origin[0] + dx, origin[1] + dy);
          context.lineTo(
            origin[0] + dx - 8 * Math.cos(angle - 0.5),
            origin[1] + dy - 8 * Math.sin(angle - 0.5),
          );
          context.moveTo(origin[0] + dx, origin[1] + dy);
          context.lineTo(
            origin[0] + dx - 8 * Math.cos(angle + 0.5),
            origin[1] + dy - 8 * Math.sin(angle + 0.5),
          );
          context.stroke();
        }
        context.fillStyle = "#617b7e";
        context.fillText("AIR FLOW", width - 78, 64);
      }
      if (sample) {
        const current = sample.observation.known_current_m_s;
        const norm = Math.hypot(...current);
        const origin: [number, number] = [width - 48, 110];
        context.strokeStyle = "#617b7e";
        context.lineWidth = 2;
        context.beginPath();
        if (norm > 0) {
          const dx = (current[0] / norm) * 24,
            dy = (-current[1] / norm) * 24;
          context.moveTo(...origin);
          context.lineTo(origin[0] + dx, origin[1] + dy);
          const angle = Math.atan2(dy, dx);
          context.moveTo(
            origin[0] + dx - 7 * Math.cos(angle - 0.5),
            origin[1] + dy - 7 * Math.sin(angle - 0.5),
          );
          context.lineTo(origin[0] + dx, origin[1] + dy);
          context.lineTo(
            origin[0] + dx - 7 * Math.cos(angle + 0.5),
            origin[1] + dy - 7 * Math.sin(angle + 0.5),
          );
        } else context.arc(...origin, 3, 0, 2 * Math.PI);
        context.stroke();
        context.fillStyle = "#617b7e";
        context.fillText(`CURRENT ${norm.toFixed(2)} m/s`, width - 130, 149);
      }
      context.fillStyle = "#617b7e";
      context.fillText("EAST (m)", width / 2 - 27, height - 5);
      context.fillText("N ↑", 18, 22);
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(element);
    return () => observer.disconnect();
  }, [replay, index, truth, comparison, elapsed]);
  return (
    <canvas
      ref={canvas}
      className="course-canvas"
      role="img"
      aria-label={
        comparison
          ? "Paired recorded sailing trajectories on a shared elapsed clock"
          : "Recorded sailing trajectory in east and north metres"
      }
    />
  );
}
