"""Click tool: draw the red and green zone polygons on a frame of the camera video.

Usage:
    uv run python tools/draw_zones.py --config configs/zones/cam_04.yaml
    uv run python tools/draw_zones.py --config configs/zones/cam_04.yaml --frame 900

Controls (the window must have focus):
    left click      add a point to the current zone
    right click / u undo the last point
    r               clear the current zone
    n / Enter       finish the current zone (needs 3+ points) and move to the next
    s               save both polygons and frame_size into the config, then quit
    q / Esc         quit without saving

Draw zones in the order they appear in the config (red first, then green). Click the
floor outline where feet stand, not the rack tops: the counter tests foot points.

The frame is shown on screen only. Nothing is saved except coordinates, because the
raw frame shows unblurred faces.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

from wzc.config import CameraConfig

ROOT = Path(__file__).resolve().parents[1]

COLOURS_BGR = {"red": (40, 40, 230), "green": (60, 180, 60)}
WINDOW = "draw zones"


def load_raw(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_frame(source: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {source}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {index} of {source}")
    return frame


def rewrite_config(text: str, frame_size: tuple[int, int], polygons: dict[str, list]) -> str:
    """Replace frame_size and each zone's polygon line, keeping every comment."""
    out: list[str] = []
    current_zone: str | None = None
    replaced: set[str] = set()
    for line in text.splitlines():
        m_id = re.match(r"^\s*-\s*id:\s*([a-z0-9_]+)", line)
        if m_id:
            current_zone = m_id.group(1)
        m_fs = re.match(r"^(\s*)frame_size:", line)
        m_poly = re.match(r"^(\s*)polygon:", line)
        if m_fs:
            out.append(
                f"{m_fs.group(1)}frame_size: [{frame_size[0]}, {frame_size[1]}]"
                "  # set by draw_zones.py from the video"
            )
            replaced.add("frame_size")
        elif m_poly and current_zone in polygons:
            pts = ", ".join(f"[{x}, {y}]" for x, y in polygons[current_zone])
            out.append(f"{m_poly.group(1)}polygon: [{pts}]")
            replaced.add(current_zone)
        else:
            out.append(line)
    missing = ({"frame_size"} | set(polygons)) - replaced
    if missing:
        raise SystemExit(f"could not find these keys in the config to replace: {sorted(missing)}")
    return "\n".join(out) + "\n"


class Drawer:
    def __init__(self, frame: np.ndarray, zones: list[dict], max_width: int) -> None:
        self.frame = frame
        self.zones = zones
        h, w = frame.shape[:2]
        self.scale = min(1.0, max_width / w)
        self.points: dict[str, list[tuple[int, int]]] = {z["id"]: [] for z in zones}
        self.current = 0

    @property
    def zone(self) -> dict | None:
        return self.zones[self.current] if self.current < len(self.zones) else None

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        z = self.zone
        if z is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            h, w = self.frame.shape[:2]
            fx = int(round(min(max(x / self.scale, 0), w - 1)))
            fy = int(round(min(max(y / self.scale, 0), h - 1)))
            self.points[z["id"]].append((fx, fy))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points[z["id"]]:
            self.points[z["id"]].pop()

    def render(self) -> np.ndarray:
        img = self.frame.copy()
        overlay = img.copy()
        for i, z in enumerate(self.zones):
            pts = self.points[z["id"]]
            colour = COLOURS_BGR[z["colour"]]
            if not pts:
                continue
            arr = np.array(pts, dtype=np.int32)
            done = i < self.current
            if done and len(pts) >= 3:
                cv2.fillPoly(overlay, [arr], colour)
            cv2.polylines(img, [arr], isClosed=done, color=colour, thickness=3)
            for p in pts:
                cv2.circle(img, p, 6, colour, -1)
            cv2.putText(img, z["id"], pts[0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        if self.scale < 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        z = self.zone
        lines = (
            [
                f"drawing {z['colour']} zone '{z['id']}': {len(self.points[z['id']])} points",
                "click=add  right-click/u=undo  r=clear  n/Enter=finish zone  q=quit",
            ]
            if z
            else ["all zones done", "s=save to config   q=quit without saving"]
        )
        for k, text in enumerate(lines):
            y = 30 + 28 * k
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        return img


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0, help="frame index to draw on")
    parser.add_argument("--max-width", type=int, default=1400, help="display width cap in pixels")
    args = parser.parse_args()

    raw = load_raw(args.config)
    source = Path(raw["source"])
    if not source.is_absolute():
        source = ROOT / source
    frame = read_frame(source, args.frame)
    h, w = frame.shape[:2]
    print(f"{source.name}: {w}x{h}, drawing on frame {args.frame}")

    drawer = Drawer(frame, raw["zones"], args.max_width)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, drawer.on_mouse)

    while True:
        cv2.imshow(WINDOW, drawer.render())
        key = cv2.waitKey(30) & 0xFF
        z = drawer.zone
        if key in (ord("q"), 27):
            print("quit without saving")
            break
        if key == ord("u") and z and drawer.points[z["id"]]:
            drawer.points[z["id"]].pop()
        elif key == ord("r") and z:
            drawer.points[z["id"]].clear()
        elif key in (ord("n"), 13) and z:
            if len(drawer.points[z["id"]]) < 3:
                print(f"zone {z['id']} needs at least 3 points")
            else:
                drawer.current += 1
        elif key == ord("s"):
            if drawer.zone is not None:
                print("finish every zone first (n / Enter)")
                continue
            text = args.config.read_text(encoding="utf-8")
            new_text = rewrite_config(text, (w, h), drawer.points)
            # Validate before writing, so a bad polygon never lands in the file.
            CameraConfig.model_validate(yaml.safe_load(new_text))
            args.config.write_text(new_text, encoding="utf-8")
            for zid, pts in drawer.points.items():
                print(f"{zid}: {pts}")
            print(f"saved to {args.config}")
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            print("window closed without saving")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
