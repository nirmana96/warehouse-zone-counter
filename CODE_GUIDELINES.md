# Code guideline: warehouse-zone-counter (week 1)

This file tells you (or a coding assistant) how to write the week 1 code. Read it before the first commit and keep it in the repo root.

## What week 1 builds

A pipeline that reads warehouse CCTV video, tracks people, and tests each person's foot point against one red and one green polygon per camera. It writes everything it sees as events to an append-only SQL table, saves a face-blurred snapshot with the track ID on every red entry, and serves a live dashboard.

Red zone: count every visit with entry time, exit time and dwell, and count repeat entries per track ID.
Green zone: count every visit with entry time, exit time and dwell. No per-person repeat counting.

## Rules that are never broken

1. **No unblurred face leaves the process.** Blur before a frame is streamed, saved or logged. There is no debug flag that turns this off.
2. **Events are append-only.** Code never runs `UPDATE` or `DELETE` on `events`. Database triggers enforce it, and a test checks that the triggers fire.
3. **Counts come from projections, never from in-memory counters.** The dashboard reads `zone_visits` and views, which are built from `events`.
4. **Zone membership uses the foot point**, the bottom-centre of the box, in the same pixel space the zones were drawn in. Week 1 uses one camera on raw frames; undistortion only matters for cross-camera floor mapping (a stretch goal).
5. **No identity.** Track IDs are anonymous integers. Appearance (ReID) features stay in memory and are never written to disk.
6. **Numbers in the README are measured**, with the CPU model next to every timing.

## Stack

| Purpose | Tool |
| --- | --- |
| Language | Python 3.11+ |
| Packages | uv, versions pinned in `uv.lock` |
| Detection and tracking | Ultralytics YOLO26n, BoT-SORT with ReID |
| Geometry | OpenCV (`pointPolygonTest`) |
| Face blur | OpenCV YuNet (`cv2.FaceDetectorYN`) |
| Config | YAML loaded into Pydantic models |
| Storage | SQLite in WAL mode, plain SQL migrations |
| API and dashboard | FastAPI, plain HTML, Chart.js |
| Quality | ruff (lint and format), pytest, GitHub Actions |
| Packaging | Docker Compose |
| Experiment log | MLflow (FPS, counts, settings per run) |

Do not add a dependency without a reason written in the pull request or commit message.

## Repo layout

```text
warehouse-zone-counter/
  configs/
    trackers/           # botsort_reid.yaml
    zones/              # cam_04.yaml
  migrations/
    001_events.sql      # events table, triggers, projections, views
  src/wzc/
    config.py           # Pydantic models for YAML
    video.py            # frame reader
    detect_track.py     # YOLO26n + BoT-SORT wrapper
    geometry.py         # foot point, point-in-polygon (pure functions)
    zone_state.py       # enter/exit state machine per track and zone
    events.py           # event dataclasses and types
    event_store.py      # append-only writer and reader
    projector.py        # events -> zone_visits
    replay.py           # rebuild projections from seq 0
    privacy.py          # face blur
    snapshots.py        # red-entry snapshot
    pipeline.py         # the main loop
    api.py              # FastAPI app
  web/                  # dashboard page
  tools/draw_zones.py   # click tool for polygons
  eval/
    labels/             # hand-counted entries (CSV)
    score.py            # count error, precision, recall, ID switches
  tests/
  scripts/download.sh
  docker-compose.yml
  pyproject.toml
  README.md
```

Video and model weights are never committed. `scripts/download.sh` fetches the footage; weights download on first run.

## Config

One YAML file per camera, validated by Pydantic at startup. A bad config stops the program with a clear message. It never runs with defaults the user did not choose.

```yaml
camera: cam_04
source: data/amc/cam_04.mp4
recording_start: "2026-01-01T08:00:00Z"   # video time 0; footage has no wall clock
frame_size: [1920, 1080]                  # written by draw_zones.py; checked against the video
classes: [person]
anchor: bottom_center
enter_frames: 5
exit_frames: 5
lost_frames: 60                           # missing longer than this = track lost, open visit closed
detector:
  model: yolo26n.pt
  tracker: configs/trackers/botsort_reid.yaml
  conf: 0.3
  imgsz: 960
  device: cpu
zones:
  - id: aisle_main
    colour: red
    polygon: [[860, 400], [1060, 400], [1120, 1000], [800, 1000]]
    track_repeats: true
    snapshot_on_enter: true
  - id: walkway_front
    colour: green
    polygon: [[0, 820], [700, 820], [700, 1080], [0, 1080]]
    track_repeats: false
    snapshot_on_enter: false
```

Polygon coordinates are in video pixel space. Draw them with `tools/draw_zones.py`, never by hand. Pydantic rejects unknown keys, polygons under 3 points, points outside `frame_size`, a missing timezone, and green zones with `track_repeats` or `snapshot_on_enter`.

## Events

Entry time is the first frame of the inside streak and exit time the first frame of the outside streak, so dwell is not padded by the confirmation delay. Events are written when confirmed, so the log is not always in `occurred_at` order; sort by `occurred_at` when reading time.

All events share one shape. `occurred_at` is video time: `recording_start + frame_idx / fps`, in ISO 8601 UTC. Never use the wall clock for it.

```python
@dataclass(frozen=True)
class Event:
    event_id: str          # uuid4, set once, makes retries safe
    event_type: str
    occurred_at: str
    camera_id: str
    track_id: int | None
    zone_id: str | None
    payload: dict          # JSON-serialisable only
    schema_version: int = 1
```

| `event_type` | Written when | Payload |
| --- | --- | --- |
| `track_started` | Tracker gives a new ID | `box` |
| `zone_entered` | Foot point inside for `enter_frames` in a row | `colour`, `foot_point`, `visit_no` (red only) |
| `zone_exited` | Foot point outside for `exit_frames` in a row, track lost, or video ended | `dwell_s`, `entered_at`, `reason`, `visit_no` (red only) |
| `snapshot_saved` | Right after a red `zone_entered` | `path`, `faces_blurred` |
| `track_lost` | ID missing for more than `lost_frames` | `last_seen`, `reason` |

Adding a field to a payload is fine. Renaming or removing one means a new `schema_version` and a projector that handles both.

## Module contracts

Keep these signatures stable. Everything else can change.

```python
# detect_track.py
@dataclass(frozen=True)
class Track:
    track_id: int
    xyxy: tuple[float, float, float, float]
    conf: float

class Tracker:
    def update(self, frame: np.ndarray) -> list[Track]: ...

# geometry.py (pure, no I/O)
def foot_point(xyxy) -> tuple[float, float]: ...
def inside(point, polygon) -> bool: ...

# zone_state.py
class ZoneStateMachine:
    def step(self, frame_idx: int, tracks: list[Track]) -> list[Event]: ...
    # emits track_started, zone_entered, zone_exited, track_lost
    # holds per (track_id, zone_id): inside_streak, outside_streak, state, entered_at
    # holds per (track_id, red zone_id): visit_no counter

# event_store.py
class EventStore:
    def append(self, events: list[Event]) -> None: ...   # one transaction, ignores duplicate event_id
    def read_after(self, seq: int, limit: int = 1000) -> list[tuple[int, Event]]: ...

# projector.py
class Projector:
    def run_once(self) -> int: ...   # applies new events, returns how many

# privacy.py
def blur_faces(frame: np.ndarray) -> tuple[np.ndarray, int]: ...   # returns blurred frame and face count

# snapshots.py
def save_red_snapshot(frame: np.ndarray, track: Track, zone, out_dir: Path) -> Path: ...
```

`zone_state.py` does not know about video, SQL or the dashboard. It takes tracks and returns events, so it can be tested with plain lists.

## The main loop

```text
for each frame:
    tracks = tracker.update(frame)
    events = zone_state.step(frame_idx, tracks)
    frame, n_faces = blur_faces(frame)            # always, before anything below
    for e in events where e is red zone_entered:
        path = save_red_snapshot(frame, track, zone)
        events.append(snapshot_saved(e, path, n_faces))
    store.append(events)                           # one transaction per frame
    stream.publish(draw_overlay(frame, zones, tracks))
```

The projector runs as its own loop (or every N frames in the same process) and reads `events` after its last `seq`. The API only reads projections and views.

## Database

- Schema lives in `migrations/001_events.sql`: the `events` table, the two triggers that block `UPDATE` and `DELETE`, `zone_visits`, `projection_offsets` and the `red_repeats` view.
- Open SQLite with `PRAGMA journal_mode=WAL` and `PRAGMA foreign_keys=ON`.
- Use parameterised queries only. No string-built SQL.
- `python -m wzc.replay` deletes the projection tables' rows, resets `projection_offsets`, and rebuilds from `seq` 0. It never touches `events`.

## API

| Endpoint | Returns |
| --- | --- |
| `GET /stream/{camera}` | MJPEG stream, blurred, zones and track IDs drawn |
| `GET /api/counters` | Red entries, green visits, unique red track IDs, IDs with 2+ red entries |
| `GET /api/visits?zone=&start=&end=` | Rows from `zone_visits` |
| `GET /api/red-repeats?min_entries=` | Rows from `red_repeats` |
| `GET /api/events?after_seq=` | Raw events for the feed |
| `GET /snapshots/{file}` | A saved snapshot |

The dashboard page polls `/api/counters` and `/api/events` once a second. No endpoint writes to `events`.

## Code style

- Type hints on every public function. Frozen dataclasses for data passed between modules.
- `ruff check` and `ruff format` pass before every commit.
- Use the `logging` module with one logger per module. No `print` outside `tools/`.
- No global state. Build objects in `pipeline.py` and pass them in, so tests can swap them.
- Geometry and state logic stay pure. I/O lives at the edges (video, store, API).
- Fail loudly on bad input (config errors, polygon with fewer than 3 points, video that won't open or does not match `frame_size`).
- Comments explain why, not what.

## Tests

Each one must pass in CI with no video or GPU.

1. `geometry`: foot point maths, point on the edge, point outside, a concave polygon.
2. `zone_state`: fake tracks walking in and out. Entry after exactly 5 frames, no entry after 4, a flicker on the line gives one visit, a second red entry gives `visit_no = 2`, and green entries never carry `visit_no`.
3. `event_store`: the same `event_id` twice is stored once. `UPDATE` and `DELETE` raise.
4. `replay`: run the projector live on a fixed list of events, run a replay, and check the tables are identical.
5. `privacy`: a frame with a known face comes back with that region changed, and `faces_blurred >= 1`.
6. `api`: counters on a small seeded database return the expected numbers.

A short clip (10 to 20 seconds, cut from the footage, not committed) is used for a local smoke test only.

## Evaluation

- Labels: `eval/labels/<camera>.csv` with `zone_id, track_hint, entered_at, exited_at`, hand-counted at half speed.
- `eval/score.py` matches system visits to labels within 2 seconds and reports count error, precision and recall per zone, plus ID switches.
- Log every run to MLflow: tracker settings, `enter_frames`, `exit_frames`, model format, FPS, scores.
- FPS is measured over the full clip, after warm-up, for PyTorch, ONNX and OpenVINO, with the CPU model recorded.

## Git and CI

- Commit at the end of every session with a message saying what changed and why.
- GitHub Actions runs `ruff check` and `pytest` on every push. The README badge must be green before the repo goes public.
- `main` always runs. Unfinished work lives on a branch.

## Done when

- [ ] `docker compose up` shows the dashboard on `cam_04` with live counters and the red entry feed
- [ ] Every red entry has a blurred snapshot with its track ID drawn
- [ ] Repeat red entries per track ID show in the dashboard and in `red_repeats`
- [ ] A replay rebuilds identical tables
- [ ] README shows counted vs labelled visits per zone, ID switches, and FPS with the CPU model named
- [ ] README has a privacy section: faces blurred everywhere, no identification, ReID features never stored
