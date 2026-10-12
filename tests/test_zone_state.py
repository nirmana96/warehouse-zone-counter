"""Zone state machine on fake tracks. fps=10, enter_frames=5, exit_frames=5, lost_frames=10.

Red zone covers foot x 0..100, green zone x 200..300 (see conftest.py).
"""

from wzc.detect_track import Track
from wzc.events import TRACK_LOST, TRACK_STARTED, ZONE_ENTERED, ZONE_EXITED
from wzc.zone_state import ZoneStateMachine

RED_X, GREEN_X, OUT_X = 50, 250, 150


def person(track_id: int, foot_x: float) -> Track:
    # Foot point = bottom-centre = (foot_x, 50), inside the zones' y range.
    return Track(track_id=track_id, xyxy=(foot_x - 5, 0, foot_x + 5, 50), conf=0.9)


def run(zsm: ZoneStateMachine, path: list[float | None], track_id: int = 1, start: int = 0):
    """Feed one track along a path of foot x positions (None = not detected that frame)."""
    events = []
    for i, x in enumerate(path):
        tracks = [] if x is None else [person(track_id, x)]
        events += zsm.step(start + i, tracks)
    return events


def of_type(events, event_type):
    return [e for e in events if e.event_type == event_type]


def test_enters_after_exactly_enter_frames(cfg, clock, ids):
    zsm = ZoneStateMachine(cfg, clock, ids)
    assert of_type(run(zsm, [RED_X] * 4), ZONE_ENTERED) == []
    entered = of_type(zsm.step(4, [person(1, RED_X)]), ZONE_ENTERED)
    assert len(entered) == 1
    e = entered[0]
    assert e.zone_id == "red_a"
    assert e.occurred_at == "2026-01-01T08:00:00.000Z"  # first frame of the streak, not frame 4
    assert e.payload["confirmed_at"] == "2026-01-01T08:00:00.400Z"
    assert e.payload["visit_no"] == 1


def test_four_frames_inside_is_not_an_entry(cfg, clock, ids):
    zsm = ZoneStateMachine(cfg, clock, ids)
    events = run(zsm, [RED_X] * 4 + [OUT_X] * 10)
    assert of_type(events, ZONE_ENTERED) == []


def test_exit_time_and_dwell(cfg, clock, ids):
    zsm = ZoneStateMachine(cfg, clock, ids)
    events = run(zsm, [RED_X] * 10 + [OUT_X] * 5)
    exited = of_type(events, ZONE_EXITED)
    assert len(exited) == 1
    assert exited[0].occurred_at == "2026-01-01T08:00:01.000Z"  # frame 10, first outside frame
    assert exited[0].payload["dwell_s"] == 1.0
    assert exited[0].payload["reason"] == "walked_out"
    assert exited[0].payload["visit_no"] == 1


def test_flicker_on_the_line_gives_one_visit(cfg, clock, ids):
    zsm = ZoneStateMachine(cfg, clock, ids)
    flicker = [OUT_X, OUT_X, RED_X] * 4  # never 5 outside in a row
    events = run(zsm, [RED_X] * 5 + flicker + [OUT_X] * 5)
    assert len(of_type(events, ZONE_ENTERED)) == 1
    assert len(of_type(events, ZONE_EXITED)) == 1


def test_second_red_entry_is_visit_two(cfg, clock, ids):
    zsm = ZoneStateMachine(cfg, clock, ids)
    events = run(zsm, [RED_X] * 5 + [OUT_X] * 5 + [RED_X] * 5)
    visits = [e.payload["visit_no"] for e in of_type(events, ZONE_ENTERED)]
    assert visits == [1, 2]


def test_visit_numbers_are_per_track(cfg, clock, ids):
    zsm = ZoneStateMachine(cfg, clock, ids)
    events = []
    for f in range(5):
        events += zsm.step(f, [person(1, RED_X), person(2, RED_X)])
    entered = of_type(events, ZONE_ENTERED)
    assert sorted((e.track_id, e.payload["visit_no"]) for e in entered) == [(1, 1), (2, 1)]


def test_green_entries_never_carry_visit_no(cfg, clock, ids):
    zsm = ZoneStateMachine(cfg, clock, ids)
    events = run(zsm, [GREEN_X] * 5 + [OUT_X] * 5 + [GREEN_X] * 5 + [OUT_X] * 5)
    entered = of_type(events, ZONE_ENTERED)
    exited = of_type(events, ZONE_EXITED)
    assert len(entered) == 2 and len(exited) == 2
    assert all(e.zone_id == "green_a" for e in entered)
    assert all("visit_no" not in e.payload for e in entered + exited)


def test_track_started_once(cfg, clock, ids):
    zsm = ZoneStateMachine(cfg, clock, ids)
    events = run(zsm, [OUT_X] * 20)
    started = of_type(events, TRACK_STARTED)
    assert len(started) == 1 and started[0].track_id == 1


def test_short_gap_keeps_the_visit_open(cfg, clock, ids):
    zsm = ZoneStateMachine(cfg, clock, ids)
    events = run(zsm, [RED_X] * 5 + [None] * 8 + [RED_X] * 3)  # hidden 8 frames < lost_frames
    assert len(of_type(events, ZONE_ENTERED)) == 1
    assert of_type(events, ZONE_EXITED) == []
    assert of_type(events, TRACK_LOST) == []


def test_lost_track_closes_the_open_visit(cfg, clock, ids):
    zsm = ZoneStateMachine(cfg, clock, ids)
    events = run(zsm, [RED_X] * 7 + [None] * 11)  # last seen frame 6, lost when gap > 10
    exited = of_type(events, ZONE_EXITED)
    lost = of_type(events, TRACK_LOST)
    assert len(exited) == 1 and len(lost) == 1
    assert exited[0].payload["reason"] == "track_lost"
    assert exited[0].occurred_at == "2026-01-01T08:00:00.700Z"  # frame 7, right after last seen
    assert exited[0].payload["dwell_s"] == 0.7
    assert events.index(exited[0]) < events.index(lost[0])


def test_same_id_after_loss_counts_as_new_visit(cfg, clock, ids):
    zsm = ZoneStateMachine(cfg, clock, ids)
    events = run(zsm, [RED_X] * 5 + [None] * 11 + [RED_X] * 5)
    visits = [e.payload["visit_no"] for e in of_type(events, ZONE_ENTERED)]
    assert visits == [1, 2]
    assert len(of_type(events, TRACK_STARTED)) == 2


def test_flush_closes_everything(cfg, clock, ids):
    zsm = ZoneStateMachine(cfg, clock, ids)
    run(zsm, [RED_X] * 6)
    events = zsm.flush(6)
    exited = of_type(events, ZONE_EXITED)
    assert len(exited) == 1 and exited[0].payload["reason"] == "stream_ended"
    assert len(of_type(events, TRACK_LOST)) == 1
    assert zsm.flush(7) == []


def test_every_event_id_is_unique(cfg, clock, ids):
    zsm = ZoneStateMachine(cfg, clock, ids)
    events = run(zsm, [RED_X] * 5 + [OUT_X] * 5 + [GREEN_X] * 5) + zsm.flush(15)
    assert len({e.event_id for e in events}) == len(events)
