"""
Procedural conflict simulator (Phase 6).

Generates synthetic forklift-pedestrian scenarios on a warehouse aisle, since
real near-misses cannot be safely staged or collected. Agents follow waypoints
at constant speed; each timestep is labelled SAFE / CAUTION / IMMINENT by the
TTC physics from src.risk.ttc. Labelling gates on BOTH time-to-collision AND
distance to avoid over-firing IMMINENT. The output snapshots feed the BEV
rasteriser (Phase 7) to build the BRIN training set.
"""
import numpy as np


class Agent:
    """An agent moving along waypoints at constant speed."""
    def __init__(self, agent_id, cls, waypoints, speed, radius):
        self.id = agent_id
        self.cls = cls
        self.waypoints = [np.array(w, dtype=float) for w in waypoints]
        self.speed = speed          # m/s
        self.radius = radius        # m
        self.pos = self.waypoints[0].copy()
        self.wp_idx = 1             # heading toward waypoint 1
        self.vel = np.zeros(2)
        self.done = False

    def step(self, dt):
        """Advance one timestep of dt seconds toward the current waypoint."""
        if self.done or self.wp_idx >= len(self.waypoints):
            self.vel = np.zeros(2)
            self.done = True
            return
        target = self.waypoints[self.wp_idx]
        to_target = target - self.pos
        dist = np.linalg.norm(to_target)
        step_len = self.speed * dt
        if dist <= step_len:            # reached this waypoint
            self.pos = target.copy()
            self.wp_idx += 1
            self.vel = np.zeros(2) if self.wp_idx >= len(self.waypoints) \
                       else (self.waypoints[self.wp_idx] - self.pos)
        else:
            direction = to_target / dist
            self.pos = self.pos + direction * step_len
            self.vel = direction * self.speed


def simulate(agents, duration=10.0, hz=10.0):
    """Step all agents forward; return list of per-timestep snapshots."""
    dt = 1.0 / hz
    n_steps = int(duration * hz)
    history = []
    for step in range(n_steps):
        t = step * dt
        snapshot = []
        for a in agents:
            a.step(dt)
            snapshot.append({
                "t": round(t, 2), "id": a.id, "cls": a.cls,
                "x": a.pos[0], "y": a.pos[1],
                "vx": a.vel[0], "vy": a.vel[1],
                "radius": a.radius,
            })
        history.append(snapshot)
    return history


def label_timestep(agents_snapshot):
    """
    Label one snapshot SAFE / CAUTION / IMMINENT from the closest
    person-forklift pair. Gates on both TTC and distance so that a low TTC
    at long range does not spuriously fire IMMINENT. Returns (label, min_ttc).
    """
    from src.risk.ttc import time_to_collision
    import numpy as np

    persons = [a for a in agents_snapshot if a["cls"] == "person"]
    forklifts = [a for a in agents_snapshot if a["cls"] == "forklift"]

    min_ttc = np.inf
    min_dist = np.inf
    for p in persons:
        for f in forklifts:
            dist = np.hypot(f["x"] - p["x"], f["y"] - p["y"])
            min_dist = min(min_dist, dist)
            ttc = time_to_collision(
                (f["x"], f["y"]), (f["vx"], f["vy"]),
                (p["x"], p["y"]), (p["vx"], p["vy"]),
                radius_a=f["radius"], radius_b=p["radius"],
            )
            min_ttc = min(min_ttc, ttc)

    if min_ttc < 1.5 and min_dist < 4.0:
        label = "IMMINENT"
    elif min_ttc < 3.0 and min_dist < 8.0:
        label = "CAUTION"
    else:
        label = "SAFE"
    return label, min_ttc


def random_scenario(rng, aisle_w=3.5, aisle_l=20.0, force_conflict=False):
    """Generate one random scenario: 1 forklift, 1-2 pedestrians. Sparse by design."""
    import numpy as np
    agents = []

    # ONE forklift driving the aisle
    fx = rng.uniform(0.5, aisle_w - 0.5)
    start_y, end_y = (0, aisle_l) if rng.random() < 0.5 else (aisle_l, 0)
    fork_speed = rng.uniform(2.0, 4.0)
    agents.append(Agent("F0", "forklift", [(fx, start_y), (fx, end_y)],
                        speed=fork_speed, radius=1.5))

    n_ped = rng.integers(1, 3)   # 1 or 2 pedestrians

    for i in range(n_ped):
        if force_conflict and i == 0:
            # Time a pedestrian to cross the forklift's lane roughly when the
            # forklift arrives there — a genuine crossing conflict, then exit.
            # Forklift reaches y=cross_y at t = |cross_y - start_y| / speed
            cross_y = rng.uniform(6, aisle_l - 6)
            t_fork = abs(cross_y - start_y) / fork_speed
            ped_speed = rng.uniform(1.0, 1.6)
            # pedestrian should reach fx around t_fork -> start it so timing lines up
            # start off to the side, walk across and exit past the far edge
            side_start = -0.5 if rng.random() < 0.5 else aisle_w + 0.5
            side_end = aisle_w + 0.5 if side_start < 0 else -0.5
            agents.append(Agent(f"P{i}", "person",
                                [(side_start, cross_y), (side_end, cross_y)],
                                speed=ped_speed, radius=0.4))
        else:
            # random benign pedestrian, short path somewhere in the aisle
            sx, sy = rng.uniform(0, aisle_w), rng.uniform(0, aisle_l)
            ex, ey = rng.uniform(0, aisle_w), rng.uniform(0, aisle_l)
            agents.append(Agent(f"P{i}", "person",
                                [(sx, sy), (ex, ey)],
                                speed=rng.uniform(1.0, 1.6), radius=0.4))
    return agents