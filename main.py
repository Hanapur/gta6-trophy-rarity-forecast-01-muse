"""Forecaster for gta6-trophy-rarity-forecast-01 (reference implementation).

How the page behaves drives the model: Exophase refreshes its counts on its own schedule,
so a case history is a step function -- consecutive identical reads are one observation.
Collapse the history to its distinct states before fitting anything.

Two regimes:

* Several distinct states. Work on the earn rate itself, on the logit scale, against
  x = ln(1 + hours since release). Each trophy gets a local trend (recent states) blended
  with its long trend, the slope damped slightly for the extrapolation and shrunk toward
  its trophy-class median, anchored at the last observed rate. The spread is never
  narrower than the house trend baseline's own formula -- a random-walk term in the
  horizon plus the fit's residual scatter -- plus slope uncertainty, so an honest
  interval covers about nine times in ten. A frozen-page branch (no refresh lands
  before the target, so the answer is the last read verbatim) is mixed in with its
  Poisson probability; over a multi-day horizon that weight vanishes.

* A single distinct state (the launch-night practice shape). Nothing can be fitted:
  the answer is a lottery between the page not refreshing (rates exactly as last read)
  and it refreshing (the tracked pool multiplies, earners catch up by trophy class).
  The five quantiles are the exact quantiles of that two-branch mixture on the logit
  scale.

Runs offline, standard library only, deterministic: the same case always writes the
same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)  # the game's release day, 00:00 UTC

# --- launch priors, used only when a history holds a single distinct state -----------------
# Relative growth rate of the tracked pool per hour at age h hours: g(h) = G0 / (h + G1).
PRIOR_G0, PRIOR_G1 = 2.6, 4.0
# The pool multiplier of one Exophase refresh on launch night, and the mean hours between
# refreshes then; both soften with age in `refresh_prior`.
PRIOR_JUMP_LAUNCH = 5.0
PRIOR_REFRESH_HOURS = 5.0
# Earner growth over a horizon, by trophy class, as a multiplier 1 + A * (1 - exp(-h / TAU)).
# Sized from the one launch transition on record (638 -> 3,868 tracked): story earners grew
# about x1.3, casual-activity earners about x1.2 at the median with wide scatter, grind
# earners barely at all.
CLASS_GROWTH = {  # class: (A, TAU hours, sigma_lnE)
    "story": (0.80, 6.0, 0.55),
    "activity": (0.50, 6.0, 0.75),
    "grind": (0.15, 10.0, 0.42),
}
SIGMA_FLOOR = 0.22
LOCAL_WINDOW_H = 24.0
SLOPE_DAMP = 0.92


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y: float) -> float:
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, whole numbers as ints (the output shape the kit uses)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(when: datetime) -> float:
    return (when - RELEASE).total_seconds() / 3600.0


def x_of_hours(h: float) -> float:
    return math.log(1.0 + max(h, 0.0))


def classify(trophy: dict) -> str:
    """Rough trophy class from its description, for priors and pooled shrinkage groups."""
    desc = (trophy.get("description") or "").lower()
    tier = trophy.get("tier") or ""
    if tier == "platinum" or "100% completion" in desc:
        return "grind"
    if desc.startswith("complete chapter") or "final mission" in desc:
        return "story"
    if any(k in desc for k in ("every", " all ", "collect", "250", "100 locations",
                               "50 challenges", "10 vehicles", "100 vehicles")):
        return "grind"
    if desc.startswith("complete"):
        return "story"
    return "activity"


def distinct_states(case: dict):
    """Collapse consecutive identical reads: a state is (hours, players, [rates])."""
    states = []
    for read in case["history"]:
        rates = read["rates"]
        key = (read["players_tracked"], tuple(rates))
        if states and states[-1][3] == key:
            continue
        h = hours_since_release(parse_time(read["slot"]))
        states.append((h, float(read["players_tracked"]), rates, key))
    return [(h, n, rates) for h, n, rates, _ in states]


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b x. Returns (a, b, resid_sd, mean_x, sxx, sum_w)."""
    sw = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0, 0.0, mx, sxx, sw
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(len(xs) - 2, 1)
    sd = math.sqrt(sum(w * r * r for r, w in zip(resid, ws)) / max(sw, 1e-9) * len(xs) / dof) if len(xs) > 2 else 0.0
    return a, b, sd, mx, sxx, sw


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mixture_quantile(branches, tau: float) -> float:
    """Quantile of a mixture of normals on the logit scale, by bisection.

    branches: list of (weight, centre, sigma). Deterministic: fixed iteration count.
    """
    lo = min(c - 7.0 * s for _, c, s in branches) - 0.5
    hi = max(c + 7.0 * s for _, c, s in branches) + 0.5
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        cdf = sum(w * norm_cdf((mid - c) / s) for w, c, s in branches)
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def refresh_prior(h_last: float, horizon_h: float):
    """Launch refresh lottery for a single-state history.

    Returns (p_refresh, drift_ln, jump_ln, mean_interval_h): the chance a refresh lands in
    the window, the pool's smooth ln-growth, the ln pool multiplier a refresh reveals, and
    the mean hours between refreshes at this age. A page that has not refreshed shows the
    old rates untouched, so the two branches are kept apart, not averaged.
    """
    h0 = max(h_last, 0.0)
    growth = PRIOR_G0 * math.log((h0 + horizon_h + PRIOR_G1) / (h0 + PRIOR_G1))
    drift = growth * 0.35  # only the smooth part accrues without a refresh
    age_soft = 1.0 / (1.0 + h0 / 24.0)
    jump = 1.0 + (PRIOR_JUMP_LAUNCH - 1.0) * age_soft
    interval = PRIOR_REFRESH_HOURS * (1.0 + h0 / 12.0)
    p_ref = 1.0 - math.exp(-max(horizon_h, 0.0) / interval)
    return p_ref, drift, math.log(jump), interval


def state_gap_hours(states, h_last_read: float) -> float:
    """Typical hours between refreshes, from the gaps between distinct states.

    The wait since the last state change counts too: a page unchanged for ten hours
    before the checkpoint is refreshing slowly, whatever the earlier gaps were.
    """
    if len(states) < 2:
        return 6.0
    gaps = [states[k + 1][0] - states[k][0] for k in range(len(states) - 1)]
    since_change = max(h_last_read - states[-1][0], 0.0)
    return max(sum(gaps) / len(gaps), since_change, 0.5)


def forecast_single_state(case, trophies, states, h_last, horizon_h):
    """Launch-lottery forecast: exact quantiles of the refresh / no-refresh mixture."""
    p_ref, drift_ln, jump_ln, interval_h = refresh_prior(h_last, horizon_h)
    n_last = states[-1][1]
    n_refresh = n_last * math.exp(drift_ln + jump_ln)
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = states[-1][2][i]
        last_e = (last_rate / 100.0) * n_last if last_rate is not None else 0.0
        if last_rate is not None and last_e >= n_last - 1e-9:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        cls = classify(trophy)
        big_a, tau, s_e = CLASS_GROWTH[cls]
        span = horizon_h + 0.5 * interval_h
        mult = 1.0 + big_a * (1.0 - math.exp(-span / tau)) if span > 0 else 1.0
        e_ref = min(max(last_e, 0.35) * mult, 0.999 * n_refresh)
        rate_ref = 100.0 * e_ref / n_refresh
        branches = [
            (1.0 - p_ref, logit(last_rate if last_rate is not None else 0.0),
             math.sqrt(0.06 ** 2 + 0.10 ** 2)),
            (p_ref, logit(rate_ref),
             math.sqrt(s_e ** 2 + 0.38 ** 2 + 0.10 ** 2 + SIGMA_FLOOR ** 2)),
        ]
        q = [round4(inv_logit(mixture_quantile(branches, t))) for t in LEVELS]
        q = [min(100.0, max(0.0, float(v))) for v in q]
        q.sort()
        rows.append({"id": trophy["id"], "q": q})
    return rows


def forecast_trend(states, trophies, h_last, h_target, gap_h):
    """Rate-space trend forecast for histories with several distinct states."""
    x_t = x_of_hours(h_target)
    horizon_h = max(h_target - h_last, 0.0)
    d_days = horizon_h / 24.0
    h_state_last = states[-1][0]
    # First pass: per-trophy local/long fits on the logit of the rate.
    fits = []
    for i in range(len(trophies)):
        pts = []
        for h, n, rates in states:
            r = rates[i]
            if r is None:
                continue
            floor = 100.0 * 0.35 / max(n, 1.0)  # a third of an earner, in rate points
            pts.append((x_of_hours(h), logit(max(r, floor)), h))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        a_l, b_l, sd_l, mx_l, sxx_l, _ = wls_line(xs, ys, [1.0] * len(xs))
        loc = [p for p in pts if p[2] >= h_state_last - LOCAL_WINDOW_H]
        local = None
        if len(loc) >= 3 and loc[-1][0] - loc[0][0] >= 0.2:
            lxs = [p[0] for p in loc]
            lys = [p[1] for p in loc]
            local = wls_line(lxs, lys, [1.0] * len(lxs))
        if local is not None:
            slope = 0.7 * local[1] + 0.3 * b_l
            resid = max(local[2], sd_l * 0.7)
            se_b = local[2] / math.sqrt(max(local[4], 1e-9)) if local[4] > 1e-9 else 0.5
        else:
            slope = b_l
            resid = sd_l
            se_b = sd_l / math.sqrt(max(sxx_l, 1e-9)) if sxx_l > 1e-9 else 0.5
        fits.append({
            "slope": slope, "resid": resid, "se_b": se_b,
            "y_last": ys[-1] if ys else logit(0.0), "x_last": xs[-1] if xs else x_t,
            "last_rate": states[-1][2][i],
            "last_e": (states[-1][2][i] / 100.0) * states[-1][1]
                      if states[-1][2][i] is not None else 0.0,
        })
    # Class-median slopes, for shrinkage.
    by_class = {}
    for i, trophy in enumerate(trophies):
        by_class.setdefault(classify(trophy), []).append(fits[i]["slope"])
    class_med = {}
    for cls, vals in by_class.items():
        vals = sorted(vals)
        class_med[cls] = vals[len(vals) // 2]
    w_frozen = min(math.exp(-horizon_h / max(gap_h, 0.5)), 0.6) if horizon_h > 0 else 0.0
    few_state = max(0, 4 - len(states)) * 0.25
    rows = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        last_rate = f["last_rate"]
        if last_rate is not None and f["last_e"] >= states[-1][1] - 1e-9:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        w_own = min(f["last_e"], 30.0) / 10.0 + 1.0
        slope = (w_own * f["slope"] + 3.0 * class_med[classify(trophy)]) / (w_own + 3.0)
        dx = x_t - f["x_last"]
        centre = f["y_last"] + slope * dx * SLOPE_DAMP
        resid = max(f["resid"], 0.25)
        sigma = math.sqrt((0.5 * math.sqrt(d_days)) ** 2 + resid ** 2
                          + (f["se_b"] * dx) ** 2 + few_state ** 2 + 0.15 ** 2)
        sigma = max(sigma, 0.40)
        if w_frozen > 0.01 and last_rate is not None:
            branches = [(1.0 - w_frozen, centre, sigma),
                        (w_frozen, logit(last_rate), 0.08)]
            q = [round4(inv_logit(mixture_quantile(branches, t))) for t in LEVELS]
        else:
            q = [round4(inv_logit(centre + z * sigma)) for z in Z]
        q = [min(100.0, max(0.0, float(v))) for v in q]
        q.sort()
        rows.append({"id": trophy["id"], "q": q})
    return rows


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    states = distinct_states(case)
    if not states:
        raise ValueError("case has no history")
    h_last = hours_since_release(parse_time(case["history"][-1]["slot"]))
    h_target = h_last + case["horizon_minutes"] / 60.0
    if case.get("target"):
        h_target = hours_since_release(parse_time(case["target"]))
    horizon_h = max(h_target - h_last, 0.0)
    if len(states) == 1:
        rows = forecast_single_state(case, trophies, states, h_last, horizon_h)
    else:
        gap_h = state_gap_hours(states, h_last)
        rows = forecast_trend(states, trophies, h_last, h_target, gap_h)
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
