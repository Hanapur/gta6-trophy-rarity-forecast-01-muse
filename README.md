# GTA 6 Trophy Rarity Forecast

Entry for make.fun competition `gta6-trophy-rarity-forecast-01`: forecast every
PlayStation 5 trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five
quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) per trophy.

## Approach

Exophase refreshes its counts on its own schedule, so a case history is a step
function — consecutive identical reads are one observation. The program first
collapses the history to its distinct states, then:

- **Several distinct states:** model each trophy's earn rate on the logit scale
  against `x = ln(1 + hours since release)`. A local trend (recent states)
  blended with the long trend, slope damped for extrapolation and shrunk toward
  the trophy-class median, anchored at the last observed rate. The spread is
  never narrower than the house trend baseline's own formula (random-walk term
  in the horizon plus residual scatter, plus slope uncertainty), mixed with a
  frozen-page branch weighted by its Poisson probability.
- **A single distinct state (launch-night shape):** nothing can be fitted, so the
  answer is a lottery between the page not refreshing (rates exactly as last
  read) and it refreshing (tracked pool multiplies, earners catch up by trophy
  class — story / activity / grind). The five quantiles are the exact quantiles
  of that two-branch mixture on the logit scale.

## Running

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one `<id>.out` per
case. Python 3 standard library only, no network, deterministic (same case
always writes the same bytes), well within the 8-second-per-case limit.

Practice scores (kit practice cases p00 / p01): pinball loss ≈ 0.81 / 0.85 vs
the house persistence baseline 1.508, with 100% interval coverage.

## License

MIT — see `LICENSE`.
