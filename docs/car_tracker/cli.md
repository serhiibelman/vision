
# cli

`car_tracker/src/car_tracker/cli.py` — command line entry point. Implemented.

Installed as `car-tracker` via `[project.scripts]`.

## Commands

| Command | Reads | Writes |
|---|---|---|
| `telemetry` | SRT | `outputs/telemetry.csv`, optional flight-path map |
| `calibrate` | video + SRT | `outputs/calibration.json` |
| `detect` | video | `outputs/detections.csv` |
| `track` | detections + SRT + calibration | `outputs/tracks.csv` |
| `map` | tracks | `results/map.html`, `results/tracks.geojson`, `outputs/track_features.csv` |
| `all` | video + SRT | everything above |

## Why file boundaries

Detection over 4979 frames takes ~10–15 min on a GPU and ~16 h on a CPU; every other
stage takes seconds. Staging through files means the expensive pass runs once while
thresholds and the map are re-iterated freely.

`all` reuses an existing `calibration.json` unless `--recalibrate` is passed, for the
same reason.

## Flags

| Flag | Applies to | Meaning |
|---|---|---|
| `--video`, `--srt` | most | input paths, default `../tech-assignment/` |
| `--stride N` | `detect`, `all` | process every Nth frame |
| `--start`, `--end` | `detect`, `all` | inclusive frame range |
| `--limit N` | `detect`, `all` | stop after N frames |
| `--conf`, `--weights` | `detect`, `all` | detector threshold and weights |
| `--pairs N` | `calibrate`, `all` | frame pairs used for calibration |
| `--recalibrate` | `all` | ignore a stored calibration |

## Calibration persistence

`save_calibration` / `load_calibration` round-trip a `Calibration` through JSON. A
missing file yields uncalibrated defaults rather than an error, so stages run in any
order — at the cost of accuracy, which the residual in the report makes visible.

## Examples

```bash
car-tracker all                                        # full pipeline
car-tracker detect --start 1000 --end 1300 --stride 3  # a 10 s window on a CPU
car-tracker map                                        # redraw after tuning thresholds
```

## Tests

`car_tracker/tests/test_cli.py` — 20 tests. Every subcommand parses, frame-range
resolution, calibration JSON round-trip, and the `telemetry` command end to end against
a generated SRT.
