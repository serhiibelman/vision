# Spikes

Throwaway experiments, **not part of the `car_tracker` package** and not shipped in
the wheel. They are kept in the repository as evidence of how the design decisions in
`../../DECISIONS.md` were reached — each one answers a go/no-go question that was open
before the corresponding module was written.

| Script | Question it answered | Decision |
|---|---|---|
| `yolo_spike.py` | Does COCO-pretrained YOLO detect cars in nadir footage, and does tiling matter? | D1 — DOTA-OBB weights, 640 px tiles at native resolution |
| `compare.py` | Manual side-by-side inspection of the candidate detectors | supporting evidence for D1 |

They import `car_tracker` but nothing imports them. They are linted (`ruff check .`
covers this directory) but deliberately **not tested** and **not maintained** — they
are a record of a past experiment, so expect them to reference paths and outputs from
the run that produced the numbers in `DECISIONS.md` rather than to work unchanged
against arbitrary input.
