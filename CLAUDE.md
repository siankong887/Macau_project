# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Macau OD Matrix Bayesian Estimation System — estimates Origin-Destination traffic matrices for Macau's 23 transportation zones from camera-observed vehicle counts using Bayesian inference. Four vehicle types: car, bus, truck, motorcycle.

Two main subsystems:
- **`sumo_part/`** — Core estimation pipeline (Bayesian OD inference + SUMO simulation validation)
- **`CV_part/`** — Computer vision module (YOLO-based vehicle detection/counting from HLS streams)

## Common Commands

All commands run from the `sumo_part/` directory with the `.venv` activated:

```bash
# Full pipeline: estimate all vehicle types + SUMO simulation
python -m sumo_part run-all --K 3 --theta 0.1 --beta 100 --mode batch

# Individual steps
python -m sumo_part prepare-network
python -m sumo_part prepare-observations --vehicle car
python -m sumo_part build-h --K 3 --theta 0.1
python -m sumo_part estimate-od --vehicle car --mode batch --beta 100
python -m sumo_part prepare-sumo-assets --vehicle car

# Tests (from sumo_part/ directory)
python -m unittest discover tests
python -m unittest tests.test_sumo_due              # single test module
python -m unittest tests.test_sumo_due.TestClass.test_method  # single test
```

## Architecture

### Data Flow (sumo_part)

```
Camera Config + Road Network Graph + Camera Observations
  → prepare_network: load GraphML, map zone centroids to nearest nodes
  → prepare_gates: map camera gates to road edges
  → build_assignment: K-shortest paths + Logit model → sparse allocation matrix H
  → prepare_observations: parse camera data → observation vector g
  → Bayesian update: g = H·x + ε → posterior OD matrix with 95% CI
  → SUMO simulation: validate posterior OD via microsimulation (post-processing only)
```

### Key Modules (sumo_part/sumo_part/)

| Module | Role |
|---|---|
| `cli.py` | argparse CLI with 6 subcommands |
| `paths.py` | `ProjectPaths` dataclass — all file paths derived from `from_root()` |
| `types.py` | Core dataclasses: `GateInfo`, `ObservationData`, `PathCache`, `SumoAssignmentConfig`, etc. |
| `network.py` | Load road graph (NetworkX MultiDiGraph), assign zones to nodes |
| `observations.py` | Parse camera JSON → gate/edge/turning observation vectors |
| `assignment.py` | Build sparse CSR allocation matrix H (K-shortest paths + Logit choice model) |
| `prior.py` | Uniform prior distribution construction |
| `bayes.py` | Three Bayesian update modes: `batch` (default), `sequential`, `error_free` |
| `sumo_backend.py` | SUMO integration: TAZ mapping, demand generation, DUA iteration, route parsing |
| `pipeline.py` | Orchestrates the full workflow; `run_all()` is the main entry point |

### Key Parameters

| Param | Default | Meaning |
|---|---|---|
| `K` | 3 | Number of candidate shortest paths per OD pair |
| `theta` | 0.1 | Logit temperature for path choice probability |
| `beta` | 100.0 | Prior variance multiplier (higher = weaker/less informative prior) |
| `mode` | `batch` | Bayesian update method (`batch`/`sequential`/`error_free`) |

### SUMO's Role

SUMO is a **post-processing validator**, not part of the Bayesian estimation chain. The analytic H (K-shortest paths + Logit) drives estimation; SUMO runs afterward to simulate the posterior OD and produce diagnostic turning statistics.

## Dependencies

Core: `numpy`, `scipy`, `networkx`, `shapely`
SUMO: external binary (detected via `sumolib`/`SUMO_HOME`)
CV module: `ultralytics` (YOLO), `torch`, `PyNvVideoCodec`, `m3u8`, `av`

## Data Layout (sumo_part/)

- `data/network/macau_drive.graphml` — road network graph
- `data/config/a1_copy_2.json` — camera locations and gate definitions
- `data/observations/time_limit.json` — camera traffic counts by time period
- `data/zones/macau_zones_23.geojson` — 23 zone boundaries
- `data/speeds/speed_mapping.csv` — road type speed assignments
- `output/` — OD matrices (CSV), turning ratios (JSON), cached H matrices (NPZ)

## Documentation

Detailed Chinese-language technical documentation is in `sumo_part/sumo_part/README.md` (data flow diagrams, math formulations, module-by-module descriptions).
