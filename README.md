# Telecom Backhaul SLA QUBO (D-Wave Ocean)

`backhaul_qubo_ocean.py` implements a telecom-style QUBO for backhaul allocation/throttling with:
- chunked per-stream bandwidth decisions
- weighted SLA shortfall minimization
- stability penalty vs previous interval
- minimum guarantees and link capacities via quadratic penalties

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install dimod dwave-system dwave-samplers
```

For Leap/QPU runs, configure credentials (for example with `dwave setup`).

## Run

Use local simulated annealing:

```bash
python backhaul_qubo_ocean.py --solver sa --input sample_scenario.json
```

Use default built-in scenario:

```bash
python backhaul_qubo_ocean.py --solver sa
```

Use Leap hybrid solver:

```bash
python backhaul_qubo_ocean.py --solver hybrid --input sample_scenario.json
```

Use QPU:

```bash
python backhaul_qubo_ocean.py --solver qpu --num-reads 1000 --qpu-topology zephyr
```

Write result JSON:

```bash
python backhaul_qubo_ocean.py --solver sa --output result.json
```

## Input format

See `sample_scenario.json`.

Top-level keys:
- `delta_mbps`: allocation chunk size
- `links`: list of `{name, capacity_mbps}`
- `streams`: list of `{name, link, demand_mbps, min_guarantee_mbps, priority_weight, prev_alloc_mbps}`
