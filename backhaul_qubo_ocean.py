#!/usr/bin/env python3
"""Telecom backhaul bandwidth allocation as a QUBO (D-Wave Ocean).

Model overview
- Decision variables allocate bandwidth chunks to each (site x service-class) stream.
- Objective minimizes weighted SLA shortfall and instability vs previous interval.
- Capacity and minimum-guarantee constraints are enforced via quadratic penalties.

This script can run with:
- LeapHybridSampler (cloud) when configured
- DWaveSampler (QPU) when configured
- SimulatedAnnealingSampler (local)
- ExactSolver (local, very small instances)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import dimod


@dataclass(frozen=True)
class Stream:
    name: str
    link: str
    demand_mbps: int
    min_guarantee_mbps: int
    priority_weight: float
    prev_alloc_mbps: int


@dataclass(frozen=True)
class Link:
    name: str
    capacity_mbps: int


def bit_width_for_nonnegative_integer(max_value: int) -> int:
    if max_value <= 0:
        return 0
    return int(math.floor(math.log2(max_value))) + 1


def alloc_var_name(stream_name: str, chunk_idx: int) -> str:
    return f"y::{stream_name}::{chunk_idx}"


def shortfall_var_name(stream_name: str, bit_idx: int) -> str:
    return f"sf::{stream_name}::{bit_idx}"


def guarantee_slack_var_name(stream_name: str, bit_idx: int) -> str:
    return f"gsl::{stream_name}::{bit_idx}"


def capacity_slack_var_name(link_name: str, bit_idx: int) -> str:
    return f"csl::{link_name}::{bit_idx}"


def add_square_penalty(
    bqm: dimod.BinaryQuadraticModel,
    terms: Iterable[Tuple[str, float]],
    constant: float,
    strength: float,
) -> None:
    """Add `strength * (sum(a_i x_i) + constant)^2` to BQM."""
    terms_list = list(terms)
    if not terms_list:
        bqm.offset += strength * (constant**2)
        return

    for i, (vi, ai) in enumerate(terms_list):
        bqm.add_variable(vi, strength * (ai * ai + 2.0 * ai * constant))
        for j in range(i + 1, len(terms_list)):
            vj, aj = terms_list[j]
            bqm.add_interaction(vi, vj, strength * 2.0 * ai * aj)

    bqm.offset += strength * (constant**2)


def build_backhaul_bqm(
    streams: List[Stream],
    links: List[Link],
    delta_mbps: int = 10,
    min_chunks_per_stream: int = 1,
    lambda_shortfall: float = 6.0,
    lambda_stability: float = 0.03,
    lambda_capacity: float = 15.0,
    lambda_min_guarantee: float = 12.0,
) -> Tuple[dimod.BinaryQuadraticModel, Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Build BQM for telecom backhaul allocation.

    Returns
    - bqm
    - max_chunks_by_stream
    - shortfall_bits_by_stream
    - capacity_slack_bits_by_link
    """
    if delta_mbps <= 0:
        raise ValueError("delta_mbps must be positive")

    link_map = {l.name: l for l in links}
    for s in streams:
        if s.link not in link_map:
            raise ValueError(f"stream {s.name} references unknown link {s.link}")

    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)

    max_chunks_by_stream: Dict[str, int] = {}
    shortfall_bits_by_stream: Dict[str, int] = {}
    guarantee_slack_bits_by_stream: Dict[str, int] = {}

    for s in streams:
        target_max = max(s.demand_mbps, s.min_guarantee_mbps, delta_mbps * min_chunks_per_stream)
        n_chunks = int(math.ceil(target_max / delta_mbps))
        max_chunks_by_stream[s.name] = n_chunks

        demand_chunks = int(math.ceil(max(0, s.demand_mbps) / delta_mbps))
        shortfall_bits = bit_width_for_nonnegative_integer(demand_chunks)
        shortfall_bits_by_stream[s.name] = shortfall_bits

        over_guarantee_chunks = max(0, n_chunks - int(math.ceil(max(0, s.min_guarantee_mbps) / delta_mbps)))
        guarantee_slack_bits_by_stream[s.name] = bit_width_for_nonnegative_integer(over_guarantee_chunks)

        # Stability penalty: lambda_stability * (B_s - B_prev)^2
        terms: List[Tuple[str, float]] = []
        for k in range(n_chunks):
            terms.append((alloc_var_name(s.name, k), float(delta_mbps)))
        add_square_penalty(
            bqm,
            terms=terms,
            constant=-float(s.prev_alloc_mbps),
            strength=lambda_stability,
        )

        # Demand shortfall via equality: B_s + SF_s = D_s
        # with SF_s binary-encoded integer in chunk units.
        sf_terms: List[Tuple[str, float]] = []
        sf_linear_weight = lambda_shortfall * s.priority_weight
        for bit in range(shortfall_bits):
            v = shortfall_var_name(s.name, bit)
            coeff = float(delta_mbps * (2**bit))
            sf_terms.append((v, coeff))
            bqm.add_variable(v, sf_linear_weight * coeff)

        all_terms = terms + sf_terms
        add_square_penalty(
            bqm,
            terms=all_terms,
            constant=-float(s.demand_mbps),
            strength=lambda_shortfall,
        )

        # Minimum guarantee soft-hard constraint: B_s - GS_s = G_s
        # GS_s is nonnegative slack capturing bandwidth above guarantee.
        g_terms: List[Tuple[str, float]] = []
        for bit in range(guarantee_slack_bits_by_stream[s.name]):
            g_terms.append(
                (
                    guarantee_slack_var_name(s.name, bit),
                    -float(delta_mbps * (2**bit)),
                )
            )

        add_square_penalty(
            bqm,
            terms=terms + g_terms,
            constant=-float(s.min_guarantee_mbps),
            strength=lambda_min_guarantee,
        )

    # Link capacity constraints: sum(B_s on link) + capacity_slack = C_link
    capacity_slack_bits_by_link: Dict[str, int] = {}
    streams_by_link: Dict[str, List[Stream]] = {l.name: [] for l in links}
    for s in streams:
        streams_by_link[s.link].append(s)

    for link in links:
        terms: List[Tuple[str, float]] = []
        max_allocatable_on_link = 0
        for s in streams_by_link[link.name]:
            n_chunks = max_chunks_by_stream[s.name]
            max_allocatable_on_link += n_chunks * delta_mbps
            for k in range(n_chunks):
                terms.append((alloc_var_name(s.name, k), float(delta_mbps)))

        slack_max = max(0, link.capacity_mbps)
        bits = bit_width_for_nonnegative_integer(int(math.ceil(slack_max / delta_mbps)))
        capacity_slack_bits_by_link[link.name] = bits
        for bit in range(bits):
            terms.append(
                (
                    capacity_slack_var_name(link.name, bit),
                    float(delta_mbps * (2**bit)),
                )
            )

        add_square_penalty(
            bqm,
            terms=terms,
            constant=-float(link.capacity_mbps),
            strength=lambda_capacity,
        )

    return bqm, max_chunks_by_stream, shortfall_bits_by_stream, capacity_slack_bits_by_link


def decode_solution(
    sample: dimod.SampleSet,
    streams: List[Stream],
    links: List[Link],
    delta_mbps: int,
    max_chunks_by_stream: Dict[str, int],
    shortfall_bits_by_stream: Dict[str, int],
) -> Dict[str, object]:
    best = sample.first.sample

    stream_results = []
    by_link_alloc: Dict[str, int] = {l.name: 0 for l in links}

    total_weighted_shortfall = 0.0
    total_weighted_demand = 0.0

    for s in streams:
        n_chunks = max_chunks_by_stream[s.name]
        chunks = sum(int(best.get(alloc_var_name(s.name, k), 0)) for k in range(n_chunks))
        alloc = chunks * delta_mbps

        sf_bits = shortfall_bits_by_stream[s.name]
        sf_chunks = sum(int(best.get(shortfall_var_name(s.name, b), 0)) * (2**b) for b in range(sf_bits))
        shortfall = sf_chunks * delta_mbps

        min_met = alloc >= s.min_guarantee_mbps
        by_link_alloc[s.link] += alloc

        weighted_sf = s.priority_weight * max(0, s.demand_mbps - alloc)
        weighted_dm = s.priority_weight * s.demand_mbps
        total_weighted_shortfall += weighted_sf
        total_weighted_demand += weighted_dm

        stream_results.append(
            {
                "stream": s.name,
                "link": s.link,
                "demand_mbps": s.demand_mbps,
                "min_guarantee_mbps": s.min_guarantee_mbps,
                "prev_alloc_mbps": s.prev_alloc_mbps,
                "alloc_mbps": alloc,
                "shortfall_mbps": max(0, s.demand_mbps - alloc),
                "model_shortfall_mbps": shortfall,
                "priority_weight": s.priority_weight,
                "guarantee_met": min_met,
            }
        )

    link_results = []
    for l in links:
        used = by_link_alloc[l.name]
        cap = l.capacity_mbps
        link_results.append(
            {
                "link": l.name,
                "used_mbps": used,
                "capacity_mbps": cap,
                "utilization": (used / cap) if cap > 0 else None,
                "binding": used >= cap,
                "headroom_mbps": cap - used,
            }
        )

    sla_risk = (total_weighted_shortfall / total_weighted_demand) if total_weighted_demand > 0 else 0.0

    return {
        "energy": sample.first.energy,
        "num_variables": len(sample.variables),
        "sla_risk_score": sla_risk,
        "streams": stream_results,
        "links": link_results,
    }


def sample_bqm(
    bqm: dimod.BinaryQuadraticModel,
    solver: str,
    num_reads: int,
    qpu_topology: Optional[str] = None,
) -> dimod.SampleSet:
    solver = solver.lower()

    if solver == "exact":
        return dimod.ExactSolver().sample(bqm)

    if solver == "sa":
        try:
            from dwave.samplers import SimulatedAnnealingSampler
        except Exception as exc:  # pragma: no cover - import fallback
            raise RuntimeError(
                "SimulatedAnnealingSampler unavailable. Install Ocean package `dwave-samplers`."
            ) from exc
        return SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads)

    if solver == "hybrid":
        try:
            from dwave.system import LeapHybridSampler
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "LeapHybridSampler unavailable. Install `dwave-system` and configure credentials."
            ) from exc
        return LeapHybridSampler().sample(bqm)

    if solver == "qpu":
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("QPU sampler unavailable. Install `dwave-system`.") from exc

        sampler_kwargs = {}
        if qpu_topology:
            sampler_kwargs["topology__type"] = qpu_topology

        sampler = EmbeddingComposite(DWaveSampler(**sampler_kwargs))
        return sampler.sample(bqm, num_reads=num_reads)

    raise ValueError("unknown solver; choose one of: exact, sa, hybrid, qpu")


def default_scenario() -> Tuple[List[Stream], List[Link], int]:
    delta = 10

    links = [
        Link(name="ring_A", capacity_mbps=700),
        Link(name="ring_B", capacity_mbps=500),
    ]

    streams = [
        Stream("site01_voice", "ring_A", 120, 100, 3.5, 110),
        Stream("site01_data", "ring_A", 160, 60, 1.1, 140),
        Stream("site01_fwa", "ring_A", 110, 40, 1.0, 80),
        Stream("site02_voice", "ring_A", 100, 90, 3.2, 95),
        Stream("site02_enterprise", "ring_A", 150, 120, 2.8, 130),
        Stream("site03_data", "ring_B", 180, 70, 1.2, 150),
        Stream("site03_voice", "ring_B", 90, 80, 3.0, 90),
        Stream("site04_enterprise", "ring_B", 140, 110, 2.6, 130),
        Stream("site04_data", "ring_B", 130, 50, 1.0, 100),
    ]

    return streams, links, delta


def load_scenario(path: str) -> Tuple[List[Stream], List[Link], int]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    delta = int(payload.get("delta_mbps", 10))
    links = [Link(name=l["name"], capacity_mbps=int(l["capacity_mbps"])) for l in payload["links"]]

    streams = []
    for s in payload["streams"]:
        streams.append(
            Stream(
                name=s["name"],
                link=s["link"],
                demand_mbps=int(s["demand_mbps"]),
                min_guarantee_mbps=int(s.get("min_guarantee_mbps", 0)),
                priority_weight=float(s.get("priority_weight", 1.0)),
                prev_alloc_mbps=int(s.get("prev_alloc_mbps", 0)),
            )
        )
    return streams, links, delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Telecom backhaul SLA-aware QUBO with D-Wave Ocean")
    parser.add_argument("--input", help="Path to scenario JSON. If omitted, uses built-in scenario.")
    parser.add_argument("--solver", default="sa", choices=["exact", "sa", "hybrid", "qpu"])
    parser.add_argument("--num-reads", type=int, default=300)
    parser.add_argument("--min-chunks", type=int, default=1)

    parser.add_argument("--lambda-shortfall", type=float, default=6.0)
    parser.add_argument("--lambda-stability", type=float, default=0.03)
    parser.add_argument("--lambda-capacity", type=float, default=15.0)
    parser.add_argument("--lambda-min-guarantee", type=float, default=12.0)

    parser.add_argument("--qpu-topology", default=None, help="Optional QPU topology filter (e.g. zephyr)")
    parser.add_argument("--output", default=None, help="Optional path to write result JSON")

    args = parser.parse_args()

    if args.input:
        streams, links, delta_mbps = load_scenario(args.input)
    else:
        streams, links, delta_mbps = default_scenario()

    bqm, max_chunks, sf_bits, _ = build_backhaul_bqm(
        streams=streams,
        links=links,
        delta_mbps=delta_mbps,
        min_chunks_per_stream=args.min_chunks,
        lambda_shortfall=args.lambda_shortfall,
        lambda_stability=args.lambda_stability,
        lambda_capacity=args.lambda_capacity,
        lambda_min_guarantee=args.lambda_min_guarantee,
    )

    sampleset = sample_bqm(
        bqm,
        solver=args.solver,
        num_reads=args.num_reads,
        qpu_topology=args.qpu_topology,
    )

    result = decode_solution(
        sampleset,
        streams=streams,
        links=links,
        delta_mbps=delta_mbps,
        max_chunks_by_stream=max_chunks,
        shortfall_bits_by_stream=sf_bits,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(f"Solver: {args.solver}")
    print(f"Energy: {result['energy']:.3f}")
    print(f"Variables: {result['num_variables']}")
    print(f"SLA risk score: {result['sla_risk_score']:.4f}")

    print("\nPer-stream allocation:")
    for r in sorted(result["streams"], key=lambda x: x["stream"]):
        print(
            f"  {r['stream']:18s} link={r['link']:7s} "
            f"alloc={r['alloc_mbps']:4d} demand={r['demand_mbps']:4d} "
            f"guarantee={r['min_guarantee_mbps']:4d} "
            f"shortfall={r['shortfall_mbps']:4d} guarantee_met={r['guarantee_met']}"
        )

    print("\nLink diagnostics:")
    for lr in result["links"]:
        util = "n/a" if lr["utilization"] is None else f"{100.0 * lr['utilization']:.1f}%"
        print(
            f"  {lr['link']:7s} used={lr['used_mbps']:4d} cap={lr['capacity_mbps']:4d} "
            f"util={util:>6s} binding={lr['binding']} headroom={lr['headroom_mbps']:4d}"
        )


if __name__ == "__main__":
    main()
