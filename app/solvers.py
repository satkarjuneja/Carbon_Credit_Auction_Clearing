import dimod
import neal
import pulp
import numpy as np


def solve_sa(Q, bids_norm, bids_raw, num_reads=200, num_sweeps=1000):
    """
    Q         — QUBO matrix built from normalized bids
    bids_norm — normalized bids, used for feasibility check
    bids_raw  — raw bids in real euros/tCO2, used for welfare reporting
    """
    bids_norm = bids_norm.reset_index(drop=True)
    bids_raw  = bids_raw.reset_index(drop=True)
    n         = len(bids_norm)

    bqm = dimod.BinaryQuadraticModel(vartype='BINARY')
    for i in range(n):
        bqm.add_variable(i, Q[i, i])
    for i in range(n):
        for k in range(i+1, n):
            if Q[i, k] != 0.0:
                bqm.add_interaction(i, k, Q[i, k])

    sampler = neal.SimulatedAnnealingSampler()
    result  = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps)

    best = result.first.sample
    x    = np.array([best[i] for i in range(n)])

    # Welfare from raw bids
    accepted_raw  = bids_raw[x == 1].copy()
    welfare       = accepted_raw['value'].sum()
    pkg_bonus     = sum(
        bids_raw.loc[i, 'package_bonus']
        for i in accepted_raw.index
        if bids_raw.loc[i, 'package_partner'] in accepted_raw.index
        and bids_raw.loc[i, 'package_partner'] != -1
    )

    # Feasibility from normalized bids supply=1.0 per type
    accepted_norm = bids_norm[x == 1].copy()
    feasible      = all(
        accepted_norm[accepted_norm['type'] == tau]['quantity'].sum() <= 1.0
        for tau in bids_norm['type'].unique()
    )

    return {
        'x':          x,
        'welfare':    welfare,
        'pkg_bonus':  pkg_bonus,
        'n_accepted': int(x.sum()),
        'feasible':   feasible,
        'energy':     result.first.energy,
    }


def solve_ip(bids_raw):
    """
    Always runs on raw bids — IP handles constraints natively,
    no normalization needed.
    """
    bids  = bids_raw.reset_index(drop=True)
    n     = len(bids)
    prob  = pulp.LpProblem("auction_clearing", pulp.LpMaximize)
    x     = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n)]

    # Base welfare
    base = pulp.lpSum(bids.loc[i, 'value'] * x[i] for i in range(n))

    # Package bonus linearised via auxiliary z_ik
    pkg_vars        = {}
    pkg_bonus_terms = []
    seen            = set()
    for i in range(n):
        k = int(bids.loc[i, 'package_partner'])
        if k != -1 and (i, k) not in seen and (k, i) not in seen:
            seen.add((i, k))
            z = pulp.LpVariable(f"z_{i}_{k}", cat='Binary')
            pkg_vars[(i, k)] = z
            prob += z <= x[i]
            prob += z <= x[k]
            prob += z >= x[i] + x[k] - 1
            pkg_bonus_terms.append(bids.loc[i, 'package_bonus'] * z)

    prob += base + pulp.lpSum(pkg_bonus_terms)

    # Supply constraint per type raw quantities vs raw supply
    for tau in bids['type'].unique():
        idx = bids[bids['type'] == tau].index.tolist()
        S   = bids.loc[idx[0], 'supply']
        prob += pulp.lpSum(bids.loc[i, 'quantity'] * x[i] for i in idx) <= S

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    x_val    = np.array([pulp.value(x[i]) for i in range(n)])
    accepted = bids[x_val == 1].copy()
    pkg_bonus = sum(
        pulp.value(z) * bids.loc[i, 'package_bonus']
        for (i, k), z in pkg_vars.items()
    )
    feasible = all(
        accepted[accepted['type'] == tau]['quantity'].sum()
        <= bids[bids['type'] == tau]['supply'].iloc[0]
        for tau in bids['type'].unique()
    )

    return {
        'x_val':      x_val,
        'welfare':    pulp.value(prob.objective),
        'pkg_bonus':  pkg_bonus,
        'n_accepted': int(x_val.sum()),
        'feasible':   feasible,
        'status':     pulp.LpStatus[prob.status],
    }

def repair_solution(x, bids_norm, bids_raw):
    x         = x.copy()
    bids_norm = bids_norm.reset_index(drop=True)
    bids_raw  = bids_raw.reset_index(drop=True)
    for tau in bids_norm['type'].unique():
        mask = bids_norm['type'] == tau
        cap  = bids_norm.loc[mask, 'supply'].iloc[0]
        while True:
            accepted = mask & (x == 1)
            if bids_norm.loc[accepted, 'quantity'].sum() <= cap:
                break
            x[bids_raw.loc[accepted, 'value'].idxmin()] = 0
    return x