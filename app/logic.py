import dimod
import neal
import numpy as np
import pandas as pd
import pulp

# --- Load Data ---
df_raw = pd.read_excel('emission-spot-primary-market-auction-report-2025-data.xlsx', header=5)
df_raw = df_raw.dropna(subset=['Date'])

keep = {
    'Date': 'date',
    'Auction Name': 'auction_type',
    'Auction Price €/tCO2': 'clearing_price',
    'Minimum Bid €/tCO2': 'min_bid',
    'Maximum Bid €/tCO2': 'max_bid',
    'Mean €/tCO2': 'mean_bid',
    'Median €/tCO2': 'median_bid',
    'Auction Volume tCO2': 'supply',
    'Number of bids submitted': 'n_bids',
    'Number of successful bids': 'n_successful',
    'Average bid size': 'avg_bid_size',
    'Standard deviation of bid volume per bidder': 'std_bid_volume',
    'Cover Ratio': 'cover_ratio',
    'Total Number of Bidders': 'n_bidders',
    'Number of Successful Bidders': 'n_successful_bidders',
    'Innovation Fund\n(IF)': 'fund_IF',
    'InnoFund RRF\n(IX)': 'fund_IX',
    'Modernisation Fund\n(MF)': 'fund_MF',
    'MS RRF\n(MX)': 'fund_MX',
    'Social Climate Fund\n(SF)': 'fund_SF',
}

df = df_raw[list(keep.keys())].rename(columns=keep).copy()
df['date'] = pd.to_datetime(df['date'])

type_map = {
    'Auction 4. Period CAP3 EU': 'EUA',
    'Auction 4. Period DE': 'EUA_DE',
    'Auction 4. Period CAP3 PL': 'EUA_PL',
    'Auction 4. Period CAP3 NIR': 'EUAA',
}
df['auction_type'] = df['auction_type'].map(type_map).fillna(df['auction_type'])

sessions = {
    tau: df[df['auction_type'] == tau].iloc[len(df[df['auction_type'] == tau])//2]
    for tau in ['EUA', 'EUA_DE', 'EUA_PL', 'EUAA']
}


def reconstruct_bids_multisession(n_bids_per_type=None, package_fraction=0.3, seed=42):
    rng = np.random.default_rng(seed)
    all_bids = []

    for tau, session in sessions.items():
        # use slider value if provided, else real session n_bids
        n = n_bids_per_type if n_bids_per_type is not None else int(session['n_bids'])

        mu_q    = session['avg_bid_size']
        sigma_q = session['std_bid_volume']
        p_clear = session['clearing_price']
        p_min   = session['min_bid']
        p_max   = session['max_bid']
        supply  = session['supply']

        var   = sigma_q**2
        mu_ln = np.log(mu_q**2 / np.sqrt(var + mu_q**2))
        sg_ln = np.sqrt(np.log(1 + var / mu_q**2))
        quantities = rng.lognormal(mu_ln, sg_ln, n)
        prices     = rng.triangular(p_min, p_clear, p_max, n)

        bids = pd.DataFrame({
            'quantity':        quantities,
            'price':           prices,
            'value':           quantities * prices,
            'type':            tau,
            'supply':          supply,
            'package_partner': -1,
            'package_bonus':   0.0,
        })

        real_cover    = session['cover_ratio']
        current_cover = bids['quantity'].sum() / supply
        bids['quantity'] *= (real_cover / current_cover)
        bids['value']     = bids['quantity'] * bids['price']
        all_bids.append(bids)

    combined = pd.concat(all_bids, ignore_index=True)

    euaa_idx = combined[combined['type'] == 'EUAA'].index.tolist()
    eua_idx  = combined[combined['type'] == 'EUA'].index.tolist()
    n_packages    = int(len(euaa_idx) * package_fraction)
    euaa_partners = rng.choice(euaa_idx, size=n_packages, replace=False)
    eua_partners  = rng.choice(eua_idx,  size=n_packages, replace=False)

    for euaa_i, eua_k in zip(euaa_partners, eua_partners):
        w = 0.15 * min(combined.loc[euaa_i, 'value'], combined.loc[eua_k, 'value'])
        combined.loc[euaa_i, 'package_partner'] = eua_k
        combined.loc[eua_k,  'package_partner'] = euaa_i
        combined.loc[euaa_i, 'package_bonus']   = w
        combined.loc[eua_k,  'package_bonus']   = w

    return combined


def normalize(bids_raw):
    bids_norm = bids_raw.copy()
    for col in ['quantity', 'value', 'supply', 'package_bonus']:
        bids_norm[col] = bids_norm[col].astype(float)
    for tau in bids_norm['type'].unique():
        mask      = bids_norm['type'] == tau
        S_tau     = bids_norm.loc[mask, 'supply'].iloc[0]
        v_max_tau = bids_norm.loc[mask, 'value'].max()
        bids_norm.loc[mask, 'quantity']      /= S_tau
        bids_norm.loc[mask, 'supply']         = 1.0
        bids_norm.loc[mask, 'value']         /= v_max_tau
        bids_norm.loc[mask, 'package_bonus'] /= v_max_tau
    return bids_norm


def build_qubo(bids, A=1.0, C_mult=100.0):
    bids = bids.reset_index(drop=True).copy()
    n    = len(bids)

    for col in ['quantity', 'value', 'supply', 'package_bonus']:
        bids[col] = bids[col].astype(float)

    S_tau     = bids.groupby('type')['supply'].transform('first')
    v_max_tau = bids.groupby('type')['value'].transform('max')

    bids['quantity']      = bids['quantity'] / S_tau
    bids['supply']        = 1.0
    bids['value']         = bids['value'] / v_max_tau
    bids['package_bonus'] = bids['package_bonus'] / v_max_tau

    v     = bids['value'].to_numpy()
    q     = bids['quantity'].to_numpy()
    types = bids['type'].to_numpy()
    w_max = bids['package_bonus'].max()

    Q = np.zeros((n, n))
    Q[np.diag_indices(n)] -= A * v

    partner     = bids['package_partner'].to_numpy().astype(int)
    has_partner = (partner != -1) & (partner > np.arange(n))
    i_idx = np.where(has_partner)[0]
    k_idx = partner[i_idx]
    Q[i_idx, k_idx] -= A * bids['package_bonus'].to_numpy()[i_idx]

    C_per_type = {}
    for tau in bids['type'].unique():
        idx = np.where(types == tau)[0]
        q_t = q[idx]
        v_t = v[idx]
        S_t = bids.loc[idx[0], 'supply']

        C_tau          = C_mult * (v_t.max() + w_max + 1.0) / (2 * q_t.max())
        C_per_type[tau] = round(float(C_tau), 2)

        Q[idx, idx] += C_tau * q_t**2 - 2 * C_tau * S_t * q_t
        outer        = 2 * C_tau * np.outer(q_t, q_t)
        iu, ku       = np.triu_indices(len(idx), k=1)
        Q[idx[iu], idx[ku]] += outer[iu, ku]

    return Q, {
        'A':          A,
        'C_per_type': C_per_type,
        'C_mult':     C_mult,
        'n_bids':     n,
        'n_packages': int((bids['package_partner'] != -1).sum() // 2),
    }