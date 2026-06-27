import streamlit as st
import numpy as np
import pandas as pd
import logic
import solvers

st.set_page_config(page_title="EU ETS Auction Clearing", layout="wide")

st.title("EU ETS Carbon Credit Auction Clearing")
st.markdown("**Quantum-inspired optimisation via Ising Hamiltonian formulation**")
st.markdown("---")

with st.sidebar:
    st.header("Auction Parameters")
    n_bids = st.slider("Number of bids per type", min_value=10, max_value=100, value=60, step=10)
    package_fraction = st.slider("Package bid fraction (%)", min_value=0, max_value=40, value=30, step=5) / 100
    seed = st.number_input("Random seed", min_value=0, max_value=100, value=42)
    num_sweeps = st.slider("SA sweeps", min_value=500, max_value=5000, value=1000, step=500)
    num_reads = st.slider("SA reads", min_value=50, max_value=200, value=100, step=50)
    run = st.button("Run Auction Clearing", use_container_width=True)

if run:
    with st.spinner("Reconstructing bids..."):
        bids_raw = logic.reconstruct_bids_multisession(
            n_bids,
            package_fraction=package_fraction,
            seed=int(seed)
        )
        bids_norm = logic.normalize(bids_raw)
        Q, qparams = logic.build_qubo(bids_norm)
        Q = Q / np.abs(Q).max()

    col_sa, col_ip = st.columns(2)

    with col_sa:
        with st.spinner("Running SA solver..."):
            sa = solvers.solve_sa(Q, bids_norm, bids_raw, num_reads=num_reads, num_sweeps=num_sweeps)
            if not sa['feasible']:
                sa['x']          = solvers.repair_solution(sa['x'], bids_norm, bids_raw)
                sa['welfare']    = bids_raw[sa['x'] == 1]['value'].sum()
                sa['n_accepted'] = int(sa['x'].sum())
                sa['feasible']   = True

    with col_ip:
        with st.spinner("Running IP solver..."):
            ip = solvers.solve_ip(bids_raw)

    gap = 100 * (1 - sa['welfare'] / ip['welfare'])

    st.markdown("---")
    st.subheader("Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("IP Welfare (€)", f"{ip['welfare']:,.0f}")
    m2.metric("SA Welfare (€)", f"{sa['welfare']:,.0f}", delta=f"-{gap:.2f}% vs IP", delta_color="inverse")
    m3.metric("SA Bids Accepted", sa['n_accepted'])
    m4.metric("SA Package Bonus (€)", f"{sa['pkg_bonus']:,.0f}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### IP Solution (Optimal)")
        ip_accepted = bids_raw[ip['x_val'] == 1][['type', 'quantity', 'value', 'price']].copy()
        ip_accepted['value'] = ip_accepted['value'].map("€{:,.0f}".format)
        ip_accepted['price'] = ip_accepted['price'].map("€{:,.2f}".format)
        ip_accepted['quantity'] = ip_accepted['quantity'].map("{:,.1f}".format)
        ip_summary = bids_raw[ip['x_val'] == 1].groupby('type').agg(
            n_accepted=('value', 'count'),
            total_value=('value', 'sum'),
            total_quantity=('quantity', 'sum'),
        ).reset_index()
        ip_summary['total_value'] = ip_summary['total_value'].map("€{:,.0f}".format)
        ip_summary['total_quantity'] = ip_summary['total_quantity'].map("{:,.1f}".format)
        ip_summary.columns = ['Type', 'Accepted', 'Total Welfare', 'Total Volume']
        st.dataframe(ip_summary, use_container_width=True, hide_index=True)
        st.caption(f"Total bids accepted: {ip['n_accepted']} | Feasible: {'✅' if ip['feasible'] else '❌'}")

    with col_right:
        st.markdown("#### SA Solution (QUBO)")
        sa_summary = bids_raw[sa['x'] == 1].groupby('type').agg(
            n_accepted=('value', 'count'),
            total_value=('value', 'sum'),
            total_quantity=('quantity', 'sum'),
        ).reset_index()
        sa_summary['total_value'] = sa_summary['total_value'].map("€{:,.0f}".format)
        sa_summary['total_quantity'] = sa_summary['total_quantity'].map("{:,.1f}".format)
        sa_summary.columns = ['Type', 'Accepted', 'Total Welfare', 'Total Volume']
        st.dataframe(sa_summary, use_container_width=True, hide_index=True)
        st.caption(f"Total bids accepted: {sa['n_accepted']} | Feasible: {'✅' if sa['feasible'] else '❌'} | Gap vs IP: {gap:.2f}%")

    st.markdown("---")
    st.subheader("Per-Type Volume Utilisation")
    util_rows = []
    for tau in bids_raw['type'].unique():
        supply = bids_raw[bids_raw['type'] == tau]['supply'].iloc[0]
        ip_vol = bids_raw[(bids_raw['type'] == tau) & (ip['x_val'] == 1)]['quantity'].sum()
        sa_vol = bids_raw[(bids_raw['type'] == tau) & (sa['x'] == 1)]['quantity'].sum()
        util_rows.append({
            'Type': tau,
            'Supply': f"{supply:,.0f}",
            'IP Volume': f"{ip_vol:,.0f}",
            'IP Util %': f"{100*ip_vol/supply:.1f}%",
            'SA Volume': f"{sa_vol:,.0f}",
            'SA Util %': f"{100*sa_vol/supply:.1f}%",
        })
    st.dataframe(pd.DataFrame(util_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("QUBO Parameters")
    p1, p2, p3 = st.columns(3)
    p1.metric("Total Bids", qparams['n_bids'])
    p2.metric("Package Pairs", qparams['n_packages'])
    p3.metric("Q Matrix Size", f"{qparams['n_bids']}×{qparams['n_bids']}")

else:
    st.info("Configure auction parameters in the sidebar and click **Run Auction Clearing** to begin.")