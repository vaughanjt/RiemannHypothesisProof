"""
SESSION 40 — INTERACTIVE HODGE STAR DEFORMATION EXPLORER

Explores the space of possible arithmetic Hodge stars by deforming the
Hilbert transform J (which commutes with M — wrong direction) toward
prime-injected perturbations that break Fourier symmetry.

Deformation:  J(t) = J_hilbert + t · Σ c_p Δ_p   (antisymmetric)
where Δ_p[n,m] = log(p)/√p · sin(2π(n−m)log(p)/L) is Fourier-anti-symmetric.

Key insight: J_hilbert is Fourier-symmetric (commutes with n↔−n),
so [J_hilbert, M] ≈ 0. The perturbation Δ_p is Fourier-ODD, so
adding it breaks Fourier symmetry and may produce [J(t), M] ≠ 0.

Usage:
    python session40_hodge_explorer.py [--lambda-sq 200] [--port 8050]
"""

import numpy as np
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connes_crossterm import build_all

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, no_update, callback_context
from dash.dependencies import Input, Output, State


# ═══════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════

BG      = '#0f172a'
CARD    = '#1e293b'
TEXT    = '#e2e8f0'
DIM     = '#94a3b8'
BLUE    = '#60a5fa'
ORANGE  = '#fb923c'
GREEN   = '#4ade80'
RED     = '#f87171'
PURPLE  = '#a78bfa'
GRID    = '#334155'

PLOT_KW = dict(
    paper_bgcolor=CARD, plot_bgcolor=BG,
    font=dict(color=TEXT, size=11),
    margin=dict(l=50, r=20, t=40, b=40),
)
AXIS_KW = dict(gridcolor=GRID)


# ═══════════════════════════════════════════════════════════════
# COMPUTATION ENGINE
# ═══════════════════════════════════════════════════════════════

class HodgeEngine:
    """Precomputes matrices and evaluates deformed Hodge stars."""

    def __init__(self, lam_sq=200):
        self.rebuild(lam_sq)

    def rebuild(self, lam_sq):
        self.lam_sq = int(lam_sq)
        L_f = np.log(lam_sq)
        N = max(15, round(6 * L_f))
        self.N, self.dim = N, 2 * N + 1
        ns = np.arange(-N, N + 1, dtype=float)

        W02, M, QW = build_all(lam_sq, N, n_quad=10000)
        self.W02, self.M_raw, self.QW = W02, M, QW

        # Null(W02)
        ew, ev = np.linalg.eigh(W02)
        thresh = np.max(np.abs(ew)) * 1e-10
        P_null = ev[:, np.abs(ew) <= thresh]
        self.d_null = P_null.shape[1]
        M_null = P_null.T @ M @ P_null

        # Hilbert J in full space (vectorized)
        dim = self.dim
        J_full = np.zeros((dim, dim))
        idx = np.arange(dim)
        j_idx = dim - 1 - idx
        J_full[idx[ns > 0], j_idx[ns > 0]] = -1
        J_full[idx[ns < 0], j_idx[ns < 0]] = 1
        J_full = (J_full - J_full.T) / 2

        # Project J → null(W02), extract good subspace (J²≈−Id)
        J_null = P_null.T @ J_full @ P_null
        J_sq = J_null @ J_null
        U, S, _ = np.linalg.svd(J_sq + np.eye(self.d_null))
        good = U[:, S < 0.1]
        self.d_good = good.shape[1]

        self.J0 = good.T @ J_null @ good
        self.M0 = good.T @ M_null @ good

        # Prime perturbations (Fourier-anti-symmetric)
        n_grid = ns.reshape(-1, 1) - ns.reshape(1, -1)  # (n−m) matrix
        limit = min(int(lam_sq), 10000)
        is_p = np.ones(limit + 1, dtype=bool); is_p[:2] = False
        for i in range(2, int(limit**0.5) + 2):
            if i <= limit and is_p[i]:
                is_p[i*i::i] = False
        self.primes = [p for p in range(2, limit + 1) if is_p[p]]

        self.deltas = {}
        for p in self.primes:
            logp = np.log(p)
            w = logp * p**(-0.5)
            D = w * np.sin(2 * np.pi * n_grid * logp / L_f)
            D_good = good.T @ (P_null.T @ D @ P_null) @ good
            self.deltas[p] = (D_good - D_good.T) / 2

    def evaluate(self, t, selected_primes):
        """Compute deformed star J(t) and all metrics."""
        d, M = self.d_good, self.M0

        # Build perturbation
        pert = np.zeros((d, d))
        for p in selected_primes:
            if p in self.deltas:
                pert += self.deltas[p]
        pn = np.linalg.norm(pert, 'fro')
        jn = np.linalg.norm(self.J0, 'fro')
        if pn > 1e-10:
            pert *= jn / pn

        J = (self.J0 + t * pert)
        J = (J - J.T) / 2  # enforce antisymmetry

        # J² quality
        j2_err = np.linalg.norm(J @ J + np.eye(d), 'fro') / d

        # J eigenvalues (should be near ±i)
        evals_J = np.linalg.eigvals(J)

        # Commutator [J,M] and anti-commutator {J,M}
        JM = J @ M
        comm = JM - M @ J
        anti = JM + M @ J
        cn = np.linalg.norm(comm, 'fro')
        an = np.linalg.norm(anti, 'fro')
        ratio = cn / (cn + an) if (cn + an) > 1e-10 else 0.0

        # Hodge-Riemann form h = sym(J·M)
        h = (JM + JM.T) / 2
        evals_h = np.linalg.eigvalsh(h)
        n_pos = int(np.sum(evals_h > 1e-6))
        n_neg = int(np.sum(evals_h < -1e-6))

        # Half-space decomposition: H^{1,0} (+i) and H^{0,1} (−i)
        evals_Jc, evecs_Jc = np.linalg.eig(J)
        max_imag = np.max(np.abs(evals_Jc.imag))
        thr = 0.3 * max_imag if max_imag > 0.1 else 0.3
        M_plus, M_minus = np.array([]), np.array([])

        for mask, target in [(evals_Jc.imag > thr, 'plus'),
                             (evals_Jc.imag < -thr, 'minus')]:
            if np.sum(mask) >= 2:
                V = evecs_Jc[:, mask]
                Mh = np.real(V.conj().T @ M @ V)
                Mh = (Mh + Mh.T) / 2
                ev = np.linalg.eigvalsh(Mh)
                if target == 'plus':
                    M_plus = ev
                else:
                    M_minus = ev

        return dict(
            j2_err=j2_err, evals_J=evals_J,
            cn=cn, an=an, ratio=ratio,
            evals_h=evals_h, n_pos=n_pos, n_neg=n_neg,
            definite=(n_pos == 0 or n_neg == 0),
            M_plus=M_plus, M_minus=M_minus,
        )

    def sweep(self, selected_primes, n_pts=40):
        """Compute metrics along deformation path t=0…1."""
        ts = np.linspace(0, 1, n_pts)
        cn, an, ratio, j2 = [], [], [], []
        for t in ts:
            r = self.evaluate(t, selected_primes)
            cn.append(r['cn']); an.append(r['an'])
            ratio.append(r['ratio']); j2.append(r['j2_err'])
        return dict(t=ts, cn=np.array(cn), an=np.array(an),
                    ratio=np.array(ratio), j2=np.array(j2))


# ═══════════════════════════════════════════════════════════════
# FIGURE BUILDERS
# ═══════════════════════════════════════════════════════════════

def fig_j_eigenvalues(r):
    """J eigenvalues in complex plane — should cluster at ±i."""
    ev = r['evals_J']
    fig = go.Figure()
    # Unit circle
    th = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode='lines',
                             line=dict(color=GRID, dash='dot', width=1),
                             showlegend=False, hoverinfo='skip'))
    # ±i targets
    fig.add_trace(go.Scatter(x=[0, 0], y=[1, -1], mode='markers',
                             marker=dict(symbol='x', size=12, color=DIM),
                             showlegend=False, hoverinfo='skip'))
    # Eigenvalues
    dist = np.minimum(np.abs(ev - 1j), np.abs(ev + 1j))
    colors = [GREEN if d < 0.2 else ORANGE if d < 0.5 else RED for d in dist]
    fig.add_trace(go.Scatter(
        x=ev.real, y=ev.imag, mode='markers',
        marker=dict(size=8, color=colors, line=dict(width=1, color='white')),
        hovertemplate='%{x:.3f} + %{y:.3f}i<extra></extra>'))

    fig.update_layout(**PLOT_KW, showlegend=False, height=340,
                      title=dict(text='J Eigenvalues', font=dict(size=13)),
                      xaxis=dict(title='Re', range=[-2, 2],
                                 scaleanchor='y', **AXIS_KW),
                      yaxis=dict(title='Im', range=[-2, 2], **AXIS_KW))
    return fig


def fig_hodge_riemann(r):
    """Hodge-Riemann form eigenvalues — green positive, red negative."""
    ev = r['evals_h']
    colors = [GREEN if v > 1e-6 else RED if v < -1e-6 else DIM for v in ev]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(ev))), y=ev,
                         marker=dict(color=colors, line=dict(width=0)),
                         hovertemplate='λ_%{x} = %{y:.4f}<extra></extra>'))

    label = "DEFINITE" if r['definite'] else f"INDEFINITE ({r['n_pos']}+ / {r['n_neg']}−)"
    lc = GREEN if r['definite'] else RED
    fig.add_annotation(x=0.5, y=1.0, xref='paper', yref='paper',
                       text=f"<b>{label}</b>", font=dict(size=15, color=lc),
                       showarrow=False, bgcolor='rgba(0,0,0,0.6)', borderpad=4)

    fig.update_layout(**PLOT_KW, height=340,
                      title=dict(text='h = sym(J·M) Eigenvalues', font=dict(size=13)),
                      xaxis=dict(title='Index', showgrid=False, **AXIS_KW),
                      yaxis=dict(title='Eigenvalue', zeroline=True,
                                 zerolinecolor='white', zerolinewidth=1, **AXIS_KW))
    return fig


def fig_half_spaces(r):
    """M eigenvalues on H^{1,0} and H^{0,1} half-spaces."""
    fig = go.Figure()
    mp, mm = r['M_plus'], r['M_minus']

    if len(mp) > 0:
        fig.add_trace(go.Bar(
            x=[f'+i_{i}' for i in range(len(mp))], y=mp,
            name='H^{1,0} (+i)',
            marker=dict(color=BLUE, opacity=0.8)))
    if len(mm) > 0:
        fig.add_trace(go.Bar(
            x=[f'−i_{i}' for i in range(len(mm))], y=mm,
            name='H^{0,1} (−i)',
            marker=dict(color=ORANGE, opacity=0.8)))

    if len(mp) == 0 and len(mm) == 0:
        fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper',
                           text="Half-space decomposition unavailable",
                           font=dict(size=13, color=DIM), showarrow=False)

    fig.update_layout(**PLOT_KW, height=340, barmode='group',
                      title=dict(text='M on Half-Spaces', font=dict(size=13)),
                      legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
                      xaxis=dict(showgrid=False, **AXIS_KW),
                      yaxis=dict(title='M eigenvalue', zeroline=True,
                                 zerolinecolor='white', zerolinewidth=1, **AXIS_KW))
    return fig


def fig_sweep(sw, t_now):
    """Metric sweep t=0→1: norms and J² error."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.15, row_heights=[0.65, 0.35],
                        subplot_titles=('||[J,M]|| vs ||{J,M}||', 'J² Error'))
    t = sw['t']
    fig.add_trace(go.Scatter(x=t, y=sw['cn'], name='||[J,M]||',
                             line=dict(color=BLUE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=sw['an'], name='||{J,M}||',
                             line=dict(color=ORANGE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=sw['ratio']*100, name='Ratio%',
                             line=dict(color=GREEN, width=2, dash='dash'),
                             visible='legendonly'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=sw['j2'], name='J² err',
                             line=dict(color=RED, width=2)), row=2, col=1)

    # Vertical marker at current t
    for row in [1, 2]:
        fig.add_vline(x=t_now, line=dict(color='white', width=1, dash='dash'),
                      row=row, col=1)

    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=BG,
        font=dict(color=TEXT, size=11),
        margin=dict(l=50, r=20, t=30, b=40),
        height=340, showlegend=True,
        legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center',
                    font=dict(size=10)),
    )
    for i in range(1, 3):
        fig.update_xaxes(gridcolor=GRID, zerolinecolor='#475569', row=i, col=1)
        fig.update_yaxes(gridcolor=GRID, zerolinecolor='#475569', row=i, col=1)
    fig.update_xaxes(title_text='t', row=2, col=1)
    return fig


# ═══════════════════════════════════════════════════════════════
# DASH APP
# ═══════════════════════════════════════════════════════════════

engine = None
_sweep_cache = {'key': None, 'data': None}


def cached_sweep(primes):
    key = frozenset(primes)
    if _sweep_cache['key'] != key:
        _sweep_cache['data'] = engine.sweep(primes, n_pts=40)
        _sweep_cache['key'] = key
    return _sweep_cache['data']


def create_app():
    app = Dash(__name__)
    show_primes = engine.primes[:15]

    app.layout = html.Div(
        style={'backgroundColor': BG, 'minHeight': '100vh',
               'padding': '16px', 'fontFamily': 'monospace', 'color': TEXT},
        children=[
            # ── Header ──
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between',
                            'alignItems': 'center', 'marginBottom': '12px'}, children=[
                html.Div([
                    html.H2("HODGE STAR DEFORMATION EXPLORER",
                             style={'margin': '0', 'fontSize': '18px',
                                    'letterSpacing': '1px'}),
                    html.Div(
                        f"λ²={engine.lam_sq}  |  dim={engine.dim}  |  "
                        f"null={engine.d_null}  |  good={engine.d_good}  |  "
                        f"{len(engine.primes)} primes",
                        style={'color': DIM, 'fontSize': '12px', 'marginTop': '2px'}),
                ]),
                html.Div(
                    "Session 40",
                    style={'color': PURPLE, 'fontSize': '14px', 'fontWeight': 'bold'}),
            ]),

            # ── Deformation slider ──
            html.Div(style={'padding': '10px 16px', 'backgroundColor': CARD,
                            'borderRadius': '8px', 'marginBottom': '8px'}, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center',
                                'gap': '16px'}, children=[
                    html.Span("Deformation t :",
                              style={'color': DIM, 'whiteSpace': 'nowrap'}),
                    html.Div(style={'flex': '1'}, children=[
                        dcc.Slider(id='slider-t', min=0, max=1, step=0.01, value=0,
                                   marks={i/4: f'{i/4:.2f}' for i in range(5)},
                                   tooltip={'placement': 'bottom',
                                            'always_visible': True}),
                    ]),
                ]),
                html.Div(
                    "t=0: pure Hilbert (commutes with M)  →  "
                    "t=1: full prime injection (breaks Fourier symmetry)",
                    style={'color': DIM, 'fontSize': '11px', 'marginTop': '4px'}),
            ]),

            # ── Prime controls ──
            html.Div(style={'padding': '10px 16px', 'backgroundColor': CARD,
                            'borderRadius': '8px', 'marginBottom': '10px'}, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center',
                                'gap': '8px', 'flexWrap': 'wrap'}, children=[
                    html.Span("Primes:", style={'color': DIM, 'marginRight': '4px'}),
                    dcc.Checklist(
                        id='prime-list',
                        options=[{'label': html.Span(
                            f' {p}', style={'fontSize': '12px', 'marginRight': '6px'}
                        ), 'value': p} for p in show_primes],
                        value=show_primes,
                        inline=True,
                    ),
                    html.Button("All", id='btn-all', n_clicks=0,
                                style={'fontSize': '11px', 'padding': '2px 10px',
                                       'backgroundColor': GRID, 'color': TEXT,
                                       'border': f'1px solid {DIM}',
                                       'borderRadius': '4px', 'cursor': 'pointer'}),
                    html.Button("None", id='btn-none', n_clicks=0,
                                style={'fontSize': '11px', 'padding': '2px 10px',
                                       'backgroundColor': GRID, 'color': TEXT,
                                       'border': f'1px solid {DIM}',
                                       'borderRadius': '4px', 'cursor': 'pointer'}),
                    html.Span(id='prime-info', style={'color': DIM, 'fontSize': '11px'}),
                ]),
            ]),

            # ── Loading wrapper around plots ──
            dcc.Loading(type='circle', color=BLUE, children=[
                # 2×2 plot grid
                html.Div(style={'display': 'grid',
                                'gridTemplateColumns': '1fr 1fr',
                                'gap': '6px', 'marginBottom': '8px'}, children=[
                    dcc.Graph(id='plot-j',
                              config={'displayModeBar': True, 'toImageButtonOptions':
                                      {'format': 'png', 'height': 600, 'width': 800}}),
                    dcc.Graph(id='plot-hr',
                              config={'displayModeBar': True, 'toImageButtonOptions':
                                      {'format': 'png', 'height': 600, 'width': 800}}),
                    dcc.Graph(id='plot-half',
                              config={'displayModeBar': True, 'toImageButtonOptions':
                                      {'format': 'png', 'height': 600, 'width': 800}}),
                    dcc.Graph(id='plot-sweep',
                              config={'displayModeBar': True, 'toImageButtonOptions':
                                      {'format': 'png', 'height': 600, 'width': 800}}),
                ]),
            ]),

            # ── Metrics bar ──
            html.Div(id='metrics-bar', style={
                'display': 'flex', 'justifyContent': 'space-around',
                'padding': '12px 16px', 'backgroundColor': CARD,
                'borderRadius': '8px', 'fontSize': '13px',
            }),
        ]
    )

    # ── All/None buttons for primes ──
    @app.callback(
        Output('prime-list', 'value'),
        [Input('btn-all', 'n_clicks'), Input('btn-none', 'n_clicks')],
        [State('prime-list', 'options')],
        prevent_initial_call=True,
    )
    def toggle_primes(all_c, none_c, options):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        btn = ctx.triggered[0]['prop_id']
        if 'btn-all' in btn:
            return [o['value'] if isinstance(o, dict) else o for o in options]
        if 'btn-none' in btn:
            return []
        return no_update

    # ── Main update callback ──
    @app.callback(
        [Output('plot-j', 'figure'),
         Output('plot-hr', 'figure'),
         Output('plot-half', 'figure'),
         Output('plot-sweep', 'figure'),
         Output('metrics-bar', 'children'),
         Output('prime-info', 'children')],
        [Input('slider-t', 'value'),
         Input('prime-list', 'value')],
    )
    def update_all(t, primes_on):
        primes = [int(p) for p in (primes_on or [])]
        result = engine.evaluate(t, primes)
        sw = cached_sweep(primes)

        f1 = fig_j_eigenvalues(result)
        f2 = fig_hodge_riemann(result)
        f3 = fig_half_spaces(result)
        f4 = fig_sweep(sw, t)

        # Metrics bar
        def _m(label, val, color):
            return html.Span([
                html.Span(f"{label}: ", style={'color': DIM}),
                html.Span(val, style={'color': color, 'fontWeight': 'bold'}),
            ])

        j2c = GREEN if result['j2_err'] < 0.01 else ORANGE if result['j2_err'] < 0.1 else RED
        rc = GREEN if result['ratio'] > 0.3 else ORANGE if result['ratio'] > 0.05 else DIM
        dc = GREEN if result['definite'] else RED
        dt = "YES" if result['definite'] else "NO"

        metrics = [
            _m("J² err", f"{result['j2_err']:.4f}", j2c),
            _m("||[J,M]||", f"{result['cn']:.3f}", BLUE),
            _m("||{{J,M}}||", f"{result['an']:.3f}", ORANGE),
            _m("Comm ratio", f"{result['ratio']*100:.1f}%", rc),
            _m("H-R definite", dt, dc),
        ]

        info = f"({len(primes)} of {len(engine.primes)} primes)"
        return f1, f2, f3, f4, metrics, info

    return app


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hodge Star Deformation Explorer')
    parser.add_argument('--lambda-sq', type=int, default=200,
                        help='Lambda squared (default: 200)')
    parser.add_argument('--port', type=int, default=8050,
                        help='Server port (default: 8050)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not auto-open browser')
    args = parser.parse_args()

    print(f"\n  Building matrices for lam^2={args.lambda_sq}...", flush=True)
    t0 = time.time()
    engine = HodgeEngine(args.lambda_sq)
    dt = time.time() - t0
    print(f"  Ready in {dt:.1f}s", flush=True)
    print(f"  dim={engine.dim}  null={engine.d_null}  good={engine.d_good}  "
          f"primes={len(engine.primes)}", flush=True)

    # Quick sanity check: J² error at t=0
    r0 = engine.evaluate(0, [])
    print(f"  Baseline J^2 err = {r0['j2_err']:.2e}  "
          f"[J,M]={r0['cn']:.4f}  {{J,M}}={r0['an']:.4f}", flush=True)

    app = create_app()

    url = f'http://localhost:{args.port}'
    print(f"\n  Open: {url}\n", flush=True)
    if not args.no_browser:
        import webbrowser
        webbrowser.open(url)

    app.run(debug=False, port=args.port)
