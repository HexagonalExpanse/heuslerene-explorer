import os, base64
from pathlib import Path

import dash
from dash import Dash, html, dcc, Input, Output, State, ALL
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import hdbscan
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from plotly.colors import qualitative
import itertools


# ── 1) Paths & dropdown options ────────────────────────────────────────────────
CSV_DIR = Path(os.path.join(".", "fingerprints"))
print(os.getcwd())

feature_values = {
    "Resnet"        : [0, 18, 50],
    "Latent Space"  : [1, 2, 3],
    "Training Steps": [30, 60, 90, 120],
    "Split"         : [0, 0.1, 0.5],
    "Window Size"   : [1, 2, 3],
    "Line Width"    : [1],
}
feature_options = {
    name: [{"label": str(v), "value": v} for v in vals]
    for name, vals in feature_values.items()
}
feature_options["Resnet"] = [{"label": "Unencoded", "value": 0}] + [
    {"label": str(v), "value": v} for v in feature_values["Resnet"] if v != 0
]


clust_opts = [
    {"label": "K‑Means", "value": "kmeans"},
    {"label": "GMM",     "value": "gmm"},
    {"label": "HDBSCAN", "value": "hdbscan"},
]

recluster_options = [
    {'label': 'Off', 'value' :'off'},
    {'label': 'DBSCAN', 'value': 'DBSCAN'},
]

# ── 2) Dash app layout ─────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)
app.layout = html.Div([
    html.H2("UMAP + Clustering Explorer (from CSV)", style={"textAlign":"center"}),

    html.Div([
        html.Div([
            html.Label(name, style={"fontSize":"0.8rem","marginBottom":"2px"}),
            dcc.Dropdown(
                id=f"feat-{i}", options=feature_options[name],
                value=feature_values[name][0], clearable=False,
                style={"width":"180px"}
            )
        ], style={"margin":"4px","display":"flex","flexDirection":"column"})
        for i, name in enumerate(feature_values)
    ], style={"display":"flex","flexWrap":"wrap"}),

    html.Div([
        html.Div([
            dcc.Dropdown(
                id="algo-dd", options=clust_opts,
                value="hdbscan", clearable=False,
                style={"width":"200px","marginRight":"8px"}
            ),
            html.Button("Advanced Options", id="toggle-adv-btn", n_clicks=0),
        ], style={"display":"flex", "alignItems":"center", "gap":"10px"}),

        html.Div(id="param-controls"),

        html.Div(id="adv-panel", style={"display": "none", "marginTop": "10px"})
    ], style={"display":"flex", "flexDirection":"column"}),


    html.Div([
        dcc.Graph(
            id="umap-plot", config={"scrollZoom":True},
            style={"height":"650px","width":"70%","display":"inline-block"}
        ),
        dcc.Store(id="latent-store"),
        dcc.Store(id="adv-visible", data=False),
        dcc.Store(id="recluster-active", data=False),
        dcc.Store(id="hide-noise-active", data=False),
        dcc.Store(id="hide-nonnoise-active", data=False),
        html.Img(
            id="hover-image",
            style={
                "width":"300px","height":"auto",
                "display":"inline-block","verticalAlign":"top",
                "marginLeft":"16px"
            }
        ),
        html.Div(id="recluster-popup", style={"display": "none"})
    ])
], style={"width":"80%","margin":"0 auto"})

# ── 3) Callbacks ─────────────────────────────────────────────
@app.callback(
    Output("adv-panel", "style"),
    Input("toggle-adv-btn", "n_clicks"),
    prevent_initial_call=True
)
def toggle_advanced(n):
    show = n % 2 == 1
    return {"display": "flex", "flexDirection": "row", "alignItems": "center", "gap": "20px", "marginTop": "10px"} if show else {"display": "none"}


@app.callback(
    Output("adv-panel", "children"),
    Input("toggle-adv-btn", "n_clicks")
)
def advanced_controls(_):
    return [
        html.Div([
            html.Label("Double Cluster"),
            dcc.Dropdown(
                id="double-cluster-dd",
                options=[
                    {'label': 'Off', 'value': 'off'},
                    {'label': 'DBSCAN', 'value': 'DBSCAN'}
                ],
                value='off',
                clearable=False,
                style={"width": "150px"}
            )
        ], style={"display": "flex", "flexDirection": "column"}),

        html.Button("Recluster Noise", id="recluster-noise-btn", n_clicks=0),
        html.Button("Hide Noise", id="hide-noise-btn", n_clicks=0),
        html.Button("Hide Non-Noise", id="hide-nonnoise-btn", n_clicks=0),

        html.Div([
            html.Div([
                html.Label("Recluster Min Cluster Size:"),
                dcc.Input(id="recluster-min-cluster-size", type="number", value=3, min=1)
            ], style={"display": "flex", "flexDirection": "column"}),

            html.Div([
                html.Label("Recluster Min Samples:"),
                dcc.Input(id="recluster-min-samples", type="number", value=5, min=1)
            ], style={"display": "flex", "flexDirection": "column", "marginLeft": "10px"})
        ], style={"display": "flex", "flexDirection": "row", "alignItems": "center", "marginLeft": "20px"}),

    ]


@app.callback(
    Output("param-controls", "children"),
    Input("algo-dd", "value"),
    Input("feat-4", "value")  # Window Size
)
def make_param_inputs(algo, window_size):
    if algo in ["kmeans", "gmm"]:
        return html.Span([
            "num clusters:",
            dcc.Input(
                id={"type": "param", "name": "num_clusters"},
                type="number", value=5, step=1,
                style={"width": "70px"}
            )
        ])

    # Control whether the dropdown is disabled or not
    allow_selection = window_size == 1

    return html.Div([
        html.Label("Cluster Min:"),
        dcc.Input(
            id={"type": "param", "name": "cluster_min"},
            type="number", value=3, min=1, step=1,
            style={"width": "100px"}
        ),

        html.Label("Sample Min:", style={'marginLeft': '20px'}),
        dcc.Input(
            id={"type": "param", "name": "sample_min"},
            type="number", value=3, min=1, step=1,
            style={"width": "100px"}
        ),

        html.Label("Selection Method:", style={'marginLeft': '20px'}),
        dcc.Dropdown(
            id={"type": "param", "name": "selection_method"},
            options=[
                {"label": "leaf", "value": "leaf"},
                {"label": "eom", "value": "eom"}
            ],
            value="eom" if allow_selection else "leaf",
            clearable=False,
            disabled=False,
            style={"width": "150px"}
        )
    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'})




@app.callback(
    Output("latent-store", "data"),
    [Input(f"feat-{i}", "value") for i in range(len(feature_values))])
def load_from_csv(resnet, latent_dim, steps, split, window, width):
    if resnet == 0:
        fpath    = CSV_DIR / "unencoded" / f"unencoded_window{window}eV_width{width}_embedding.csv"
        emb_path = fpath
    else:
        base = f"resnet{resnet}_latent{latent_dim}_window{window}eV_steps{steps}_split{(split*10)}_width{width}"
        fpath    = CSV_DIR / f"{base}_.csv"
        emb_path = CSV_DIR / f"{base}_embedding.csv"

    # 2) load fingerprints
    df       = pd.read_csv(fpath)
    filenames= df["filename"].tolist()        # must have this column
    imgs     = df["image_b64"].tolist()
    X        = df.filter(like="z").to_numpy()

    # 3) load embedding as a DataFrame
    emb_df   = pd.read_csv(emb_path)          # columns: filename, z0, zq
    emb_df   = emb_df.set_index("filename")   # index by filename
    # 4) re‑order to match df rows
    try:
        coords = emb_df.loc[filenames, ["z0","z1"]].to_numpy()
    except KeyError as e:
        missing = set(filenames) - set(emb_df.index)
        raise ValueError(f"Embedding missing rows for: {missing}")
    return {
        "latents": X.tolist(),
        "imgs":     imgs,
        "emb":      coords.tolist()
    }


@app.callback(
    Output("hide-noise-active", "data"),
    Input("hide-noise-btn", "n_clicks"),
    prevent_initial_call=True
)
def toggle_hide_noise(n):
    return n % 2 == 1

@app.callback(
    Output("hide-nonnoise-active", "data"),
    Input("hide-nonnoise-btn", "n_clicks"),
    prevent_initial_call=True
)
def toggle_hide_nonnoise(n):
    return n % 2 == 1

@app.callback(
    Output("recluster-active", "data"),
    Input("recluster-noise-btn", "n_clicks"),
    prevent_initial_call=True
)
def toggle_recluster_noise(n):
    return n % 2 == 1

@app.callback(
    Output("recluster-noise-btn", "style"),
    Input("recluster-active", "data")
)
def update_recluster_noise_style(active):
    return {"backgroundColor": "lightgreen" if active else "lightgray"}


@app.callback(
    Output("hide-noise-btn", "style"),
    Input("hide-noise-active", "data")
)
def update_hide_noise_style(active):
    return {"backgroundColor": "lightgreen" if active else "lightgray"}

@app.callback(
    Output("hide-nonnoise-btn", "style"),
    Input("hide-nonnoise-active", "data")
)
def update_hide_nonnoise_style(active):
    return {"backgroundColor": "lightgreen" if active else "lightgray"}


@app.callback(
    Output("umap-plot", "figure"),
    Input("latent-store", "data"),
    Input("algo-dd", "value"),
    Input({"type": "param", "name": ALL}, "value"),
    Input("hide-noise-active", "data"),
    Input("hide-nonnoise-active", "data"),
    Input("double-cluster-dd", "value"),
    Input("feat-0", "value"),
    Input("recluster-active", "data"),
    Input("recluster-min-cluster-size", "value"),
    Input("recluster-min-samples", "value")
)
def update_plot(store, algo, pvals, hide_noise, hide_nonnoise, double_cluster, resnet, recluster_active,
                recluster_mcs, recluster_ms):
    if not store:
        raise dash.exceptions.PreventUpdate

    X = np.array(store["latents"])
    imgs = np.array(store["imgs"])
    proj = np.array(store["emb"])

    if resnet == 0:
        X = proj  # Use UMAP for clustering

    # Cluster
    if algo == "kmeans":
        k = int(pvals[0]) if pvals else 5
        labels = KMeans(n_clusters=k).fit_predict(X)
    elif algo == "gmm":
        k = int(pvals[0]) if pvals else 5
        labels = GaussianMixture(n_components=k).fit(X).predict(X)
    else:
        mcs, ms, csm = (int(pvals[0]), int(pvals[1]), pvals[2]) if len(pvals) >= 3 else (3, 3, 'leaf')
        labels = hdbscan.HDBSCAN(
            min_cluster_size=mcs, min_samples=ms, cluster_selection_method=csm, p=0.2
        ).fit_predict(X)

    labels = np.array(labels)
    was_noise = labels == -1
    max_label = labels.max()
    outline_mask = np.zeros_like(labels, dtype=bool)

    if recluster_active and np.any(was_noise):
        noise_proj = X[was_noise]
        sub_labels = hdbscan.HDBSCAN(
            min_cluster_size=recluster_mcs or 3,
            min_samples=recluster_ms or 3,
            cluster_selection_method='eom',
            p=0.2
        ).fit_predict(noise_proj)
    
        offset = max_label + 1
        reclustered_labels = np.where(sub_labels != -1, sub_labels + offset, -1)
        labels[was_noise] = reclustered_labels
        outline_mask[was_noise] = reclustered_labels != -1
        max_label = max(max_label, np.max(reclustered_labels))

    if hide_noise and hide_nonnoise:
        keep = np.zeros_like(labels, dtype=bool)
    elif hide_noise:
        keep = ~was_noise
    elif hide_nonnoise:
        keep = was_noise
    else:
        keep = np.ones_like(labels, dtype=bool)

    # ── Color Mapping ────────────────────────────────
    from plotly.colors import qualitative
    import itertools

    color_list = qualitative.Plotly + qualitative.D3 + qualitative.Set3
    color_cycle = itertools.cycle(color_list)
    unique_labels = sorted(set(labels[keep]))

    label_to_color = {-1: "black"}
    for lbl in unique_labels:
        if lbl != -1:
            label_to_color[lbl] = next(color_cycle)

    colors = [label_to_color[lbl] for lbl in labels[keep]]

    # ── Plot ─────────────────────────────────────
    fig = go.Figure()

    if double_cluster == "DBSCAN":
        db_labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(proj[~was_noise])
        db_mask = (db_labels != -1)
        fig.add_trace(go.Scatter(
            x=proj[~was_noise][db_mask][:, 0],
            y=proj[~was_noise][db_mask][:, 1],
            mode="markers",
            marker=dict(
                size=22,
                color=db_labels[db_mask],
                opacity=0.3,
                colorscale="Viridis",
                line=dict(width=0)
            ),
            hoverinfo="skip",
            showlegend=False
        ))

    # Main points
    # Add cluster labels to customdata for hover text
    hover_customdata = np.stack([imgs[keep], labels[keep]], axis=1)

    fig.add_trace(go.Scatter(
        x=proj[keep][:, 0],
        y=proj[keep][:, 1],
        mode="markers",
        marker=dict(
            size=8,
            color=colors  # manually assigned hex/rgb colors
        ),
        customdata=hover_customdata,
        hovertemplate="<b>Cluster</b>: %{customdata[1]}<extra></extra>",
        showlegend=False
    ))


    # Outline for reclustered noise
    outline_visible = keep & outline_mask
    fig.add_trace(go.Scatter(
        x=proj[outline_visible][:, 0],
        y=proj[outline_visible][:, 1],
        mode="markers",
        marker=dict(
            size=8,
            color='rgba(0,0,0,0)',
            line=dict(color="black", width=1)
        ),
        hoverinfo="skip",
        showlegend=False
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_visible=False,
        yaxis_visible=False
    )

    return fig



@app.callback(
    Output("hover-image", "src"),
    [Input("umap-plot", "hoverData"),
     Input("umap-plot","clickData")],
    prevent_initial_call=True
)
def show_hover_image(hover, click):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    triggered_input = ctx.triggered[0]["prop_id"].split(".")[0]
    data = hover if triggered_input == "umap-plot" and hover else click

    if not data or "points" not in data:
        return dash.no_update
    b64 = data["points"][0]["customdata"][0]
    return f"data:image/png;base64,{b64}"
# ── 4) Run ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
