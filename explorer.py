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
import functools
import dash_bootstrap_components as dbc
import re

# ── 1) Paths & dropdown options ────────────────────────────────────────────────
CSV_DIR = Path(os.path.join(".", "fingerprints"))
print(os.getcwd())


ELEMENTS = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
    'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
    'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
}

df = pd.read_csv("fingerprints/unencoded/unencoded_window1eV_width1_embedding.csv")

# Remove ".png" and extract elements
filenames = df["filename"].str.replace(".png", "", regex=False)

def extract_elements(formula):
    parts = re.findall(r'[A-Z][a-z]?', formula)
    return [el for el in parts if el in ELEMENTS]

df['elements'] = filenames.apply(extract_elements)

# Flatten → unique → sort
element_list = sorted(set(df['elements'].explode()))

feature_values = {
    "Resnet"        : [0, 18, 50],
    "Latent Space"  : [1, 2, 3],
    "Training Steps": [30, 60, 90, 120],
    "Split"         : [0, 0.1, 0.5],
    "Window Size"   : [1, 2, 3],
    "Line Width"    : [1],
}
feature_defaults = {
    "Resnet": 50,
    "Latent Space": 3,
    "Training Steps": 90,
    "Split": 0.1,
    "Window Size": 1,
    "Line Width": 1,
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
app = Dash(__name__, title="Band Exploration", suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    html.H2("UMAP + Clustering Explorer (from CSV)", style={"textAlign":"center"}),

    html.Div([
        html.Div([
            html.Label(name, style={"fontSize":"0.8rem","marginBottom":"2px"}),
            dcc.Dropdown(
                id=f"feat-{i}", options=feature_options[name],
                value=feature_defaults[name], clearable=False,
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
                style={"width":"200px"}
            ),
            html.Button("Advanced Options", id="toggle-adv-btn", n_clicks=0),

            html.Div([
                html.Label("Filter by Element:", style={"fontSize": "0.8rem"}),
                dbc.DropdownMenu(
                    label="Select Elements",
                    children=[
                        dbc.Checklist(
                            id="element-checklist",
                            options=[{"label": el, "value": el} for el in element_list],
                            value=[],
                            inline=False,
                            style={
                                "columnCount": 3,  # ⬅️ This makes it 3 columns
                                "padding": "10px"
                            }
                        )
                    ],
                    toggle_style={"width": "200px"},
                    className="ml-2"
                )
            ], style={"display": "flex", "flexDirection": "column", "marginLeft": "10px"})

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
    ]),
    html.Button("Open Welcome Message", id="open-welcome-btn", n_clicks=0),
    html.Button(id="close-on-outside-click", style={"display": "none"}),

    html.Div(
    id="welcome-container",
    children=[
        html.Div([
            html.Button("×", id="close-welcome-btn",
                        style={"float": "right", "fontSize": "20px", "border": "none", "background": "none"}),
            dcc.Markdown("""
### Welcome to bands.heuslerene.com!

This tool enables interactive exploration of the thousands of band structures produced in our high-throughput study of Monolayer Heusler Alloys.

At its core, the tool uses a Residual Neural Network (ResNet) to autoencode each band structure image into a compact fingerprint of size **49**, **98**, or **147**, depending on the **Latent Space** dimension. These fingerprints are then projected into two dimensions using **UMAP**, allowing you to visualize and cluster the data. Hover over any point to see the corresponding band structure image.

---

#### 🔧 Interface Overview
- **Autoencoder Model**  
- **Clustering Model**  
- **Advanced Options**

---

#### ✅ Recommended Settings
To reproduce the results from our paper:

**Autoencoder Parameters**
- Resnet: **50**  
- Latent Space: **3**  
- Training Steps: **90**  
- Split: **0.1**  
- Window Size: **1 eV** (±1 eV around the Fermi energy)  
- Line Width: **1**

**Clustering Parameters**
- Algorithm: **HDBSCAN**  
- Cluster Min: **3**  
- Sample Min: **2**  
- Selection Method: `'leaf'`

---

#### ⚙️ Advanced Options
- Apply a second **DBSCAN** clustering to the UMAP embedding (after excluding HDBSCAN noise points)  
- Recluster **noise points** using the HDBSCAN parameters  
- **Hide** noise points, non-noise points, or both  

---

#### ℹ️ Additional Notes
- `'Unencoded'` ResNet means using the full 50,176-dimensional image representation as features  
- For `'Unencoded'`, clustering is done only on the UMAP projection  
- Hover images are not the exact model inputs but represent the same band structures  
- Use the **Open Welcome Message** button to view this message again anytime

            """),
        ], style={
        "maxHeight": "400px",
        "overflowY": "auto",
        "paddingRight": "10px"
    }),
    ],     style={
        "display": "flex",  # <- Show on startup
        "position": "fixed",
        "top": "50%",
        "left": "50%",
        "transform": "translate(-50%, -50%)",
        "zIndex": "1000",
        "background": "white",
        "padding": "20px",
        "borderRadius": "10px",
        "boxShadow": "0 0 10px rgba(0, 0, 0, 0.3)",
        "width": "90%",
        "maxWidth": "500px",
        "maxHeight": "80%",
        "overflowY": "auto"
    }
)

], style={"width":"80%","margin":"0 auto"})

# ── 3) Callbacks ─────────────────────────────────────────────
@app.callback(
    Output("welcome-container", "style"),
    Input("open-welcome-btn", "n_clicks"),
    Input("welcome-container", "n_clicks"),
    Input("close-welcome-btn", "n_clicks"),
    State("welcome-container", "style"),
    prevent_initial_call=True
)
def toggle_welcome(open_clicks, container_clicks, close_clicks, style):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    new_style = style.copy()
    if triggered_id == "open-welcome-btn":
        new_style["display"] = "flex"
    else:
        new_style["display"] = "none"
    return new_style



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
            type="number", value=2, min=1, step=1,
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
    img_dir = Path("images") / f"bands_10width_{window}eV"
    imgs = [str(img_dir / fname) for fname in filenames]
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
    Input("recluster-min-samples", "value"),
    Input("element-checklist", "value")  # ← NEW
)
def update_plot(store, algo, pvals, hide_noise, hide_nonnoise, double_cluster,
                resnet, recluster_active, recluster_mcs, recluster_ms, selected_elements):
    if not store:
        raise dash.exceptions.PreventUpdate
    if double_cluster is None:
        double_cluster = "off"

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

    # Already loaded earlier:
    filenames = [Path(p).stem for p in imgs]  # Remove .png
    element_tags = [re.findall(r"[A-Z][a-z]?", name) for name in filenames]

    # Get a boolean mask: does each point contain a selected element?
    highlight_mask = np.array([
        all(e in tags for e in selected_elements) for tags in element_tags
    ]) if selected_elements else np.zeros(len(imgs), dtype=bool)


    highlight_mask = highlight_mask & keep  # Only show what's kept

    fig.add_trace(go.Scatter(
        x=proj[highlight_mask][:, 0],
        y=proj[highlight_mask][:, 1],
        mode="markers",
        marker=dict(
            size=9,
            color='rgba(0,0,0,0)',
            line=dict(color="gold", width=2)
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



@functools.lru_cache(maxsize=512)
def load_image_b64(img_path):
    try:
        with open(img_path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

@app.callback(
    Output("hover-image", "src"),
    [Input("umap-plot", "hoverData"), Input("umap-plot", "clickData")],
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

    img_path = data["points"][0]["customdata"][0]
    return load_image_b64(img_path)


# ── 4) Run ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug = False)
