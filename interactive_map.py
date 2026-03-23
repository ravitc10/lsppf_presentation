import json
from pathlib import Path
import os

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State


# ----------------------------
# Files
# ----------------------------
COORDS_FILE = Path("final_1_tsne.json")

if not COORDS_FILE.exists():
    raise FileNotFoundError(f"Missing {COORDS_FILE.resolve()}")


# ----------------------------
# Load data
# ----------------------------
coords = json.loads(COORDS_FILE.read_text(encoding="utf-8"))

rows = []
for p in coords:
    rows.append({
        "x": float(p["x"]),
        "y": float(p["y"]),
        "name": (p.get("name") or "").strip(),
        "comment": (p.get("comment") or "").strip(),
    })

df = pd.DataFrame(rows)


# ----------------------------
# Wrap text for hover
# ----------------------------
def wrap_text(text, width=60):
    words = text.split()
    lines, current = [], []

    for w in words:
        if sum(len(x) for x in current) + len(current) + len(w) > width:
            lines.append(" ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        lines.append(" ".join(current))

    return "<br>".join(lines)


df["wrapped_comment"] = df["comment"].apply(lambda x: wrap_text(x, 60))


# ----------------------------
# Color mapping (pastel)
# ----------------------------
pastel_colors = [
    "#AEC6CF", "#FFB7B2", "#B5EAD7", "#FFDAC1",
    "#C7CEEA", "#E2F0CB", "#FF9AA2", "#D5AAFF"
]

unique_names = sorted(df["name"].unique())

color_map = {
    name: pastel_colors[i % len(pastel_colors)]
    for i, name in enumerate(unique_names)
}


# ----------------------------
# Build figure
# ----------------------------
def build_figure(annotation=None):
    fig = go.Figure()

    for name in unique_names:
        subset = df[df["name"] == name]

        fig.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers",
                marker=dict(size=12, color=color_map[name]),
                customdata=subset[["name", "wrapped_comment"]].values,
                hovertemplate="<b>%{customdata[0]}</b><br><br>%{customdata[1]}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Discussion Comments — Interactive t-SNE Map",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="#f9fafb",
        paper_bgcolor="#f9fafb",
        hoverlabel=dict(bgcolor="white", font_size=13, align="left"),
    )

    if annotation:
        fig.update_layout(annotations=[annotation])

    return fig


# ----------------------------
# Custom key
# ----------------------------
legend_items = []
for name in unique_names:
    legend_items.append(
        html.Div(
            style={"display": "flex", "alignItems": "center", "marginBottom": "6px"},
            children=[
                html.Div(
                    style={
                        "width": "12px",
                        "height": "12px",
                        "backgroundColor": color_map[name],
                        "marginRight": "8px",
                        "borderRadius": "50%",
                    }
                ),
                html.Span(name),
            ],
        )
    )


# ----------------------------
# Dash app
# ----------------------------
app = Dash(__name__)
server = app.server


app.layout = html.Div(
    style={
        "display": "flex",
        "flexDirection": "column",
        "height": "100vh",
        "padding": "10px",
        "fontFamily": "system-ui",
        "backgroundColor": "#eef2f7",
    },
    children=[

        html.Div(
            style={"textAlign": "center", "marginBottom": "10px"},
            children=[
                html.H2("Discussion Map"),
                html.P("Click a point to show its comment"),
            ],
        ),

        # store only ONE annotation
        dcc.Store(id="annotation-store", data=None),

        html.Div(
            style={"display": "flex", "height": "92vh"},
            children=[

                html.Div(
                    style={"width": "200px", "padding": "10px"},
                    children=[
                        html.H4("Key"),
                        *legend_items
                    ],
                ),

                html.Div(
                    style={
                        "flex": "1",
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                    },
                    children=[
                        html.Div(
                            style={
                                "width": "95%",
                                "height": "95%",
                                "backgroundColor": "white",
                                "borderRadius": "16px",
                                "boxShadow": "0px 10px 30px rgba(0,0,0,0.15)",
                                "padding": "10px",
                            },
                            children=[
                                dcc.Graph(
                                    id="graph",
                                    figure=build_figure(),
                                    clear_on_unhover=True,
                                    style={"height": "100%", "width": "100%"},
                                )
                            ],
                        )
                    ],
                ),
            ],
        ),
    ],
)


# ----------------------------
# Callback: single annotation (click anywhere clears)
# ----------------------------
@app.callback(
    Output("graph", "figure"),
    Output("annotation-store", "data"),
    Input("graph", "clickData"),
)
def update_annotation(clickData):
    if clickData is None:
        return build_figure(), None

    point = clickData["points"][0]

    annotation = dict(
        x=point["x"],
        y=point["y"],
        text=f"<b>{point['customdata'][0]}</b><br>{point['customdata'][1]}",
        showarrow=True,
        arrowhead=2,
        ax=20,
        ay=-20,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12),
    )

    return build_figure(annotation), annotation


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
