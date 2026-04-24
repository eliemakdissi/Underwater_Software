import threading
import time
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

import Params


def _build_layout():
    rows = []
    for key, value in Params.all().items():
        rows.append(html.Div([
            html.Label(key, style={'display': 'inline-block', 'width': '260px', 'fontFamily': 'monospace'}),
            dcc.Input(
                id={'type': 'param-input', 'key': key},
                type='number',
                value=value,
                step='any',
                style={'width': '180px'},
            ),
            html.Span(f' (default {Params.DEFAULTS[key]})',
                      style={'color': '#888', 'marginLeft': '10px', 'fontFamily': 'monospace'}),
        ], style={'margin': '6px 0'}))

    return html.Div([
        html.H2("SLAM - Live Hyperparameters"),
        html.Div(rows),
        html.Button('Apply', id='apply-btn', n_clicks=0,
                    style={'marginTop': '12px', 'padding': '8px 20px'}),
        html.Div(id='status', style={'marginTop': '10px', 'fontFamily': 'monospace'}),
    ], style={'maxWidth': '700px', 'margin': '20px auto', 'fontFamily': 'sans-serif'})


def start_param_server(host='0.0.0.0', port=8051):
    app = Dash(__name__)
    keys = list(Params.all().keys())
    app.layout = _build_layout()

    @app.callback(
        Output('status', 'children'),
        Input('apply-btn', 'n_clicks'),
        [State({'type': 'param-input', 'key': k}, 'value') for k in keys],
        prevent_initial_call=True,
    )
    def apply(n_clicks, *values):
        changed = []
        errors = []
        for k, v in zip(keys, values):
            if v is None:
                continue
            before = Params.get(k)
            try:
                Params.set(k, v)
            except (KeyError, ValueError) as e:
                errors.append(f"{k}: {e}")
                continue
            after = Params.get(k)
            if before != after:
                changed.append(f"{k}: {before} -> {after}")

        ts = time.strftime('%H:%M:%S')
        parts = [f"Applied at {ts}."]
        if changed:
            parts.append("Changed: " + ", ".join(changed))
        else:
            parts.append("No changes.")
        if errors:
            parts.append("Errors: " + "; ".join(errors))
        return " | ".join(parts)

    thread = threading.Thread(
        target=app.run,
        kwargs={'host': host, 'port': port, 'debug': False},
        daemon=True,
    )
    thread.start()
    print(f"[Params] Param server started at http://{host}:{port}")
