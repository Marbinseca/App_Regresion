import dash
from dash import dcc, html, Input, Output, State, no_update, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.io as pio
import io
import base64
import statsmodels.api as sm

# Inicializa la aplicación Dash con un tema de Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Exponer el servidor Flask para el despliegue en producción (ej. Gunicorn en Render)
server = app.server

# --- Layout de la Aplicación ---
app.layout = dbc.Container([
    dcc.Store(id='stored-dataframe-json'),
    dcc.Store(id='store-summary-text'),
    dcc.Store(id='store-dataframe-predictions'),
    dcc.Download(id="download-summary-txt"),
    dcc.Download(id="download-data-csv"),
    dcc.Download(id="download-main-graph-png"),
    dcc.Download(id="download-residual-graph-png"),
    
    # Título
    dbc.Row(dbc.Col(html.H1("Análisis de Regresión Lineal", className="text-center text-primary my-4"))),

    # Sección 1: Carga de Archivos
    dbc.Card([
        dbc.CardHeader(html.H4("1. Carga tu conjunto de datos")),
        dbc.CardBody([
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Arrastra y suelta o ', html.A('selecciona un archivo (CSV o Excel)')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '5px', 
                    'textAlign': 'center', 'margin': '10px 0'
                },
            ),
            html.Div(id='output-upload-state', className="text-center mt-3")
        ])
    ], className="mb-4"),

    # Sección 2: Controles
    dbc.Card(id='controls-section', style={'display': 'none'}, children=[
        dbc.CardHeader(html.H4("2. Configura tu modelo de regresión")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Variables Independientes (X):", className="fw-bold"),
                    dcc.Dropdown(id='xaxis-column', placeholder="Selecciona una o más variables...", multi=True)
                ], width=6),
                dbc.Col([
                    html.Label("Variable Dependiente (Eje Y):", className="fw-bold"),
                    dcc.Dropdown(id='yaxis-column', placeholder="Selecciona la variable Y...")
                ], width=6)
            ]),
            dbc.Row(dbc.Col(
                dbc.Button('Generar Análisis', id='submit-button', n_clicks=0, color="primary", className="mt-3 w-80")
            ))
        ])
    ], className="mb-4"),

    # Sección 3, 4 y 5: Resultados
    html.Div(id='results-section', style={'display': 'none'}, children=[
        dbc.Row([
            # Columna Izquierda: Gráfica e Interpretación
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("3. Visualización del Modelo")),
                    dbc.CardBody(dcc.Loading(dcc.Graph(id='regression-graph')))
                ], className="mb-4"),
                dbc.Card([
                    dbc.CardHeader(html.H4("4. Análisis de Residuos")),
                    dbc.CardBody(dcc.Loading(dcc.Graph(id='residual-graph')))
                ], className="mb-4"),
                dbc.Card([
                    dbc.CardHeader(html.H4("6. Interpretación del Análisis")),
                    dbc.CardBody(dcc.Loading(id='model-interpretation'))
                ]),
                dbc.Card([
                    dbc.CardHeader(html.H4("7. Descargar Resultados")),
                    dbc.CardBody([
                        dbc.Button("Resumen (.txt)", id="btn-download-summary", color="info", outline=True, className="me-2 mb-2"),
                        dbc.Button("Datos Completos (.csv)", id="btn-download-data", color="info", outline=True, className="me-2 mb-2"),
                    ])
                ], className="mt-4")
            ], md=7),
            # Columna Derecha: Resumen Técnico
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("5. Resumen Técnico (Statsmodels)")),
                    dbc.CardBody(dcc.Loading(html.Pre(id='ols-summary', style={'fontSize': '0.8em'})))
                ])
            ], md=5)
        ])
    ])
], fluid=True)

# --- Callbacks ---

@app.callback(
    Output('results-section', 'style'),
    Input('submit-button', 'n_clicks'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def manage_results_visibility(n_clicks, contents):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'upload-data':
        # Si se sube un nuevo archivo, ocultar los resultados
        return {'display': 'none'}
    
    if triggered_id == 'submit-button' and n_clicks > 0:
        # Si se presiona el botón, mostrar los resultados
        return {'display': 'block'}
        
    return {'display': 'none'}

@app.callback(
    Output('output-upload-state', 'children'),
    Output('xaxis-column', 'options'),
    Output('yaxis-column', 'options'),
    Output('controls-section', 'style'),
    Output('stored-dataframe-json', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def upload_file(contents, filename):
    if contents is None:
        return no_update

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename or 'xlsx' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return dbc.Alert("Formato de archivo no soportado.", color="danger"), [], [], {'display': 'none'}, None

        all_columns = df.columns.tolist()
        if not all_columns:
            return dbc.Alert("El archivo no contiene columnas.", color="warning"), [], [], {'display': 'none'}, None

        df_json_output = df.to_json(date_format='iso', orient='split')
        column_options = [{'label': col, 'value': col} for col in all_columns]
        message = dbc.Alert(f'Archivo "{filename}" cargado exitosamente.', color="success")
        controls_style = {'display': 'block'}
        
        return message, column_options, column_options, controls_style, df_json_output

    except Exception as e:
        return dbc.Alert(f"Error al procesar el archivo: {e}", color="danger"), [], [], {'display': 'none'}, None

def generate_elegant_interpretation(model, x_cols, y_col):
    """Genera una interpretación visual y educativa del modelo."""
    r_squared = model.rsquared
    f_pvalue = model.f_pvalue
    params = model.params
    pvalues = model.pvalues
    
    # 1. Tarjeta de Calidad del Ajuste (R-cuadrado)
    r_squared_card = dbc.Card([
        dbc.CardBody([
            html.H6("Calidad del Ajuste (R²)", className="card-title"),
            html.H3(f"{r_squared:.3f}", className="text-primary"),
            dbc.Progress(label=f"{r_squared*100:.1f}%", value=r_squared*100, style={"height": "20px"}),
            html.P(f"El modelo explica el {r_squared*100:.1f}% de la variabilidad de '{y_col}'.", className="mt-2 text-muted")
        ])
    ], className="mb-3")

    # 2. Tarjeta de Significancia
    def get_badge(p_value):
        if p_value < 0.05:
            return dbc.Badge("Significativo", color="success", className="ms-2")
        return dbc.Badge("No Significativo", color="danger", className="ms-2")

    variable_significance_rows = []
    for col in x_cols:
        variable_significance_rows.append(
            html.Div([html.Span(f"Variable '{col}':"), get_badge(pvalues[col])], className="d-flex justify-content-between align-items-center mt-1")
        )

    significance_card = dbc.Card([
        dbc.CardBody([
            html.H6("Significancia Estadística (p-valor < 0.05)", className="card-title"),
            html.Div([html.Span(f"Modelo General (Prueba F):"), get_badge(f_pvalue)], className="d-flex justify-content-between align-items-center"),
            html.Hr(),
            *variable_significance_rows
        ])
    ], className="mb-3")

    # 3. Tarjeta de la Ecuación
    intercept = params['const']
    equation_str = f"{y_col} = {intercept:.2f}"
    for col in x_cols:
        coef = params[col]
        equation_str += f" {'+' if coef >= 0 else '-'} {abs(coef):.2f} * {col}"

    equation_card = dbc.Card([
        dbc.CardBody([
            html.H6("Ecuación del Modelo de Regresión", className="card-title"),
            html.Div(equation_str, className="text-center text-success fw-bold my-2 p-2 bg-light rounded", style={'fontSize': '0.9em', 'wordBreak': 'break-all'})
        ])
    ], className="mb-3")

    # 4. Alerta de Conclusión
    significant_vars = [col for col in x_cols if pvalues[col] < 0.05]
    conclusion_text = f"El modelo es globalmente {'significativo' if f_pvalue < 0.05 else 'no significativo'} y explica un {r_squared*100:.1f}% de la varianza. "
    if significant_vars:
        conclusion_text += f"Las variables significativas son: {', '.join(significant_vars)}."
    else:
        conclusion_text += "Ninguna de las variables independientes parece ser un predictor significativo."
    conclusion_color = "success" if f_pvalue < 0.05 and significant_vars else "warning"

    conclusion_alert = dbc.Alert(conclusion_text, color=conclusion_color)

    return [r_squared_card, significance_card, equation_card, conclusion_alert]

@app.callback(
    Output('regression-graph', 'figure'),
    Output('residual-graph', 'figure'),
    Output('ols-summary', 'children'),
    Output('model-interpretation', 'children'),
    Output('store-summary-text', 'data'),
    Output('store-dataframe-predictions', 'data'),
    Input('submit-button', 'n_clicks'),
    [State('xaxis-column', 'value'),
     State('yaxis-column', 'value'),
     State('stored-dataframe-json', 'data')],
    prevent_initial_call=True
)
def update_regression_analysis(n_clicks, x_cols, y_col, df_json):
    if n_clicks == 0 or not x_cols or not y_col or not df_json:
        return no_update, no_update, no_update, no_update, no_update, no_update

    # Crear una figura vacía por defecto para los casos de error
    empty_fig = px.scatter(title="Error en el análisis", template="plotly_white")
    empty_residual_fig = px.scatter(title="Error en el análisis de residuos", template="plotly_white")

    try:
        df = pd.read_json(io.StringIO(df_json), orient='split')
        
        all_cols = x_cols + [y_col]
        df_filtered = df[all_cols].copy()
        
        for col in all_cols:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

        df_filtered.dropna(inplace=True)

        # Para OLS, el número de observaciones (filas) debe ser mayor que el número de parámetros (columnas X + constante)
        if df_filtered.shape[0] <= len(x_cols) + 1:
            raise ValueError(f"No hay suficientes datos ({df_filtered.shape[0]} filas) para el número de variables seleccionadas ({len(x_cols)}). Se necesitan más de {len(x_cols) + 1} filas.")

        X = sm.add_constant(df_filtered[x_cols])
        y = df_filtered[y_col]
        model = sm.OLS(y, X).fit()
        summary_text = str(model.summary())
        
        # Añadir predicciones y residuos al dataframe para los gráficos
        df_filtered['predicted_y'] = model.predict(X)
        df_filtered['residuals'] = model.resid

        # Generar la interpretación visual
        interpretation_components = generate_elegant_interpretation(model, x_cols, y_col)

        if len(x_cols) == 1:
            # Regresión simple: gráfico de dispersión con línea de tendencia
            fig = px.scatter(df_filtered,
                             x=x_cols[0],
                             y=y_col,
                             trendline="ols",
                             title=f'Regresión: {y_col} vs. {x_cols[0]}',
                             template="plotly_white"
                            )
        else:
            # Regresión múltiple: gráfico de valores reales vs. predichos
            fig = px.scatter(df_filtered, x=y_col, y='predicted_y',
                             labels={'x': f'Valores Reales ({y_col})', 'y': 'Valores Predichos'},
                             title='Valores Reales vs. Predichos por el Modelo',
                             template="plotly_white")
            # Añadir línea de 45 grados como referencia
            min_val = min(df_filtered[y_col].min(), df_filtered['predicted_y'].min())
            max_val = max(df_filtered[y_col].max(), df_filtered['predicted_y'].max())
            fig.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color='red', dash='dash'))
        
        fig.update_layout(transition_duration=300)

        # Crear el gráfico de residuos
        residual_fig = px.scatter(
            df_filtered,
            x='predicted_y',
            y='residuals',
            title='Gráfico de Residuos (Residuos vs. Predichos)',
            labels={'predicted_y': 'Valores Predichos', 'residuals': 'Residuos'},
            template='plotly_white'
        )
        residual_fig.add_hline(y=0, line_dash="dash", line_color="red")
        residual_fig.update_layout(transition_duration=300)

        # Almacenar resultados para descarga
        df_predictions_json = df_filtered.to_json(date_format='iso', orient='split')

        return fig, residual_fig, summary_text, interpretation_components, summary_text, df_predictions_json

    except Exception as e:
        error_message = f"Se produjo un error durante el análisis: {e}"
        interpretation = dbc.Alert(error_message, color="danger")
        return empty_fig, empty_residual_fig, error_message, interpretation, no_update, no_update

# --- Callbacks de Descarga ---

@app.callback(
    Output("download-summary-txt", "data"),
    Input("btn-download-summary", "n_clicks"),
    State("store-summary-text", "data"),
    prevent_initial_call=True,
)
def download_summary(n_clicks, summary_text):
    if summary_text:
        return dict(content=summary_text, filename="resumen_regresion.txt")
    return no_update

@app.callback(
    Output("download-data-csv", "data"),
    Input("btn-download-data", "n_clicks"),
    State("store-dataframe-predictions", "data"),
    prevent_initial_call=True,
)
def download_data(n_clicks, df_json):
    if df_json:
        df = pd.read_json(io.StringIO(df_json), orient='split')
        return dcc.send_data_frame(df.to_csv, "datos_con_predicciones.csv", index=False)
    return no_update


# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=False)
