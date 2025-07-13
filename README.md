# Aplicación de Análisis de Regresión Lineal con Dash

Este repositorio contiene una aplicación web interactiva construida para democratizar el análisis de regresión lineal. Permite a cualquier usuario, sin importar su nivel de experiencia en estadística, pasar de un conjunto de datos crudo a un modelo de regresión completamente interpretado y visualizado en cuestión de segundos.

---
## 📊 Demo en Vivo
Demo: [https://app-regresion.onrender.com](https://app-regresion.onrender.com)

---

## ✨ Características Principales

-   **Interfaz Intuitiva**: Un flujo guiado en 3 pasos: Cargar, Configurar, Analizar.
-   **Interpretación Automática**: Traduce resultados estadísticos complejos (R², p-valores) a tarjetas visuales y texto claro y accionable.
-   **Visualización Dinámica**: Los gráficos se adaptan automáticamente si realizas una regresión simple o múltiple.
-   **Todo Incluido**: Desde el análisis de residuos hasta la exportación completa de resultados con un solo clic.

---

## 🚀 Funcionalidades Detalladas

Esta herramienta te guía a través del proceso completo de construcción, visualización e interpretación de un modelo de regresión lineal.

-   📤 **Carga de Datos Flexible**: Sube tus archivos `.csv` o `.xlsx` con un simple arrastrar y soltar.
-   📊 **Modelo a Medida**: Selecciona múltiples variables independientes (X) y una variable dependiente (Y) para construir tu modelo.
-   📈 **Visualización Inteligente y Dinámica**:
    -   **Gráfico Principal Adaptativo**:
        -   Para **regresión simple**, muestra un gráfico de dispersión con la línea de tendencia OLS.
        -   Para **regresión múltiple**, muestra un gráfico de **valores reales vs. predichos**, una técnica estándar para evaluar el ajuste del modelo, con una línea de referencia de 45°.
    -   **Análisis de Residuos**: Genera automáticamente un gráfico de residuos vs. valores predichos para ayudarte a verificar la homocedasticidad, uno de los supuestos clave del modelo.
-   🧠 **Interpretación Automática (¡La Magia de la App!)**: En lugar de solo mostrar números, la aplicación los traduce para ti:
    -   **Tarjeta de Calidad de Ajuste**: Un medidor visual para el **R-cuadrado (R²)** que te dice qué tan bien el modelo explica tus datos.
    -   **Tarjeta de Significancia**: Usa insignias (badges) de "Significativo" / "No Significativo" para que sepas de un vistazo si el modelo general (Prueba F) y cada una de tus variables (p-valor) son estadísticamente relevantes.
    -   **Tarjeta de Ecuación**: Presenta la ecuación matemática final del modelo, lista para que la uses.
    -   **Conclusión General**: Una alerta de resumen que te da el veredicto final sobre la utilidad del modelo en lenguaje sencillo.
-   📋 **Resumen Técnico Completo**: Accede al resumen detallado del modelo OLS de `statsmodels` para un análisis profundo.
-   💾 **Exportación con un Clic**: Descarga fácilmente todos los resultados:
    -   Gráfico principal (`.png`)
    -   Gráfico de residuos (`.png`)
    -   Resumen técnico completo (`.txt`)
    -   Tus datos originales enriquecidos con las predicciones y residuos del modelo (`.csv`).

---

## 🛠️ Tecnologías Utilizadas

-   **Backend y Lógica**: Python
-   **Framework Web**: Dash
-   **Librería de Gráficos**: Plotly
-   **Análisis Estadístico**: Statsmodels, Pandas
-   **Componentes de UI**: Dash Bootstrap Components

---

## 🏃‍♀️ Instalación y Uso

Para ejecutar la aplicación en tu máquina local, sigue estos pasos:

1.  **Clona el repositorio** (o descarga los archivos en una carpeta):
    ```bash
    git clone <URL-del-repositorio>
    cd <nombre-de-la-carpeta>
    ```

2.  **Crea un entorno virtual** (recomendado):
    ```bash
    python -m venv venv
    ```
    Actívalo:
    -   En Windows: `venv\Scripts\activate`
    -   En macOS/Linux: `source venv/bin/activate`

3.  **Instala las dependencias**:
    El archivo `requirements.txt` contiene todas las librerías necesarias.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecuta la aplicación**:
    Navega hasta el directorio del proyecto en tu terminal y ejecuta el script:
    ```bash
    python app_regresion.py
    ```

5.  **Abre la aplicación en tu navegador**:
    Ve a la dirección que aparece en la terminal (normalmente `http://127.0.0.1:8050`).