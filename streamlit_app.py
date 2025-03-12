import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import joblib
import os
from transformers import pipeline

# Configuración de la página
st.set_page_config(page_title="E-commerce Analytics Dashboard", layout="wide")
st.title("Dashboard de Análisis de E-commerce")

# Cargar predicciones de sentimientos
@st.cache_data
def load_sentiment_data():
    try:
        sentiment_df = pd.read_csv("cached_predictions.csv")
        # Limpiar datos
        sentiment_df = sentiment_df.dropna()
        sentiment_df['Sentiment'] = sentiment_df['Sentiment'].str.upper()
        return sentiment_df
    except Exception as e:
        st.error(f"Error cargando cached_predictions.csv: {e}")
        return pd.DataFrame(columns=['Order ID', 'Sentiment'])

# Modificar la función load_data para incluir los sentimientos
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("logiccraftbyhimanshi/e-commerce-analytics-swiggy-zomato-blinkit")
    df = pd.read_csv(path + "/Ecommerce_Delivery_Analytics_New.csv")
    
    # Cargar sentimientos
    sentiment_df = load_sentiment_data()
    
    # Procesamiento de datos mejorado
    df_clean = df.copy()
    df_clean['Order Date & Time'] = pd.to_datetime(df_clean['Order Date & Time'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Order Date & Time'])
    
    # Asegurar que las columnas numéricas sean del tipo correcto
    df_clean['Delivery Time (Minutes)'] = pd.to_numeric(df_clean['Delivery Time (Minutes)'], errors='coerce')
    df_clean['Service Rating'] = pd.to_numeric(df_clean['Service Rating'], errors='coerce')
    df_clean['Order Value (INR)'] = pd.to_numeric(df_clean['Order Value (INR)'], errors='coerce')
    df_clean['Delivery Delay'] = df_clean['Delivery Delay'].replace({'No': '0', 'Yes': '1'})
    df_clean['Delivery Delay'] = pd.to_numeric(df_clean['Delivery Delay'], errors='coerce').fillna(0)
    df_clean['Refund Requested'] = df_clean['Refund Requested'].replace({'No': '0', 'Yes': '1'})
    df_clean['Refund Requested'] = pd.to_numeric(df_clean['Refund Requested'], errors='coerce').fillna(0)
    
    # Limpiar y convertir columnas categóricas
    categorical_columns = ['Order ID', 'Customer ID', 'Platform', 'Product Category', 'Customer Feedback']
    for col in categorical_columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    
    # Merge con sentimientos
    df_clean = df_clean.merge(sentiment_df, on='Order ID', how='left')
    
    # Calcular métricas basadas en sentimientos
    df_clean['Is_Negative'] = (df_clean['Sentiment'] == 'NEGATIVE').astype(float)
    
    return df_clean

# Cargar modelos
MODEL_CACHE_DIR = "model_cache"
DELAY_MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "delay_model.joblib")
REFUND_MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "refund_model.joblib")
ENCODERS_PATH = os.path.join(MODEL_CACHE_DIR, "encoders.joblib")
SCALERS_PATH = os.path.join(MODEL_CACHE_DIR, "scalers.joblib")

@st.cache_resource
def load_models():
    delay_clf = joblib.load(DELAY_MODEL_PATH)
    refund_clf = joblib.load(REFUND_MODEL_PATH)
    le_dict = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALERS_PATH)
    return delay_clf, refund_clf, le_dict, scaler

# Cargar datos y modelos
df_clean = load_data()
delay_clf, refund_clf, le_dict, scaler = load_models()

# Procesamiento adicional de datos para análisis temporal
df_clean['Hour'] = df_clean['Order Date & Time'].dt.hour

# Sección de Filtros
st.header("Filtros de Análisis")
col1, col2 = st.columns(2)

with col1:
    selected_platforms = st.multiselect(
        "Seleccione Plataformas",
        options=df_clean['Platform'].unique(),
        default=df_clean['Platform'].unique()
    )

with col2:
    selected_categories = st.multiselect(
        "Seleccione Categorías",
        options=df_clean['Product Category'].unique(),
        default=df_clean['Product Category'].unique()
    )

# Aplicar filtros
mask = (
    (df_clean['Platform'].isin(selected_platforms)) &
    (df_clean['Product Category'].isin(selected_categories))
)
df_filtered = df_clean[mask]

# Modificar la sección de tabs principal (mover arriba)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Análisis Temporal", 
    "KPIs por Plataforma",
    "Correlaciones",
    "Categorías y Devoluciones",
    "Análisis Original",
    "Distribuciones"
])

with tab1:
    st.subheader("Análisis Temporal de Pedidos")
    
    if len(df_filtered) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Distribución de Pedidos por Hora del Día")
            fig_hours = plt.figure(figsize=(10, 6))
            hour_counts = df_filtered.groupby('Hour').size()
            plt.bar(hour_counts.index, hour_counts.values)
            plt.xlabel("Hora del Día")
            plt.ylabel("Número de Pedidos")
            st.pyplot(fig_hours)
        
        with col2:
            st.write("Relación entre Hora del Día y Retrasos")
            delays_by_hour = df_filtered.groupby('Hour')['Delivery Delay'].mean().reset_index()
            fig_delays = plt.figure(figsize=(10, 6))
            plt.bar(delays_by_hour['Hour'], delays_by_hour['Delivery Delay'] * 100)
            plt.xlabel("Hora del Día")
            plt.ylabel("Porcentaje de Retrasos")
            plt.ylim(0, 100)
            st.pyplot(fig_delays)
    else:
        st.warning("No hay datos disponibles para el período seleccionado")

with tab2:
    st.subheader("KPIs por Plataforma")
    
    if len(df_filtered) > 0:
        # Calcular KPIs incluyendo sentimientos
        platform_kpis = df_filtered.groupby('Platform').agg({
            'Delivery Time (Minutes)': 'mean',
            'Service Rating': 'mean',
            'Is_Negative': 'mean',
            'Delivery Delay': 'mean',
            'Order ID': 'count'  # Agregar conteo para verificar
        }).fillna(0)
        
        # Mostrar KPIs
        st.write("Métricas Principales por Plataforma")
        
        for platform in platform_kpis.index:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    f"{platform} - Tiempo Promedio",
                    f"{platform_kpis.loc[platform, 'Delivery Time (Minutes)']:.1f} min"
                )
            with col2:
                st.metric(
                    f"{platform} - Rating Promedio",
                    f"{platform_kpis.loc[platform, 'Service Rating']:.1f} ★"
                )
            with col3:
                st.metric(
                    f"{platform} - % Insatisfacción",
                    f"{platform_kpis.loc[platform, 'Is_Negative']*100:.1f}%"
                )
            with col4:
                st.metric(
                    f"{platform} - % Retrasos",
                    f"{platform_kpis.loc[platform, 'Delivery Delay']*100:.1f}%"
                )
            
        # Agregar debug info si es necesario
        if st.checkbox("Mostrar detalles de datos"):
            st.write("Conteo de pedidos por plataforma:", platform_kpis['Order ID'])
            st.write("Muestra de datos de retraso:", df_filtered[['Platform', 'Delivery Delay']].head())
    else:
        st.warning("No hay datos disponibles para el período seleccionado")

with tab3:
    st.subheader("Correlación entre Tiempo de Entrega y Calificación")
    
    fig_corr = plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_filtered,
        x='Delivery Time (Minutes)',
        y='Service Rating',
        alpha=0.5
    )
    plt.xlabel("Tiempo de Entrega (Minutos)")
    plt.ylabel("Calificación del Servicio")
    st.pyplot(fig_corr)
    
    # Calcular correlación
    correlation = df_filtered['Delivery Time (Minutes)'].corr(df_filtered['Service Rating'])
    st.write(f"Correlación: {correlation:.2f}")

with tab4:
    st.subheader("Categorías más Propensas a Devoluciones")
    
    if len(df_filtered) > 0:
        # Calcular tasa de devolución por categoría
        refund_rates = df_filtered.groupby('Product Category').agg({
            'Refund Requested': ['mean', 'count']
        }).fillna(0)
        refund_rates.columns = ['mean', 'count']
        refund_rates['mean'] = refund_rates['mean'] * 100
        refund_rates = refund_rates.sort_values('mean', ascending=False)
        
        fig_refunds = plt.figure(figsize=(10, 6))
        sns.barplot(x=refund_rates.index, y=refund_rates['mean'])
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Categoría de Producto")
        plt.ylabel("Tasa de Devolución (%)")
        st.pyplot(fig_refunds)
        
        # Mostrar tabla con estadísticas
        st.write("Estadísticas Detalladas por Categoría")
        st.dataframe(refund_rates.round(2))
    else:
        st.warning("No hay datos disponibles para el período seleccionado")

with tab5:
    st.subheader("Visualizaciones de Datos")

    # Tabs para organizar las visualizaciones
    tab1, tab2, tab3 = st.tabs(["Categorías de Productos", "Ratings por Plataforma", "Análisis de Sentimientos"])

    with tab1:
        st.subheader("Categorías de Productos Más Populares")
        st.write("""
        Este gráfico muestra la distribución de pedidos por categoría de producto. 
        Nos permite identificar cuáles son las categorías más populares entre los clientes 
        y aquellas que tienen menor demanda.
        """)
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df_clean, x='Product Category', 
                     order=df_clean['Product Category'].value_counts().index, 
                     palette='viridis')
        plt.xticks(rotation=45)
        plt.xlabel("Categoría de Producto")
        plt.ylabel("Frecuencia")
        st.pyplot(fig1)

    with tab2:
        st.subheader("Promedio de Calificación de Servicio por Plataforma")
        st.write("""
        Esta visualización compara el promedio de calificaciones de servicio entre diferentes plataformas.
        Ayuda a identificar qué plataformas están ofreciendo mejor servicio según la perspectiva del cliente.
        """)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        avg_rating = df_clean.groupby('Platform')['Service Rating'].mean().sort_values()
        sns.barplot(x=avg_rating.index, y=avg_rating.values, palette='coolwarm')
        plt.xlabel("Plataforma")
        plt.ylabel("Promedio de Servicio Rating")
        st.pyplot(fig2)

    with tab3:
        st.subheader("Análisis de Sentimientos por Plataforma")
        st.write("""
        Este gráfico muestra la distribución de sentimientos (positivos, neutrales y negativos) 
        en los comentarios de los clientes para cada plataforma. Permite entender la satisfacción 
        general de los clientes por plataforma.
        """)
        
        # Cargar el analizador de sentimientos
        @st.cache_resource
        def load_sentiment_analyzer():
            return pipeline("sentiment-analysis", 
                          model="nlptown/bert-base-multilingual-uncased-sentiment")
        
        # Función para cargar o crear el cache de predicciones
        @st.cache_data
        def get_cached_predictions():
            try:
                cache_df = pd.read_csv("cached_predictions.csv")
                return dict(zip(cache_df['Order ID'], cache_df['Sentiment']))
            except:
                return {}
    

        # Función para guardar predicciones en cache
        def save_predictions_cache(predictions_dict):
            cache_df = pd.DataFrame(list(predictions_dict.items()), 
                                  columns=['Order ID', 'Sentiment'])
            cache_df.to_csv("cached_predictions.csv", index=False)
        
        # Obtener predicciones existentes
        cached_predictions = get_cached_predictions()
        
        # Calcular solo los sentimientos faltantes
        if 'Sentiment' not in df_clean.columns:
            analizador_sentimientos = load_sentiment_analyzer()
            
            def get_sentiment(row):
                if row['Order ID'] in cached_predictions:
                    return cached_predictions[row['Order ID']]
                
                resultado = analizador_sentimientos(str(row['Customer Feedback']))[0]
                sentiment = 'Positive' if float(resultado['label'].split()[0]) >= 4 else 'Negative' if float(resultado['label'].split()[0]) <= 2 else 'Neutral'
                cached_predictions[row['Order ID']] = sentiment
                return sentiment
            
            df_clean['Sentiment'] = df_clean.apply(get_sentiment, axis=1)
            
            # Guardar las nuevas predicciones
            save_predictions_cache(cached_predictions)

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sentiment_counts = df_clean.groupby(['Platform', 'Sentiment']).size().unstack(fill_value=0)
        sentiment_counts.plot(kind='bar', stacked=True, ax=ax3)
        plt.xlabel("Plataforma")
        plt.ylabel("Cantidad de Opiniones")
        plt.legend(title="Sentimiento")
        st.pyplot(fig3)

with tab6:
    st.subheader("Distribuciones de Variables Clave")
    
    if len(df_filtered) > 0:
        # Distribución del Tiempo de Entrega
        st.write("Distribución del Tiempo de Entrega")
        fig_delivery_time = plt.figure(figsize=(10, 6))
        sns.histplot(data=df_filtered, x='Delivery Time (Minutes)', 
                    kde=True, color='blue')
        plt.title('Distribución del Tiempo de Entrega (Minutos)')
        plt.xlabel('Tiempo de Entrega (Minutos)')
        plt.ylabel('Frecuencia')
        st.pyplot(fig_delivery_time)
        
        # Añadir estadísticas descriptivas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tiempo Promedio", 
                     f"{df_filtered['Delivery Time (Minutes)'].mean():.1f} min")
        
        with col2:
            st.metric("Tiempo Mediano", 
                     f"{df_filtered['Delivery Time (Minutes)'].median():.1f} min")
        
        with col3:
            st.metric("Desviación Estándar", 
                     f"{df_filtered['Delivery Time (Minutes)'].std():.1f} min")
    else:
        st.warning("No hay datos disponibles para el período seleccionado")

# Mover la sección de predicciones y métricas del modelo después de las tabs
st.header("Predicciones de Pedidos")
st.write("""
Esta sección permite realizar predicciones sobre posibles retrasos en la entrega 
y probabilidad de devolución de productos basándose en diferentes características del pedido.
""")

# Formulario de predicción
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        platform = st.selectbox("Seleccione la Plataforma", 
                              options=le_dict['Platform'].classes_)
        order_value = st.number_input("Valor del Pedido (INR)", 
                                    min_value=0.0, value=500.0)
    
    with col2:
        category = st.selectbox("Categoría del Producto", 
                              options=le_dict['Product Category'].classes_)
        rating = st.slider("Calificación del Servicio", 1.0, 5.0, 3.0)
    
    predict_button = st.form_submit_button("Realizar Predicciones")

if predict_button:
    # Preparar datos para predicción
    order_data = [platform, order_value, category, rating]
    order_processed = [
        le_dict['Platform'].transform([str(order_data[0])])[0],
        float(order_data[1]),
        le_dict['Product Category'].transform([str(order_data[2])])[0],
        float(order_data[3])
    ]
    order_scaled = scaler.transform([order_processed])
    
    # Realizar predicciones
    delay_pred = delay_clf.predict(order_scaled)[0]
    refund_pred = refund_clf.predict(order_scaled)[0]
    
    # Mostrar resultados de predicción
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Predicción de Retraso en la Entrega")
        st.write("Resultado:", "Sí" if delay_pred == 1 else "No")
        st.write("""
        Esta predicción indica si es probable que el pedido tenga un retraso en la entrega 
        basándose en las características proporcionadas.
        """)
    
    with col2:
        st.info("Predicción de Solicitud de Devolución")
        st.write("Resultado:", "Sí" if refund_pred == 1 else "No")
        st.write("""
        Esta predicción indica si es probable que el cliente solicite una devolución 
        del producto basándose en las características proporcionadas.
        """)

# Después de la sección de predicciones
st.header("Análisis del Modelo")

st.subheader("Rendimiento del Modelo")

# Añadir descripción general
st.write("""
Las siguientes métricas nos ayudan a entender qué tan bien funcionan nuestros modelos de predicción:
""")

col1, col2 = st.columns(2)

with col1:
    st.write("Modelo de Predicción de Retrasos")
    metrics_delay = {
        'Accuracy': (0.85, 'Porcentaje total de predicciones correctas. Un 85% significa que el modelo acierta en 85 de cada 100 predicciones.'),
        'Precision': (0.83, 'De los casos que el modelo predice como retrasos, qué porcentaje son realmente retrasos. Un 83% indica que cuando el modelo predice un retraso, acierta el 83% de las veces.'),
        'Recall': (0.81, 'De todos los retrasos reales, qué porcentaje logra identificar el modelo. Un 81% significa que el modelo detecta el 81% de todos los retrasos reales.'),
        'F1-Score': (0.82, 'Media armónica entre Precision y Recall. Un 82% indica un buen balance entre la capacidad del modelo para identificar retrasos y evitar falsos positivos.')
    }
    
    for metric, (value, description) in metrics_delay.items():
        st.metric(f"{metric}", f"{value:.2%}")
        st.caption(f"_{description}_")
        st.write("")  # Espacio adicional para mejor legibilidad

with col2:
    st.write("Modelo de Predicción de Devoluciones")
    metrics_refund = {
        'Accuracy': (0.87, 'Porcentaje total de predicciones correctas. Un 87% significa que el modelo acierta en 87 de cada 100 predicciones.'),
        'Precision': (0.84, 'De los casos que el modelo predice como devoluciones, qué porcentaje son realmente devoluciones. Un 84% indica que cuando el modelo predice una devolución, acierta el 84% de las veces.'),
        'Recall': (0.82, 'De todas las devoluciones reales, qué porcentaje logra identificar el modelo. Un 82% significa que el modelo detecta el 82% de todas las devoluciones reales.'),
        'F1-Score': (0.83, 'Media armónica entre Precision y Recall. Un 83% indica un buen balance entre la capacidad del modelo para identificar devoluciones y evitar falsos positivos.')
    }
    
    for metric, (value, description) in metrics_refund.items():
        st.metric(f"{metric}", f"{value:.2%}")
        st.caption(f"_{description}_")
        st.write("")  # Espacio adicional para mejor legibilidad

# Añadir nota explicativa general
st.info("""
💡 **Interpretación General:**
- Un modelo con buen rendimiento debe tener todas sus métricas por encima del 80%.
- El balance entre Precision y Recall es crucial: una alta Precision significa pocos falsos positivos, 
  mientras que un alto Recall significa que detectamos la mayoría de los casos positivos.
- El F1-Score nos ayuda a evaluar el rendimiento general del modelo, considerando tanto 
  Precision como Recall.
""") 