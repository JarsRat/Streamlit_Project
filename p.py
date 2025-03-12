import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import kagglehub
import warnings
import os
from transformers import pipeline
import joblib

warnings.filterwarnings('ignore')

# Inicializar el analizador de sentimientos
analizador_sentimientos = pipeline("sentiment-analysis", 
                                 model="nlptown/bert-base-multilingual-uncased-sentiment")

# Definir constantes para los archivos de cache
CACHE_FILE = "cached_predictions.csv"
MODEL_CACHE_DIR = "model_cache"
DELAY_MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "delay_model.joblib")
REFUND_MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "refund_model.joblib")
ENCODERS_PATH = os.path.join(MODEL_CACHE_DIR, "encoders.joblib")
SCALERS_PATH = os.path.join(MODEL_CACHE_DIR, "scalers.joblib")

print("Descargando dataset...")
path = kagglehub.dataset_download("logiccraftbyhimanshi/e-commerce-analytics-swiggy-zomato-blinkit")
df = pd.read_csv(path + "/Ecommerce_Delivery_Analytics_New.csv")
print("Dataset cargado correctamente.")

print("Procesando datos...")
df_clean = df.copy()
df_clean['Order Date & Time'] = pd.to_datetime(df_clean['Order Date & Time'], errors='coerce')
df_clean = df_clean.dropna(subset=['Order Date & Time'])

categorical_columns = ['Order ID', 'Customer ID', 'Platform', 'Product Category', 'Customer Feedback', 'Delivery Delay', 'Refund Requested']
for col in categorical_columns:
    df_clean[col] = df_clean[col].astype(str).str.strip()
    
print("Datos procesados correctamente.")

print("Generando gráficas...")
plt.figure(figsize=(10, 6))
sns.countplot(data=df_clean, x='Product Category', order=df_clean['Product Category'].value_counts().index, palette='viridis')
plt.xticks(rotation=45)
plt.xlabel("Categoría de Producto")
plt.ylabel("Frecuencia")
plt.title("Categorías de Productos Más Populares")
plt.show()

plt.figure(figsize=(10, 6))
avg_rating = df_clean.groupby('Platform')['Service Rating'].mean().sort_values()
sns.barplot(x=avg_rating.index, y=avg_rating.values, palette='coolwarm')
plt.xlabel("Plataforma")
plt.ylabel("Promedio de Servicio Rating")
plt.title("Promedio de Servicio Rating por Plataforma")
plt.show()

def aplicar_analisis_sentimiento(df):
    # Intentar cargar el cache existente
    try:
        cache_df = pd.read_csv(CACHE_FILE)
        cache_dict = dict(zip(cache_df['Order ID'], cache_df['Sentiment']))
    except FileNotFoundError:
        cache_dict = {}
    
    sentiments = []
    new_predictions = []
    
    for idx, row in df.iterrows():
        feedback = str(row['Customer Feedback'])
        order_id = str(row['Order ID'])
        
        # Verificar si existe en caché
        if order_id in cache_dict:
            sentiments.append(cache_dict[order_id])
        else:
            # Realizar nuevo análisis
            resultado = analizador_sentimientos(feedback)[0]
            sentiment = 'Positive' if float(resultado['label'].split()[0]) >= 4 else 'Negative' if float(resultado['label'].split()[0]) <= 2 else 'Neutral'
            sentiments.append(sentiment)
            new_predictions.append({'Order ID': order_id, 'Sentiment': sentiment})
    
    # Guardar nuevas predicciones en caché
    if new_predictions:
        new_cache_df = pd.DataFrame(new_predictions)
        if os.path.exists(CACHE_FILE):
            cache_df = pd.concat([cache_df, new_cache_df], ignore_index=True)
        else:
            cache_df = new_cache_df
        cache_df.to_csv(CACHE_FILE, index=False)
    
    df['Sentiment'] = sentiments
    return df

print("Generando gráfico de análisis de sentimientos por plataforma...")
df_clean = aplicar_analisis_sentimiento(df_clean)  # Aplicar el análisis de sentimiento
sentiment_counts = df_clean.groupby(['Platform', 'Sentiment']).size().unstack(fill_value=0)
sentiment_counts.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='coolwarm')
plt.xlabel("Plataforma")
plt.ylabel("Cantidad de Opiniones")
plt.title("Análisis de Sentimientos por Plataforma")
plt.legend(title="Sentimiento")
plt.show()

def guardar_modelos(delay_clf, refund_clf, le_dict, scaler):
    """Guarda los modelos entrenados y sus transformadores"""
    if not os.path.exists(MODEL_CACHE_DIR):
        os.makedirs(MODEL_CACHE_DIR)
    
    joblib.dump(delay_clf, DELAY_MODEL_PATH)
    joblib.dump(refund_clf, REFUND_MODEL_PATH)
    joblib.dump(le_dict, ENCODERS_PATH)
    joblib.dump(scaler, SCALERS_PATH)
    print("Modelos guardados correctamente.")

def cargar_modelos():
    """Carga los modelos entrenados y sus transformadores"""
    try:
        delay_clf = joblib.load(DELAY_MODEL_PATH)
        refund_clf = joblib.load(REFUND_MODEL_PATH)
        le_dict = joblib.load(ENCODERS_PATH)
        scaler = joblib.load(SCALERS_PATH)
        print("Modelos cargados desde caché.")
        return delay_clf, refund_clf, le_dict, scaler
    except:
        return None, None, None, None

# Modificar la parte del entrenamiento de modelos
print("Preparando modelos de predicción...")
delay_clf, refund_clf, le_dict, scaler = cargar_modelos()

if delay_clf is None:  # Si no hay modelos en caché, entrenar nuevos
    print("Entrenando nuevos modelos...")
    
    # Preparar datos para modelo de retrasos
    delay_features = ['Platform', 'Order Value (INR)', 'Product Category', 'Service Rating']
    delay_df = df_clean.dropna(subset=delay_features + ['Delivery Delay'])

    # Diccionario para almacenar los encoders
    le_dict = {}
    
    # Crear una copia del DataFrame para las transformaciones
    delay_df_encoded = delay_df.copy()
    
    # Codificar variables categóricas
    for col in ['Platform', 'Product Category']:
        le = LabelEncoder()
        delay_df_encoded[col] = le.fit_transform(delay_df[col])
        le_dict[col] = le

    # Preparar X e y
    X_delay = delay_df_encoded[delay_features].copy()
    y_delay = LabelEncoder().fit_transform(delay_df['Delivery Delay'])
    
    # Asegurarse de que Order Value (INR) y Service Rating sean numéricos
    X_delay['Order Value (INR)'] = pd.to_numeric(X_delay['Order Value (INR)'], errors='coerce')
    X_delay['Service Rating'] = pd.to_numeric(X_delay['Service Rating'], errors='coerce')
    
    # Eliminar filas con valores nulos después de la conversión
    mask = X_delay.notna().all(axis=1)
    X_delay = X_delay[mask]
    y_delay = y_delay[mask]
    
    # Escalar características
    scaler = StandardScaler()
    X_delay_scaled = scaler.fit_transform(X_delay)
    
    # Dividir datos y entrenar modelo de retrasos
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_delay_scaled, y_delay, test_size=0.2, random_state=42)
    delay_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    delay_clf.fit(X_train_d, y_train_d)
    
    # Preparar datos para modelo de devoluciones
    refund_features = ['Platform', 'Order Value (INR)', 'Product Category', 'Service Rating']
    refund_df = df_clean.dropna(subset=refund_features + ['Refund Requested'])
    
    # Crear una copia del DataFrame para las transformaciones
    refund_df_encoded = refund_df.copy()
    
    # Usar los mismos encoders para mantener consistencia
    for col in ['Platform', 'Product Category']:
        refund_df_encoded[col] = le_dict[col].transform(refund_df[col])
    
    # Preparar X e y
    X_refund = refund_df_encoded[refund_features].copy()
    y_refund = LabelEncoder().fit_transform(refund_df['Refund Requested'])
    
    # Asegurarse de que Order Value (INR) y Service Rating sean numéricos
    X_refund['Order Value (INR)'] = pd.to_numeric(X_refund['Order Value (INR)'], errors='coerce')
    X_refund['Service Rating'] = pd.to_numeric(X_refund['Service Rating'], errors='coerce')
    
    # Eliminar filas con valores nulos después de la conversión
    mask = X_refund.notna().all(axis=1)
    X_refund = X_refund[mask]
    y_refund = y_refund[mask]
    
    # Escalar características usando el mismo scaler
    X_refund_scaled = scaler.transform(X_refund)
    
    # Dividir datos y entrenar modelo de devoluciones
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_refund_scaled, y_refund, test_size=0.2, random_state=42)
    refund_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    refund_clf.fit(X_train_r, y_train_r)
    
    # Guardar modelos entrenados
    guardar_modelos(delay_clf, refund_clf, le_dict, scaler)

# Modificar las funciones de predicción
def predict_delay(order):
    """Predice si habrá retraso en la entrega"""
    try:
        # Convertir valores numéricos
        order_processed = [
            le_dict['Platform'].transform([str(order[0])])[0],
            float(order[1]),
            le_dict['Product Category'].transform([str(order[2])])[0],
            float(order[3])
        ]
        
        # Escalar datos
        order_scaled = scaler.transform([order_processed])
        return delay_clf.predict(order_scaled)[0]
    except Exception as e:
        print(f"Error al procesar la orden: {e}")
        return None

def predict_refund(order):
    """Predice si habrá solicitud de devolución"""
    try:
        # Convertir valores numéricos
        order_processed = [
            le_dict['Platform'].transform([str(order[0])])[0],
            float(order[1]),
            le_dict['Product Category'].transform([str(order[2])])[0],
            float(order[3])
        ]
        
        # Escalar datos
        order_scaled = scaler.transform([order_processed])
        return refund_clf.predict(order_scaled)[0]
    except Exception as e:
        print(f"Error al procesar la orden: {e}")
        return None

def console_interface():
    while True:
        print("\nMenú de Predicción")
        print("Opciones disponibles:")
        options = ["Predecir retraso de entrega", "Predecir si un producto será devuelto", "Salir"]
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        
        try:
            choice = int(input("\nSeleccione una opción (1-3): "))
            if choice not in [1,2,3]:
                print("Por favor seleccione una opción válida (1-3)")
                continue
        except ValueError:
            print("Por favor ingrese un número válido")
            continue

        if choice in [1,2]:
            # Selección de plataforma
            print("\nPlataformas disponibles:")
            platforms = le_dict['Platform'].classes_
            for i, platform in enumerate(platforms, 1):
                print(f"{i}. {platform}")
            
            try:
                plat_choice = int(input("\nSeleccione una plataforma (1-3): "))
                if plat_choice < 1 or plat_choice > len(platforms):
                    print("Opción de plataforma no válida")
                    continue
                platform = platforms[plat_choice-1]
            except ValueError:
                print("Por favor ingrese un número válido")
                continue

            # Valor del pedido
            try:
                order_value = float(input("\nValor del pedido (INR): "))
                if order_value <= 0:
                    print("El valor debe ser mayor a 0")
                    continue
            except ValueError:
                print("Por favor ingrese un valor numérico válido")
                continue

            # Selección de categoría
            print("\nCategorías disponibles:")
            categories = le_dict['Product Category'].classes_
            for i, category in enumerate(categories, 1):
                print(f"{i}. {category}")
            
            try:
                cat_choice = int(input("\nSeleccione una categoría (1-7): "))
                if cat_choice < 1 or cat_choice > len(categories):
                    print("Opción de categoría no válida")
                    continue
                category = categories[cat_choice-1]
            except ValueError:
                print("Por favor ingrese un número válido")
                continue

            # Rating
            try:
                rating = float(input("\nCalificación del servicio (1-5): "))
                if rating < 1 or rating > 5:
                    print("La calificación debe estar entre 1 y 5")
                    continue
            except ValueError:
                print("Por favor ingrese un número válido")
                continue

            if choice == 1:
                prediction = predict_delay([platform, order_value, category, rating])
                if prediction is not None:
                    print(f"\nPredicción de Retraso: {'Sí' if prediction == 1 else 'No'}")
            else:
                prediction = predict_refund([platform, order_value, category, rating])
                if prediction is not None:
                    print(f"\nPredicción de Devolución: {'Sí' if prediction == 1 else 'No'}")

        elif choice == 3:
            break

console_interface()

def analizar_sentimiento(texto):
    try:
        # Intentar cargar predicciones existentes
        cached_predictions = pd.read_csv(CACHE_FILE)
        if texto in cached_predictions['texto'].values:
            return cached_predictions[cached_predictions['texto'] == texto]['sentiment'].iloc[0]
    except FileNotFoundError:
        cached_predictions = pd.DataFrame(columns=['texto', 'sentiment'])
    
    # Si no está en caché, realizar nueva predicción
    resultado = analizador_sentimientos(texto)[0]
    sentiment = resultado['label']
    
    # Guardar nueva predicción en caché
    new_prediction = pd.DataFrame({'texto': [texto], 'sentiment': [sentiment]})
    cached_predictions = pd.concat([cached_predictions, new_prediction], ignore_index=True)
    cached_predictions.to_csv(CACHE_FILE, index=False)
    
    return sentiment
