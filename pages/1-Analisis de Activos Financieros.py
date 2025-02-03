import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import os
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from itertools import product
import base64

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
# Suprimir advertencias ValueWarning
warnings.simplefilter("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Configuración de Streamlit
st.set_page_config(page_title="Analisis de Acciones", page_icon="img/stock-market.png", layout="wide")

theme_plotly = None

#"""" codigo de particulas que se agregan en le background""""
particles_js = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Particles.js</title>
  <style>
  #particles-js {
    background-color: #191970;    
    position: fixed;
    width: 100vw;
    height: 100vh;
    top: 0;
    left: 0;
    z-index: -1; /* Send the animation to the back */
  }
  .content {
    position: relative;
    z-index: 1;
    color: white;
  }
  
</style>
</head>
<body>
  <div id="particles-js"></div>
  <div class="content">
    <!-- Placeholder for Streamlit content -->
  </div>
  <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
  <script>
    particlesJS("particles-js", {
      "particles": {
        "number": {
          "value": 300,
          "density": {
            "enable": true,
            "value_area": 800
          }
        },
        "color": {
          "value": "#fffc33"
        },
        "shape": {
          "type": "circle",
          "stroke": {
            "width": 0,
            "color": "#000000"
          },
          "polygon": {
            "nb_sides": 5
          },
          "image": {
            "src": "img/github.svg",
            "width": 100,
            "height": 100
          }
        },
        "opacity": {
          "value": 0.5,
          "random": false,
          "anim": {
            "enable": false,
            "speed": 1,
            "opacity_min": 0.2,
            "sync": false
          }
        },
        "size": {
          "value": 2,
          "random": true,
          "anim": {
            "enable": false,
            "speed": 40,
            "size_min": 0.1,
            "sync": false
          }
        },
        "line_linked": {
          "enable": true,
          "distance": 100,
          "color": "#fffc33",
          "opacity": 0.22,
          "width": 1
        },
        "move": {
          "enable": true,
          "speed": 0.2,
          "direction": "none",
          "random": false,
          "straight": false,
          "out_mode": "out",
          "bounce": true,
          "attract": {
            "enable": false,
            "rotateX": 600,
            "rotateY": 1200
          }
        }
      },
      "interactivity": {
        "detect_on": "canvas",
        "events": {
          "onhover": {
            "enable": true,
            "mode": "grab"
          },
          "onclick": {
            "enable": true,
            "mode": "repulse"
          },
          "resize": true
        },
        "modes": {
          "grab": {
            "distance": 100,
            "line_linked": {
              "opacity": 1
            }
          },
          "bubble": {
            "distance": 400,
            "size": 2,
            "duration": 2,
            "opacity": 0.5,
            "speed": 1
          },
          "repulse": {
            "distance": 200,
            "duration": 0.4
          },
          "push": {
            "particles_nb": 2
          },
          "remove": {
            "particles_nb": 3
          }
        }
      },
      "retina_detect": true
    });
  </script>
</body>
</html>
"""
globe_js = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vanta Globe Animation</title>
    <style type="text/css">
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }
      body {
        overflow: hidden;
        height: 100%;
        margin: 0;
        background-color: #1817ed; /* Fondo azul */
      }
      #canvas-globe {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
      }
    </style>
  </head>
  <body>
    <div id="canvas-globe"></div>       

    <!-- Scripts de Three.js y Vanta.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vanta/0.5.24/vanta.globe.min.js"></script>

    <script type="text/javascript">      
      document.addEventListener("DOMContentLoaded", function() {
        VANTA.GLOBE({
          el: "#canvas-globe", // El elemento donde se renderiza la animación
          mouseControls: true,
          touchControls: true,
          gyroControls: false,
          minHeight: 200.00,
          minWidth: 200.00,
          scale: 1.00,
          scaleMobile: 1.00,
          color: 0xd1ff3f, // Color verde amarillento
          backgroundColor: 0x1817ed // Fondo azul
        });
      });
    </script>
  </body>
</html>
"""
waves_js = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vanta Waves Animation</title>
    <style type="text/css">
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }
      html, body {
        height: 100%;
        margin: 0;
        overflow: hidden;
      }
      #canvas-dots {
        position: absolute;
        width: 100%;
        height: 100%;
      }
    </style>
  </head>
  <body>
    <div id="canvas-waves"></div>       
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vanta/0.5.24/vanta.waves.min.js"></script>
    
    <script type="text/javascript">      
      document.addEventListener("DOMContentLoaded", function() {
        VANTA.WAVES({
          el: "#canvas-waves", // Especificar el contenedor donde debe renderizarse
           mouseControls: true,
           touchControls: true,
           gyroControls: false,
           minHeight: 200.00,
           minWidth: 200.00,
           scale: 1.00,
           scaleMobile: 1.00,
           color: 0x15159b
        });
      });
    </script>
  </body>
</html>
"""

#""" imagen de background"""
def add_local_background_image(image):
  with open(image, "rb") as image:
    encoded_string = base64.b64encode(image.read())
    st.markdown(
      f"""
      <style>
      .stApp{{
        background-image: url(data:files/{"jpg"};base64,{encoded_string.decode()});
      }}    
      </style>
      """,
      unsafe_allow_html=True
    )
add_local_background_image("img/fondo.jpg")

#""" imagen de sidebar"""
def add_local_sidebar_image(image):
  with open(image, "rb") as image:
    encoded_string = base64.b64encode(image.read())
    st.markdown(
      f"""
      <style>
      .stSidebar{{
        background-image: url(data:files/{"jpg"};base64,{encoded_string.decode()});
      }}    
      </style>
      """,
      unsafe_allow_html=True
    )

add_local_sidebar_image("img/fondo1.jpg")

#-------------- animacion con css de los botones modelo Arima ------------------------
with open('style/style.css') as f:
        css = f.read()
        
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

st.logo(image="img/market.png",size='large')

#st.write('###')
#-------------------------desarrollo de modelo Arima --------------------
# Título principal
st.button("Modelo de pronóstico ARIMA para accion de EEUU", key="arima", use_container_width=True)

col1, col2, col3 = st.columns(3,gap="small",vertical_alignment="center")
# ----------------- Parámetros del análisis -------------------
if st.container(border=True):
    st.button("Digitar Parámetros para ejecutar el Modelo Arima", key="arimados", use_container_width=True)
    col1, col2,col3  = st.columns(3, gap="small", vertical_alignment="center", border=True)
    with col1:
        stock_symbol = st.text_input("Símbolo de Acción (Ticker)", "AAPL", help="Ejemplo: AAPL para Apple, TSLA para Tesla, etc.")
        stock_symbol = stock_symbol.upper()
    with col2:
        periodo = st.number_input("Meses históricos para calcular", min_value=1, max_value=12, value=6, step=1, help="Valor debe oscilar entre 1/12")
    with col3:
        prediction_ahead = st.number_input("Días a predecir precio", min_value=1, max_value=30, value=15, step=1, help="Valor debe oscilar entre 1/30")
    st.write('---')    
    #st.write("###")    
    #columns = st.columns((2, 1, 3))
    #if st.columns[1].button("Predecir", key="predecir"):
    if st.button("Generar Modelo Arima", key="predarima"):    
      with st.status("Generando Modelo Arima...", expanded=True) as status:
        st.write("Buscando datos...")
        time.sleep(2)
        st.write("Desplegando Modelo Arima.")
        time.sleep(1)
        st.write("Ingresando Datos...")
        time.sleep(1)
        status.update(
            label="Modelo completo!", state="complete", expanded=True
      )
      
        # Parámetros para el cálculo del modelo ARIMA       
        # Paso 1: Obtener datos de la acción para los últimos X meses
        df_data = yf.download(stock_symbol, period=f'{periodo}mo', interval='1d')
        df_data = df_data[['Close']].dropna()
        
        # Preparar la división de entrenamiento y prueba (80% para entrenamiento, 20% para prueba)
        train_size = int(len(df_data) * 0.8)
        train, test = df_data[:train_size], df_data[train_size:]
        
        # Paso 2: Ajustar el modelo ARIMA
        p_values = range(0, 4)  # Definir el rango para ARIMA(p,d,q)
        d_values = range(0, 2)
        q_values = range(0, 4)

        def evaluate_arima_model(train, test, arima_order):
            try:
                model = ARIMA(train, order=arima_order)
                model_fit = model.fit()
                predictions = model_fit.forecast(steps=len(test))
                mse = mean_squared_error(test, predictions)
                return mse, model_fit
            except Exception as e:
                print(f"Error al ajustar ARIMA con parámetros {arima_order}: {e}")
                st.write(f"Error al ajustar ARIMA con parámetros {arima_order}: {e}")
                return float('inf'), None

        results = []
        mse_values = []  # To store the MSE for each model combination
        arima_combinations = []  # Store the combinations of (p, d, q)
        for p, d, q in product(p_values, d_values, q_values):
            arima_order = (p, d, q)
            mse, model_fit = evaluate_arima_model(train['Close'], test['Close'], arima_order)
            results.append((arima_order, mse, model_fit))
            mse_values.append(mse)
            arima_combinations.append(f"({p},{d},{q})")

        # Seleccionar el mejor modelo
        best_order, best_mse, best_model = min(results, key=lambda x: x[1])

        # Comprobar si el modelo encontrado es válido
        if best_model is None:
            st.error("No se pudo encontrar un modelo ARIMA válido. Por favor, ajusta los parámetros.")
        else:
            # Paso 3: Realizar la predicción
            forecast = best_model.forecast(steps=len(test) + prediction_ahead)
            
            # Asegurarnos de que forecast es un array de numpy o manejarlo como una pandas Series
            if isinstance(forecast, pd.Series):  # Si forecast es un pandas Series
                forecast = forecast.values  # Convertimos a numpy array

            # Ahora, acceder de forma segura al último valor de forecast
            if len(forecast) > 0:
                last_predicted_price = float(forecast[-1])  # Último valor de la predicción
            else:
                st.error("No se generaron predicciones adecuadas. Revisa los parámetros del modelo.")
                last_predicted_price = None
            
            # Último precio de cierre
            latest_close_price = float(df_data['Close'].iloc[-1])

            # Diseño centrado para las métricas  
            st.write('---')    
            #st.write("###")           
            col1, col2 = st.columns(2, border=True, vertical_alignment="center")
            st.markdown("<div style='display: flex; justify-content: center; align-items: center; height: 100%;'>", unsafe_allow_html=True)
            with col1:
                st.subheader(f"Precio de Cierre de: {stock_symbol}")
                st.button(f" -- U$D {latest_close_price:,.2f}  --", key="inpulsearima") 
            with col2:
                st.subheader(f"Precio proyectado a {prediction_ahead} Día/s")
                if last_predicted_price is not None:
                    st.button(f" -- U$D {last_predicted_price:,.2f}  --", key="toinpulsearima")
            st.markdown("</div>", unsafe_allow_html=True)
  
            # Graficar los resultados
            plt.figure(figsize=(14, 4))  # Ajuste de altura para que el gráfico sea más bajo
            plt.plot(df_data.index, df_data['Close'], label='Actual', color='blue')
            plt.axvline(x=df_data.index[train_size], color='gray', linestyle='--', label='Train/Test Split')

            # Datos de entrenamiento/prueba y predicciones
            plt.plot(train.index, train['Close'], label='Train Data', color='green')
            plt.plot(test.index, forecast[:len(test)], label='Test Predictions', color='orange')

            # Predicciones futuras
            future_index = pd.date_range(start=test.index[-1], periods=prediction_ahead + 1, freq='D')[1:]
            plt.plot(future_index, forecast[len(test):], label=f'{prediction_ahead}-Day Forecast', color='red')

            plt.title(f'{stock_symbol} Predicciones del modelo ARIMA')
            plt.xlabel('Días')
            plt.ylabel('Precio (USD)')
            plt.legend()

            with st.container(border=True):          
                st.subheader(f"Predicción del Modelo ARIMA para: {stock_symbol}")    
                st.pyplot(plt)    
                
st.write('###')
# -----------------desarrollo de model LSTM----------------------------
# Título principal
st.button("Modelo de pronóstico LSTM para acciones de EEUU", key="lstm", use_container_width=True)
st.write('###')
st.warning("este modelo hace un pedido de cotizaciones a Yahoo Finance, lo cual puede fallar la respuesta de los datos, abriendo la posiblidad de dar un mensaje de error. Si se produce este evento por favor vuelta a cargar la página o refreque la misma. Otro dato no menor es que este modelo trabajo con periodo de analisis de 60 dias y el variar los meses historicos y dias a predecir puede modificar sensiblente la respueta de precios proyectado o de predicción")
st.write('---')
    # ----------------- Parámetros del análisis -------------------
if st.container(border=True):
    st.button("Digitar Parámetros para ejecutar modelo LSTM", key="lstmdos", use_container_width=True)
    col1, col2,col3  = st.columns(3, gap="small", vertical_alignment="center", border=True)
    with col1:
        asset_symbol = st.text_input("Símbolo de Acción (Ticker)", "AAPL", help="Ejemplo: AAPL para Apple, TSLA para Tesla, etc.", key = "<uniquevalueofsomesort>").upper()            
    with col2:
        periodo_tiempo = st.number_input("Meses históricos para calcular", min_value=1, max_value=12, value=6, step=1, help="Valor debe oscilar entre 1/12", key = "<uniquevalueofsomesort1>")
    with col3:
        prediction_days = st.number_input("Días a predecir precio", min_value=1, max_value=30, value=15, step=1, help="Valor debe oscilar entre 1/30", key = "<uniquevalueofsomesort2>")
    st.write('---')    

    if st.button("Generar Modelo LSTM", key="predlstm"):    
      with st.status("Generando Modelo LSTM...", expanded=True) as status:
        st.write("Buscando datos...")
        time.sleep(2)
        st.write("Desplegando Modelo LSTM.")
        time.sleep(1)
        st.write("Ingresando Datos...")
        time.sleep(1)
        status.update(
           label="Modelo completo!", state="complete", expanded=True
      )              
        # Paso 1: Obtener datos de criptomonedas para el año    
        df_data = yf.download(asset_symbol, period=f'{periodo_tiempo}mo', interval='1d')
        # Eliminar cualquier valor nulo antes de realizar el escalado
        df_data = df_data[['Close']].dropna() 
        # Escalado de los datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_data)
        # Parámetros de la red LSTM   
        train_size = int(len(scaled_data) * 0.8)    
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]      
        #test_size = int(len(scaled_data) * 0.2) 
        #test_data = scaled_data[:test_size]
        
        # Función para crear los datasets
        def create_dataset(data, time_step=1):
            X, y = [], []
            for i in range(len(data) - time_step):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        # Dividir los datos en conjunto de entrenamiento y prueba
        time_step = 60
        X_train, y_train = create_dataset(scaled_data[:train_size], time_step)
        X_test, y_test = create_dataset(scaled_data[train_size-time_step:], time_step) 
        # Reshape para el modelo LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Crear el modelo LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=1, epochs=5, verbose=0)

        # Predicción sobre los datos de prueba
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)   
    
        # transformo invirtiendo predicciones en valores actuales
        train_predictions = scaler.inverse_transform(train_predictions)
        y_train = scaler.inverse_transform(y_train.reshape(-1,1))
        test_predictions = scaler.inverse_transform(test_predictions)
        y_test = scaler.inverse_transform(y_test.reshape(-1,1))
        
        # Pronóstico para los días proyectados    
        last_60_days = scaled_data[-time_step:]
        future_input = last_60_days.reshape(1, time_step, 1)
        future_forecast = []
        for _ in range(prediction_days):  # Predicción para los próximos días
            next_pred = model.predict(future_input)[0, 0]
            future_forecast.append(next_pred)           
            next_input = np.append(future_input[0, 1:], [[next_pred]], axis=0) 
            future_input = next_input.reshape(1, time_step, 1)

        future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1,1))
        # ultimo cierre y ultimo precio pedicho
        latest_close_price = float(df_data['Close'].iloc[-1])
        last_predicted_price = float(future_forecast[-1])
    
        col1, col2= st.columns(2, border=True, vertical_alignment="center")        
        st.markdown("<div style='display: flex; justify-content: center; align-items: center; height: 100%;'>", unsafe_allow_html=True)
        with col1:     
                valor=0              
                st.subheader(f"Precio de Cierre de : {asset_symbol}")

                st.button(f" -- U$D {latest_close_price:,.2f}  --", key=f"inpulselstm") 
        with col2:
                st.subheader(f"""Precio proyectado a {prediction_days} Dia/s """)
                st.button(f" -- U$D  {last_predicted_price:,.2f}  --", key="toinpulselstm")     
        st.markdown("</div>", unsafe_allow_html=True) 
    
        # Graficar los resultados
        plt.figure(figsize=(14, 5))
        plt.plot(df_data.index, df_data['Close'], label='Actual', color='blue')
        plt.axvline(x=df_data.index[train_size], color='gray', linestyle='--', label='Train/Test Split')

        # Datos de entrenamiento/prueba y predicciones
        train_range = df_data.index[time_step:train_size]
        test_range = df_data.index[train_size:train_size + len(test_predictions)]
        plt.plot(train_range, train_predictions[:len(train_range)], label='Train Predictions', color='green')
        plt.plot(test_range, test_predictions[:len(test_range)], label='Test Predictions', color='orange')
        
        future_index = pd.date_range(start=df_data.index[-1], periods=prediction_days + 1, freq='D')[1:]    
        plt.plot(future_index, future_forecast, label=f'{prediction_days}-Day Forecast', color='red')

        plt.title(f'{asset_symbol} Predicciones del modelo LSTM')
        plt.xlabel('Días')
        plt.ylabel('Precio (USD)')
        plt.legend()

        # Mostrar el gráfico
        st.subheader(f"Predicción del Modelo LSTM para : {asset_symbol} ")
        st.pyplot(plt)
        
st.write("###")    
st.write("---")  
#------------------desarrollo de tendencia de accion con regresión -------------
# Función para obtener datos desde Yahoo Finance con el intervalo seleccionado
def get_stock_data(ticker, period, interval):
    # Intervalos válidos para cada período
    valid_intervals = {
        '1d': ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '4h'],
        '5d': ['1m', '2m', '5m', '15m', '30m', '60m', '1h', '4h', '1d'],
        '1mo': ['1m', '2m', '5m', '15m', '30m', '60m', '1h', '4h', '1d','5d'],
        '3mo': ['1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '5d'],
        '6mo': ['1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '5d','1wk','1m'],
        '12mo': ['1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '5d', '1wk','1mo','3mo'],
        '1y': ['1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d','5d','1wk','1mo','3mo'],
        '2y': ['1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d','5d','1wk','1mo','3mo'],
        '5y': ['1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d','5d','1wk','1mo','3mo','6mo','1y'],
        '10y': ['1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d','5d','1wk','1mo','3mo','6mo','1y'],
    }

    # Verificar si el intervalo es válido para el periodo
    if interval not in valid_intervals.get(period, []):
        st.error(f"El intervalo {interval} no es válido para el período {period}. Por favor, seleccione un intervalo compatible.")
        return None

    # Obtener los datos de Yahoo Finance
    data = yf.download(ticker, period=period, interval=interval)
    if data.empty:
        return None  # Si no hay datos, retornamos None
    data = data[['Close']]  # Solo tomamos los precios de cierre
    data.reset_index(inplace=True)  # Restablecer el índice para que la fecha esté como columna
    return data

# Función para correr la regresión y categorizar las tendencias
def categorize_trends(ticker, period, interval):
    df = get_stock_data(ticker, period, interval)
    
    # Verificar si no se descargaron datos
    if df is None or df.empty:
        return {
            'Simbolo': ticker,
            'Coeficiente de Tendencia': None,
            'Categoria': 'Error',
            'R-cuadrado': None,
            'Mensaje Error': 'No se encontraron datos para el intervalo seleccionado.'
        }
    
    df['time_index'] = range(len(df))
    X = sm.add_constant(df['time_index'])
    y = df['Close']
    
    try:
        model = sm.OLS(y, X).fit()
        trend = model.params['time_index']

        if trend > 0.001:
            category = 'Tendencia Positiva'
        elif trend < -0.001:
            category = 'Tendencia Negativa'
        else:
            category = 'Sin Tendencia'

        return {
            'Simbolo': ticker,
            'Coeficiente de Tendencia': round(trend,5),
            'Categoria': category,
            'R-cuadrado': round(model.rsquared,5)
        }
    except Exception as e:
        return {
            'Simbolo': ticker,
            'Coeficiente de Tendencia': None,
            'Categoria': 'Error',
            'R-cuadrado': None,
            'Mensaje Error': str(e)
        }

# Aplicación Streamlit
def main():    
    # Título principal
    st.button("Modelo de Regresion para Tendencia de Acciones ", key=f"reg", use_container_width=True)
    st.write('###')
    st.warning("Cabe la posibilidad que al hacer la petición de los valores falle y de un mensaje de error, Por Favor refrescar la página y volver a solicitar la información. Además si en algun mercado (de EEUU o Argentina es feriado), eso genera un error al momento de generar los calculos entre la cotización activos locales y activos extranjeros")
    #st.write('###')    
    if st.container(border=True):
        st.button("Digitar Parámetros para ejecutar  Modelo de Regresión", key="regdos", use_container_width=True)
        col1, col2,col3  = st.columns(3, gap="small", vertical_alignment="center", border=True)
        # Configuración de ingreso de parametros para el uso del modelo    
        with col1:  
            # Selección del periodo de ticker de la accion  
            ticker = st.text_input("Símbolo de la acción", "AAPL", help="Debe ingresar el símbolo de la acción (Ej. AAPL para Apple)").upper()
        with col2:
            # Selección del periodo de tiempo
            periodo = st.selectbox("Seleccionar periodo", ['1d', '5d', '1mo', '3mo', '6mo', '12mo', '1y', '2y', '5y', '10y'], index=0, help="Debe seleccionar entre periodo de 1dia/10años(Ej. 1mo)")
        with col3:    
            # Selección del intervalo de tiempo
            interval = st.selectbox("Seleccionar intervalo de tiempo", ['1m', '2m', '5m', '15m', '30m', '60m', '1h','1d', '5d', '1wk', '1mo', '3mo', '6mo', '1y'], index=6, help="Debe seleccionar periodo 1minuto/1año(Ej. 30m)")
        st.write("---")
        if st.button("Generar Modelo de Regresión", key="glow-on-reg"):
            
            with st.status("Descargando datos...", expanded=True) as status:
                st.write("Complentando los datos...")
                time.sleep(2)
                st.write("Desplegando modelo de Regresión.")
                time.sleep(1)
                st.write("Completando datos de Regresión...")
                time.sleep(1)
                status.update(
                    label="Regresión completa!", state="complete", expanded=True
            )
                            
                # Calcular los resultados de la regresión para el símbolo seleccionado              
                df_result = categorize_trends(ticker, periodo, interval)
                
                # Página principal        
                st.write("---")
                st.markdown("<h1 style='text-align: center;'>Análisis de Regresión de Acciones</h1>", unsafe_allow_html=True)
                st.markdown("<style>.block-container {padding-top: 0;}</style>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("<h2 style='text-align: center;'>Resultados de la Regresión</h2>", unsafe_allow_html=True)
                    st.write(df_result)

                with col2:
                    if 'Error Message' in df_result:
                        st.error(f"Error: {df_result['Error Message']}")
                    else:
                        st.markdown(f"<h2 style='text-align: center;'>Gráfica de Regresión para {ticker}</h2>", unsafe_allow_html=True)
                        data = get_stock_data(ticker, periodo, interval)
                        X = sm.add_constant(range(len(data)))
                        y = data['Close']
                        model = sm.OLS(y, X).fit()
                        data['Regression'] = model.predict(X)

                        plt.figure(figsize=(10, 6))
                        plt.scatter(data.index, data['Close'], label='Precios de Cierre', alpha=0.6)
                        plt.plot(data.index, data['Regression'], color='red', label='Línea de Regresión')
                        plt.xlabel(f"Perido de {periodo} con intervalo de {interval}")
                        plt.ylabel("Precio de Cierre")
                        plt.title(f"Análisis de Regresión para {ticker}")
                        plt.legend()
                        st.pyplot(plt)                
if __name__ == "__main__":
    main()

st.write("###")
st.write("---")
#---------- desarrollo de graficos para acciones seleccionadas -----------

# Función para validar intervalos
def validate_interval(period, interval):

    valid_intervals = {
        '1d': {'1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '4h'},
        '5d': {'1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d'},
        '1mo': {'1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '5d'},
        '3mo': {'1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '5d'},
        '6mo': {'1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '5d', '1wk'},
        '12mo': {'1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '5d', '1wk', '1mo'},
        '1y': {'1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '5d', '1wk', '1mo'},
        '2y': {'1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '5d', '1wk', '1mo'},
        '5y': {'1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '5d', '1wk', '1mo'},
        '10y': {'1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '5d', '1wk', '1mo'}
    }
    
    return interval in valid_intervals.get(period, set())

#---------- historial de precios de acciones en yahoo finance -----------
def get_historical_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        return data
    except Exception as e:
        raise Exception(f"Error dato de fecha: {e}")

#----------- se limpia la  base de datos ---------------------
def clean_data(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df.interpolate(method='linear', axis=0)
    
    return df

def add_ema(df, periods=[20, 50, 100, 200]):
    for period in periods:
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    return df

def plot_data_with_ema(df):
    # Verificar si hay datos suficientes para graficar
    if df.empty or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        st.error("No hay datos suficientes para graficar. Por favor, verifica el ticker, periodo o intervalo.")
        return

    fig = go.Figure()

    # Añadir el gráfico de velas japonesas
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlesticks'
    ))

    # Añadir las EMAs
    for ema_period in [20, 50, 100, 200]:
        if f'EMA_{ema_period}' in df:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[f'EMA_{ema_period}'],
                mode='lines',
                name=f'EMA {ema_period}'
            ))

    fig.update_layout(
        title=f"Gráfico de {ticker} con Med.Móviles Exponenciales",
        xaxis_title="tiempo",
        yaxis_title="Precio",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

def safe_format(value):
    return f"${value:,.6f}" if isinstance(value, (int, float)) and not np.isnan(value) else "N/A"

st.button("Gráfico con Medias Móviles Exponenciales de Acciones ", key=f"grafico", use_container_width=True)
st.write('###')
st.warning("Recuerde que se hace una petición de datos a Yahoo Finance,y es abre la posibilidad que la petición de los valores falle y de un mensaje de error, Por Favor refrescar la página y volver a solicitar la información.")
st.write('---')    
st.button("Digitar Parámetros para la generación del gráfico", key=f"graficodos", use_container_width=True)
with st.container(border=True):
    
    col1, col2, col3 = st.columns(3, gap='small',vertical_alignment='center')
    with col1:
        ticker = st.text_input("Ticker Symbol", "AAPL").upper()
    with col2:    
        period = st.selectbox("Periodo", options=["1d", "5d", "1mo", "3mo", "6mo", "1y","2y","5y","10y"], index=2)
    with col3:    
        interval = st.selectbox("Interval", options=["1m", "5m", "15m", "30m", "1h", "1d","5d","1wk", "1mo", "3mo", "6mo","1y"], index=5)

st.markdown(
    """
<style>
button {
    font-size: 16px;
    height: auto;
    padding-top: 20px !important;
    padding-bottom: 20px !important;   
}
</style>
""",
    unsafe_allow_html=True,
)

if st.button("hace click para graficar", key='graficar'):
        
        with st.status("Descargando datos...", expanded=True) as status:
            st.write("Completando los datos...")
            time.sleep(2)
            st.write("Desplegando el gráfico.")
            time.sleep(1)
            st.write("Confeccionando gráfico...")
            time.sleep(1)
            status.update(
                label="Gráfico completo!", state="complete", expanded=True
        )


            # Validación de intervalo
            if not validate_interval(period, interval):
                st.error(f"El intervalo {interval} no es válido para el período {period}. Por favor seleccione un intervalo adecuado.")
            else:
                st.markdown("<style> .css-18e3th9 { padding-top: 0; } </style>", unsafe_allow_html=True)

                try:
                    # Obtener datos históricos
                    df = get_historical_data(ticker, period, interval)
                    
                    # Verificar si los datos están vacíos antes de limpiar y añadir EMAs
                    if df.empty:
                        st.error(f"No se encontraron datos para {ticker}, en el {period} o {interval}. Por favor verifica tu entrada.")
                        st.warning("Por favor, ingrese un periodo e intervalo válido....")
                    else:  
                        # Título principal            
                        st.button(f"Gráfico de precios y medias móviles exponenciales de {ticker}", key="medmov", use_container_width=True)

                        df = clean_data(df)
                        df = add_ema(df)            
                    
                        # Mostrar métricas de precio y EMAs
                        current_price = df['Close'].iloc[-1]
                        ema_20 = df['EMA_20'].iloc[-1]
                        ema_50 = df['EMA_50'].iloc[-1]
                        ema_100 = df['EMA_100'].iloc[-1]
                        ema_200 = df['EMA_200'].iloc[-1]

                        # Mostrar métricas en columnas
                        col1, col2, col3, col4, col5 = st.columns(5)

                        with col1:
                            #st.markdown(f"<div style='background-color: #d4edda; padding: 10px; text-align: center; border-radius: 5px;'><h3>Current Price</h3><p style='font-size: 24px; font-weight: bold;'>{safe_format(current_price)}</p></div>", unsafe_allow_html=True)
                            st.write(f"Precio actual de : {ticker}")
                            st.button(f"U$D {current_price:,.2f}", key="precio") 
                        with col2:
                            #st.markdown(f"<div style='background-color: #f8f9fa; padding: 10px; text-align: center; border-radius: 5px;'><h3>EMA 20</h3><p style='font-size: 24px; font-weight: bold;'>{safe_format(ema_20)}</p></div>", unsafe_allow_html=True)
                            st.write(f"Med.Movil.Exp(20 dias)")
                            st.button(f"U$D {ema_20:,.2f}", key="ema20")
                        with col3:
                            #st.markdown(f"<div style='background-color: #f8f9fa; padding: 10px; text-align: center; border-radius: 5px;'><h3>EMA 50</h3><p style='font-size: 24px; font-weight: bold;'>{safe_format(ema_50)}</p></div>", unsafe_allow_html=True)
                            st.write(f"Med.Movil.Exp(50 dias)")
                            st.button(f"U$D {ema_50:,.2f}", key="ema50")
                        with col4:
                            #st.markdown(f"<div style='background-color: #f8f9fa; padding: 10px; text-align: center; border-radius: 5px;'><h3>EMA 100</h3><p style='font-size: 24px; font-weight: bold;'>{safe_format(ema_100)}</p></div>", unsafe_allow_html=True)
                            st.write(f"Med.Movil.Exp(100 dias)")
                            st.button(f"U$D {ema_100:,.2f}", key="ema100")
                        with col5:
                            #st.markdown(f"<div style='background-color: #f8f9fa; padding: 10px; text-align: center; border-radius: 5px;'><h3>EMA 200</h3><p style='font-size: 24px; font-weight: bold;'>{safe_format(ema_200)}</p></div>", unsafe_allow_html=True)
                            st.write(f"Med.Movil.Exp(200 dias)")
                            st.button(f"U$D {ema_200:,.2f}", key="ema200")

                        # Graficar datos con EMAs
                        plot_data_with_ema(df)

                except Exception as e:
                    st.error(f"Error: {e}")
                    
# --------------- footer -----------------------------
st.write("---")
with st.container():
  #st.write("---")
  st.write("&copy; - derechos reservados -  2024 -  Walter Gómez - FullStack Developer - Data Science - Business Intelligence")
  #st.write("##")
  left, right = st.columns(2, gap='medium', vertical_alignment="bottom")
  with left:
    #st.write('##')
    st.link_button("Mi LinkedIn", "https://www.linkedin.com/in/walter-gomez-fullstack-developer-datascience-businessintelligence-finanzas-python/",use_container_width=True)
  with right: 
     #st.write('##') 
    st.link_button("Mi Porfolio", "https://walter-portfolio-animado.netlify.app/", use_container_width=True)
      