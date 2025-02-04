import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import product
import base64
import streamlit.components.v1 as components

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
# Suprimir advertencias ValueWarning
warnings.simplefilter("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


theme_plotly = None

#-------------- logo de la pagina -----------------
st.set_page_config(page_title="Analisis de CCL", page_icon="img/logo2.png", layout="wide")

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/ccl.css")
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

st.logo(image="img/market.png",size='large')

st.button("Calculo de Probabilidades para Dolar CCL", key="otropulse", use_container_width=True)
st.write('###')
#--------------- generando los datos para el calculo de suba o baja del Dolar CCL--------------
#----------- Prespectivas de suba del CCL------------------------
tickers1 =['CEPU','CEPU.BA','GGAL','GGAL.BA','YPF','YPFD.BA','PAM','PAMP.BA',
           'BBAR', 'BBAR.BA', 'BMA', 'BMA.BA', 'CRESY','CRES.BA', 'IRS','IRSA.BA',
           'SUPV','SUPV.BA', 'TEO', 'TECO2.BA','TGS', 'TGSU2.BA']
dato = yf.download(tickers1, auto_adjust=True, start='2020-01-01')['Close']
# -----acciones energeticas --------------
ccl= dato['YPFD.BA']/dato['YPF']
ccl+= dato['CEPU.BA']/dato['CEPU'] * 10
ccl+= dato['PAMP.BA']/dato['PAM'] * 25
ccl+= dato['TGSU2.BA']/dato['TGS'] * 5
# -----acciones financieras --------------
ccl+= dato['GGAL.BA']/dato['GGAL'] * 10
ccl+= dato['BBAR.BA']/dato['BBAR'] * 3
ccl+= dato['SUPV.BA']/dato['SUPV'] * 5
ccl+= dato['BMA.BA']/dato['BMA'] * 10
# -----acciones otros sectores --------------
ccl+= dato['CRES.BA']/dato['CRESY'] * 10
ccl+= dato['IRSA.BA']/dato['IRS'] * 10
ccl+= dato['TECO2.BA']/dato['CRESY'] * 5

# ----------------- Parámetros del análisis -------------------
if st.container(border=True):    
    col1,col2,col3 = st.columns(3, gap="small", vertical_alignment="center", border=True)
    with col1:
        st.write('---')
    with col2:    
        ruedas = int(st.number_input("Digite el número de ruedas bursátiales :",value=60,min_value=50, max_value=100,help="valores entre 50/100"))
        #ruedas = 100  # Ruedas para filtrar por volumen operado
    with col3:    
        st.write('---')
    
    st.write('---')

st.markdown(
    """
<style>
button {
    height: auto;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

if st.button("Generar Calculo de Probabilidad", key="glow-on-hover-calcular"):#, use_container_width=True):
  ccl/= 11
  #ruedas = 60
  subas_fw = ccl.pct_change(ruedas)*100 
  values = [5,10,15,20,25,30]
  targets = ((1 + np.array(values)/100)*ccl.iloc[-1]).round(2)
  st.container(border=True)
  st.write("---")
  st.subheader('Análisis de Probabilidades de suba en el dólar CCL')
  st.markdown(f'Se calcula para {ruedas} ruedas bursátiles, de las acciones con cotización mercado local')
  st.markdown('BBAR, BMA, CEPU, CRES, GGAL, IRSA, PAMP, SUPV, TECO2, TGSU2, YPFD como referencia y con relación a sus cotizaciones en dólares en el mercado de EEUU')
  with st.container(border=True):    
      for z in range(len(values)):    
          sup_z = len(subas_fw.loc[subas_fw > values[z] ])/len(subas_fw)
          st.markdown(f'Suba mayores a : {values[z]}% - objetivo (${targets[z]}): Probabilidad de :{round(sup_z*100,1)}%')

  st.write("---")
  #----------- Prespectivas de baja del CCL------------------------
  tickers1 =['CEPU','CEPU.BA','GGAL','GGAL.BA','YPF','YPFD.BA','PAM','PAMP.BA',
            'BBAR', 'BBAR.BA', 'BMA', 'BMA.BA', 'CRESY','CRES.BA', 'IRS','IRSA.BA',
            'SUPV','SUPV.BA', 'TEO', 'TECO2.BA','TGS', 'TGSU2.BA']
  ccl_max_h = ccl.cummax()
  ccl_dd = ((ccl/ccl_max_h-1)*100).dropna().rolling(60).mean()

  values = [-10,-15,-20,-25]
  targets = ((1 + np.array(values)/100)*ccl.iloc[-1]).round(2)

  st.subheader('Análisis de Probabilidades de bajas en el dólar CCL')
  with st.container(border=True):
      for z in range(len(values)):   
          sub_z = len(ccl_dd.loc[ccl_dd < values[z] ])/len(ccl_dd)
          st.markdown(f'Baja  mayor a : {-values[z]}% -objetivo  (${targets[z]}) : Probabilidad de {round(sub_z*100,1)} %')

#---------------------- desarrollo de modelo arima-----------------------------
#st.write('###')
st.write('---')    
    # Título principal
st.button("Modelo de Predicción ARIMA para Dolar CCL", key="pulsearima", use_container_width=True)
st.write('###')
st.warning("Cabe la posibilidad que al hacer la petición de los valores falle y de un mensaje de error, Por Favor refrescar la página y volver a solicitar la información")
st.write('###')
# Parámetros para el cálculo del modelo ARIMA
# ----------------- Parámetros del análisis -------------------
if st.container(border=True):
    col1, col2 = st.columns(2, gap="small", vertical_alignment="center", border=True)
    with col1:
        periodo = st.number_input("Días/ruedas Bursátiles para calcular (ej.100 días atrás)", min_value=1, max_value=365, value=100, step=1, help="valor debe oscilar entre 1/365")
    with col2:
        prediction_days = st.number_input("Días/ruedas Bursátiles a Predecir Precio", min_value=1, max_value=30, value=15, step=1, help="valor debe oscilar entre 1/30")
    st.write('---')

if st.button("Generar Modelo Arima", key="glow-on-arima"):
    with st.status("Generando calculos...", expanded=True) as status:
        st.write("Dercagando los datos...")
        time.sleep(2)
        st.write("Ejecutanto modelo Arima.")
        time.sleep(1)
        st.write("Generado el gráfico...")
        time.sleep(1)
        status.update(
            label="Modelo Completado!", state="complete", expanded=True
        )   
    
        ccl = pd.DataFrame(ccl)
        ccl['Date'] = pd.to_datetime(ccl.index)  # Convertir el índice a datetime, si es que 'Date' es el índice
        ccl.rename(columns={0: 'Price'}, inplace=True)
        ccl.reset_index(drop=True, inplace=True)
        # Reordenar para que 'Date' esté como la primera columna
        ccl = ccl[['Date', 'Price']]
        # Aquí suponemos que 'ccl' ya tiene datos cargados

        # Asegúrate de que la columna 'Date' esté en formato datetime
        ccl['Date'] = pd.to_datetime(ccl['Date'], errors='coerce')

        # Eliminar filas con fechas inválidas (NaT) o valores NaN en 'Price'
        ccl = ccl.dropna(subset=['Date', 'Price'])   
        # Agrupar los datos por el periodo seleccionado
        end_date = ccl['Date'].max()
        start_date = end_date - pd.Timedelta(days=periodo)  # Definir el rango de días hacia atrás
        df_data = ccl[ccl['Date'] >= start_date]
        df_data['Price'] = df_data['Price']*0.10   

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
                st.error(f"Error al ajustar ARIMA con parámetros {arima_order}: {e}")
                return float('inf'), None

        results = []
        mse_values = []  # To store the MSE for each model combination
        arima_combinations = []  # Store the combinations of (p, d, q)
        for p, d, q in product(p_values, d_values, q_values):
            arima_order = (p, d, q)
            mse, model_fit = evaluate_arima_model(train['Price'], test['Price'], arima_order)
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
            forecast = best_model.forecast(steps=len(test) + prediction_days)

            # Verificar si forecast tiene elementos y acceder de manera segura
            if isinstance(forecast, pd.Series):
                last_predicted_price = float(forecast.iloc[-1])  # Si forecast es una Serie, usar .iloc[-1]
            else:
                last_predicted_price = float(forecast[-1]) if len(forecast) > 0 else None  # Si forecast es un arreglo de NumPy

            # Último precio de cierre
            latest_close_price = float(df_data['Price'].iloc[-1])

            # Diseño centrado para las métricas            
            col1, col2 = st.columns(2, border=True, vertical_alignment="center")
            st.markdown("<div style='display: flex; justify-content: center; align-items: center; height: 100%;'>", unsafe_allow_html=True)
            with col1:
                st.subheader(f"Precio de Cierre de: Dólar CCL")
                st.button(f" -- U$D {latest_close_price:,.2f}  --", key="inpulsearima")
            with col2:
                st.subheader(f"Precio proyectado a {prediction_days} Día/s")
                st.button(f" -- U$D  {last_predicted_price:,.2f}  --", key="toinpulsearima")
            st.markdown("</div>", unsafe_allow_html=True)

            # Graficar los resultados
            plt.figure(figsize=(14, 4))  # Ajuste de altura para que el gráfico sea más bajo
            plt.plot(df_data['Date'], df_data['Price'], label='Actual', color='blue')
            plt.axvline(x=df_data['Date'].iloc[train_size], color='gray', linestyle='--', label='Train/Test Split')

            # Datos de entrenamiento/prueba y predicciones
            plt.plot(train['Date'], train['Price'], label='Train Data', color='green')
            plt.plot(test['Date'], forecast[:len(test)], label='Test Predictions', color='orange')

            # Predicciones futuras
            future_index = pd.date_range(start=test['Date'].iloc[-1], periods=prediction_days + 1, freq='D')[1:]
            plt.plot(future_index, forecast[len(test):], label=f'{prediction_days}-Day Forecast', color='red')

            plt.title(f'Dólar CCL Predicciones del modelo ARIMA')
            plt.xlabel('Fecha')
            plt.ylabel('Precio (USD)')
            plt.legend()

            with st.container(border=True):
                st.subheader(f"Predicción del Modelo ARIMA para: Dólar CCL")
                st.pyplot(plt)
#---------------------- desarrollo de modelo lstm-----------------------------
#st.write('###')
st.write('---')
st.button("Modelo de Predicción LSTM para Dólar CCL", key="pulselstm", use_container_width=True)
st.write('###')
st.warning("Cabe la posibilidad que al hacer la petición de los valores falle y de un mensaje de error, Por Favor refrescar la página y volver a solicitar la información. Otro dato no menor, mientras mayor es el período mayor es el tiempo de demora en los calculo del modelo, esto puede provocar muchos o varios segundos para el despliegue de la información. Además es importante resalta que a veces puede diferir mucho entre los datos de entrenamiento y los datos de predicción")
#st.write('###')
st.write('---')
if st.container(border=True):
    st.button("Digitar Parámetros para ejecutar modelo LSTM", key="otropulselstm", use_container_width=True)

if st.container(border=True):
    col1, col2 = st.columns(2, gap="small", vertical_alignment="center", border=True)
    with col1:
        periodo_tiempo = st.number_input("Días/ruedas Bursátiles para calcular (ej.100 días atrás)", min_value=100, max_value=365, value=100, step=1, help="Valor debe oscilar entre 1/365")
    with col2:
        prediction_day = st.number_input("Días/ruedas Bursátiles a Predecir Precio", min_value=1, max_value=30, value=15, step=1, help="Valor debe oscilar entre 1/30")
    st.write('---')
    
  
    if st.button("Generar Modelo LSTM ", key="glow-on-lstm"):
    #if st.button("Click para Generar Modelo LSTM ", key="glow-on-lstm"):
        with st.status("Generando Modelo LSTM...", expanded=True) as status:
            st.write("Buscando datos...")
            time.sleep(2)
            st.write("Desplegando Modelo LSTM, puede demorar varios segundos.")
            time.sleep(1)
            st.write("Ingresando Datos...")
            time.sleep(1)
            status.update(
                label="Modelo completo!", state="complete", expanded=True
            )
            
            ccl = pd.DataFrame(ccl)           
            ccl['Date'] = pd.to_datetime(ccl.index)  # Convertir el índice a datetime, si es que 'Date' es el índice
            ccl.rename(columns={0: 'Price'}, inplace=True)
            ccl.reset_index(drop=True, inplace=True)
            # Reordenar para que 'Date' esté como la primera columna
            ccl = ccl[['Date', 'Price']]           
            # Asegúrate de que la columna 'Date' esté en formato datetime
            ccl['Date'] = pd.to_datetime(ccl['Date'], errors='coerce')
            # Eliminar filas con fechas inválidas (NaT) o valores NaN en 'Price'
            ccl = ccl.dropna(subset=['Date', 'Price'])  
            # Separar las columnas de datos: 'Price' para el escalado, 'Date' no debe ser escalado
            prices = ccl[['Price']].values  # solo los precios para el escalado             
            scaler = MinMaxScaler(feature_range=(0, 1))
            #scaled_data = scaler.fit_transform(ccl)
            scaled_prices = scaler.fit_transform(prices)           
            # Reemplazar la columna de precios escalados en el DataFrame original
            ccl['ScaledPrice'] = scaled_prices           
                        
            # Parámetros de la red LSTM   
            train_size = int(len(ccl) * 0.8)    
            train_data = ccl[['ScaledPrice']].values[:train_size]
            test_size = len(ccl) - train_size
            test_data = ccl[['ScaledPrice']].values[train_size:]                  
       
            # Agrupar los datos por el periodo seleccionado
            end_date = ccl['Date'].max()
            start_date = end_date - pd.Timedelta(days=periodo_tiempo+30)  # Periodo de tiempo hacia atrás
            df_data = ccl[ccl['Date'] >= start_date]
            df_data['Price'] = df_data['Price'] * 0.10   

            # Preparar la división de entrenamiento y prueba (80% para entrenamiento, 20% para prueba)
            train_size = int(len(df_data) * 0.8)
            train, test = df_data[:train_size], df_data[train_size:]
            
                # Parámetros de la red LSTM   
            train_size = int(len(df_data) * 0.8)    
            train_data = df_data[:train_size]
            test_data = df_data[train_size:]      
            #test_size = int(len(scaled_data) * 0.2) 
            #test_data = scaled_data[:test_size]
           
            train_data = train[['ScaledPrice']].values
            test_data = test[['ScaledPrice']].values          

            # Función para crear datasets con secuencias
            def create_dataset(data, time_step=1):
                X, y = [], []
                for i in range(len(data) - time_step):
                    X.append(data[i:(i + time_step), 0])
                    y.append(data[i + time_step, 0])
                return np.array(X), np.array(y)

            # Crear los datasets de entrenamiento y prueba
            time_step = 60
            X_train, y_train = create_dataset(train_data, time_step)
            X_test, y_test = create_dataset(train_data, time_step)               

             
            # Reshape de los datos para el modelo LSTM
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            #X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            X_test = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) 
            # Crear y entrenar el modelo LSTM
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=0)

            # Predicción sobre los datos de prueba
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            # Invertir las predicciones al rango original
            train_predictions = scaler.inverse_transform(train_predictions)*0.10
            y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
            test_predictions = scaler.inverse_transform(test_predictions)*0.10
            y_test = scaler.inverse_transform(y_train.reshape(-1, 1))

            # Pronóstico para los días proyectados
            last_60_days = scaled_prices[-time_step:]
            future_input = last_60_days.reshape(1, time_step, 1)
            future_forecast = []
            for _ in range(prediction_day):
                next_pred = model.predict(future_input)[0, 0]
                future_forecast.append(next_pred)
                next_input = np.append(future_input[0, 1:], [[next_pred]], axis=0)
                future_input = next_input.reshape(1, time_step, 1)

            future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))*0.10
            latest_close_price = float(df_data['Price'].iloc[-1])
            last_predicted_price = float(future_forecast[-1])

            # Mostrar los resultados
            col1, col2 = st.columns(2, border=True, vertical_alignment="center")
            with col1:
                st.subheader(f"Precio de Cierre de CCL:")
                st.button(f" -- U$D {latest_close_price:,.2f} --", key="inpulselstm")
            with col2:
                st.subheader(f"Precio proyectado a {prediction_day} Día/s")
                st.button(f" -- U$D {last_predicted_price:,.2f} --", key="toinpulselstm")

            # Graficar los resultados
            plt.figure(figsize=(14, 5))
            plt.plot(df_data['Date'], df_data['Price'], label='Actual', color='blue')
            plt.axvline(x=df_data['Date'].iloc[train_size], color='gray', linestyle='--', label='Train/Test Split')

            # Datos de predicción
            train_range = df_data['Date'][time_step:train_size]
            test_range = df_data['Date'][train_size:train_size + len(test_predictions)]           
                      
            plt.plot(train_range, train_predictions[:len(train_range)], label='Train Predictions', color='green')
            plt.plot(test_range, test_predictions[:len(test_range)], label='Test Predictions', color='orange')
            future_index = pd.date_range(start=df_data['Date'].iloc[-1], periods=prediction_day + 1, freq='D')[1:]
            plt.plot(future_index, future_forecast, label=f'{prediction_day}-Day Forecast', color='red')

            plt.title(f'Predicciones del modelo LSTM')
            plt.xlabel('Fecha')
            plt.ylabel('Precio (USD)')
            plt.legend()
            st.subheader(f"Predicción del Modelo LSTM para el CCL")
            st.pyplot(plt)

#-----------------Modelo de Regresión del Dolar CCL---------------------------------

#------------------desarrollo de tendencia de acción con regresión -------------
def get_stock_data(period, interval):
    tickers1 = ['CEPU','CEPU.BA','GGAL','GGAL.BA','YPF','YPFD.BA','PAM','PAMP.BA',
                'BBAR', 'BBAR.BA', 'BMA', 'BMA.BA', 'CRESY','CRES.BA', 'IRS','IRSA.BA',
                'SUPV','SUPV.BA', 'TEO', 'TECO2.BA','TGS', 'TGSU2.BA']
    
    dato = yf.download(tickers1, auto_adjust=True, start='2020-01-01')['Close']  
    # -----acciones energeticas --------------
    ccl = dato['YPFD.BA']/dato['YPF']
    ccl += dato['CEPU.BA']/dato['CEPU'] * 10
    ccl += dato['PAMP.BA']/dato['PAM'] * 25
    ccl += dato['TGSU2.BA']/dato['TGS'] * 5
    # -----acciones financieras --------------
    ccl += dato['GGAL.BA']/dato['GGAL'] * 10
    ccl += dato['BBAR.BA']/dato['BBAR'] * 3
    ccl += dato['SUPV.BA']/dato['SUPV'] * 5
    ccl += dato['BMA.BA']/dato['BMA'] * 10
    # -----acciones otros sectores --------------
    ccl += dato['CRES.BA']/dato['CRESY'] * 10
    ccl += dato['IRSA.BA']/dato['IRS'] * 10
    ccl += dato['TECO2.BA']/dato['CRESY'] * 5
    ccl /= 11
    
    ccl = pd.DataFrame(ccl)
    ccl['Date'] = pd.to_datetime(ccl.index)  # Convertir el índice a datetime, si es que 'Date' es el índice
    ccl.rename(columns={0: 'Price'}, inplace=True)
    ccl.reset_index(drop=True, inplace=True)
    
    # Reordenar para que 'Date' esté como la primera columna
    ccl = ccl[['Date', 'Price']]
    
    # Aquí aseguramos que 'ccl' tenga la columna 'Date' en formato datetime
    ccl['Date'] = pd.to_datetime(ccl['Date'], errors='coerce')

    # Eliminar filas con fechas inválidas (NaT) o valores NaN en 'Price'
    ccl = ccl.dropna(subset=['Date', 'Price'])
    
    # Agrupar los datos por el periodo seleccionado
    end_date = ccl['Date'].max()

    # Convertir el valor 'period' de string a días numéricos
    period_days_map = {
        '1mo': 30,
        '3mo': 90,
        '6mo': 180,
        '1y': 365,
        '2y': 730,
        '5y': 1825
    }
    
    if period not in period_days_map:
        raise ValueError("El periodo seleccionado no es válido.")
    
    period_days = period_days_map[period]
    start_date = end_date - pd.Timedelta(days=period_days)  # Definir el rango de días hacia atrás
    df_data = ccl[ccl['Date'] >= start_date]
    df_data['Price'] = df_data['Price'] * 0.10  # Ajuste de precio
    #st.write(df_data)
    
    # Filtrar según el intervalo (en días)
    interval_days = {'1d': 1, '5d': 5, '1wk': 7, '1mo': 30, '3mo': 90}
    interval_filter = interval_days.get(interval, 1)
    
    # Asegurarse de crear 'data' como una copia de 'df_data'
    data = df_data.copy()
    data = data[data['Date'].diff().dt.days <= interval_filter].reset_index(drop=True)
    
    return data

# Función para correr la regresión y categorizar las tendencias
def categorize_trends(period, interval):
    df = get_stock_data(period, interval)
    
    # Verificar si no se descargaron datos
    if df.empty:
        return {
            'Coeficiente de Tendencia': None,
            'Categoria': 'Error',
            'R-cuadrado': None,
            'Mensaje Error': 'No se encontraron datos para el intervalo seleccionado.'
        }
    
    df['time_index'] = range(len(df))
    X = sm.add_constant(df['time_index'])
    y = df['Price']
    
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
            'Coeficiente de Tendencia': round(trend, 5),
            'Categoria': category,
            'R-cuadrado': round(model.rsquared, 5)
        }
    except Exception as e:
        return {
            'Coeficiente de Tendencia': None,
            'Categoria': 'Error',
            'R-cuadrado': None,
            'Mensaje Error': str(e)
        }

def main():
    st.write("---")
    st.button("Modelo de Regresión para Tendencia de Precios", key="pulsereg", use_container_width=True)
    st.write('###')
    st.warning("Cabe la posibilidad que al hacer la petición de los valores falle y de un mensaje de error, Por Favor refrescar la página y volver a solicitar la información. Además si en algun mercado (de EEUU o Argentina es feriado), eso genera un error al momento de generar los calculos entre la cotización activos locales y activos extranjeros")
    #st.write('###')
    st.write('---')
    
    if st.container(border=True):
        st.button("Digitar Parámetros para ejecutar Modelo de Regresión", key="pulseregdos", use_container_width=True)
        col1, col2 = st.columns(2, gap="small", vertical_alignment="center", border=True)
        with col1:     
            # Periodo de tiempo
            periodo = st.selectbox("Seleccionar periodo", ['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=0, help="Debe seleccionar entre periodo de 1 mes y 5 años")
        with col2:
            # Intervalo de tiempo
            interval = st.selectbox("Seleccionar intervalo de tiempo", ['1d', '5d', '1wk', '1mo', '3mo'], index=0, help="Debe seleccionar entre 1 día y 3 meses")
        st.write("---")        
        
        if st.button("Generar Modelo de Regresión", key="glow-on-reg"):
            
            with st.status("Descargando datos...", expanded=True) as status:
                st.write("Completando los datos...")
                time.sleep(2)
                st.write("Desplegando modelo de Regresión.")
                time.sleep(1)
                st.write("Completando datos de Regresión...")
                time.sleep(1)
                status.update(
                    label="Regresión completa!", state="complete", expanded=True
                )
                
                # Calcular los resultados de la regresión
                df_result = categorize_trends(periodo, interval)
                
                # Página principal        
                st.write("---")
                st.markdown("<h3 style='text-align: center;'>Análisis de Regresión de Dolar CCL</h3>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("<h3 style='text-align: center;'>Resultados de la Regresión</h3>", unsafe_allow_html=True)
                    st.write(df_result)

                with col2:
                    if 'Mensaje Error' in df_result:
                        st.error(f"Error: {df_result['Mensaje Error']}")
                    else:
                        st.markdown(f"<h3 style='text-align: center;'>Gráfica de Regresión</h3>", unsafe_allow_html=True)
                        data = get_stock_data(periodo, interval)
                        X = sm.add_constant(range(len(data)))
                        y = data['Price']
                        model = sm.OLS(y, X).fit()
                        data['Regression'] = model.predict(X)

                        plt.figure(figsize=(10, 6))
                        plt.scatter(data.index, data['Price'], label='Precios de Cierre', alpha=0.6)
                        plt.plot(data.index, data['Regression'], color='red', label='Línea de Regresión')
                        plt.xlabel(f"Período de {periodo} con intervalo de {interval}")
                        plt.ylabel("Precio de Cierre")
                        plt.title(f"Análisis de Regresión de Dolar CCL")
                        plt.legend()
                        st.pyplot(plt)

if __name__ == "__main__":
    main()

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
      