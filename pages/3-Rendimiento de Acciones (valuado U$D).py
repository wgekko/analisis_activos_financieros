import streamlit as st
import pandas as pd
import base64
import yfinance as yf
import base64
import streamlit.components.v1 as components
import time

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
# Suprimir advertencias ValueWarning
warnings.simplefilter("ignore")

theme_plotly = None

#-------------- logo de la pagina -----------------

st.set_page_config(page_title="Analisis de Acciones(U$D)", page_icon="img/market.png", layout="wide")


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")
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


# ----------- Mostrar mensaje de preparación de datos -----------
st.subheader('Métrica de Valor Porcentual de Rendimiento Histórico U$D')
st.warning('Este proceso se demora varios segundos, es posible que falle el requerimiento de cotizaciones de las acciones a la página de Yahoo Finance, lo cual hace que de un mensaje de error, en ese caso por favor actualizar la página, otra posibilidad es que sea feria bursátil en EEUU o Argentina, lo cual el listado se genera con valor vacios')
st.write('---')

# ----------------- Parámetros del análisis -------------------
if st.container(border=True):
    col1,col2,col3 = st.columns(3, gap="small", vertical_alignment="center", border=True)
    with col1:
        anios = int(st.number_input("Digite el número de años de datos historico :",value=10, min_value=1, max_value=10,help="valores entre 1/10",placeholder="ingrese valor"))
        #anios = 10
    with col2:    
        ruedas = int(st.number_input("Digite el número de ruedas bursátiales :",value=100,min_value=50, max_value=300,help="valores entre 50/300"))
        #ruedas = 100  # Ruedas para filtrar por volumen operado
    with col3:    
        cantidadAcciones = int(st.number_input("Digite el número de acciones a listar :",value=30,min_value=10, max_value=40,help="valores entre 10/40"))
        #cantidadAcciones = 30  # Acciones con mayor volumen en las últimas ruedas
st.write('---')
# ------------------ Mensaje de finalización del proceso ------------------
st.write("---")  
# Descripción del modelo
st.markdown(f"""
    Este modelo toma como referencia el tipo de cambio que surge de la acción YPF 
    entre el mercado local y de EE.UU., para poder establecer una métrica 
    para valuar las {cantidadAcciones} acciones locales y medir su evolución en dólares desde hace 
    {anios} año/s hasta la fecha de hoy. Además, se pondera el volumen operado de las 
    últimas {ruedas} bursátiles seleccionadas.
    """)


columns = st.columns((2, 1, 3))
if columns[1].button("Generar Lista", key="predecir", use_container_width=True):#, use_container_width=True):


    # ----------- Datos y cálculo del CCL -----------
    adr= pd.DataFrame()
    local=pd.DataFrame()
    # Descargar datos de YPF y el mercado local (YPFD.BA) para calcular el CCL
    adr = yf.download("YPF", period=f"{anios}y", interval="1d")['Close'].astype(float)
    local = yf.download("YPFD.BA", period=f"{anios}y", interval="1d")['Close'].astype(float)
    # Convertir la columna 'Date' a datetime en ambos DataFrames
    # Asegurarse de que el índice es de tipo datetime en ambos DataFrames
    adr.index = pd.to_datetime(adr.index)
    local.index = pd.to_datetime(local.index)

    # Hacer un merge en las fechas comunes basándonos en el índice
    merged_data = pd.merge(adr, local, left_index=True, right_index=True, how='inner')

    # Seleccionar las columnas de interés
    adr = merged_data[['YPF']]  
    local = merged_data[['YPFD.BA']]  
    ccl = (local['YPFD.BA'] / adr['YPF'])

    # ----------- Tickers de acciones -----------
    tickers = ['AGRO.BA','ALUA.BA','AUSO.BA','BBAR.BA','BHIP.BA','BMA.BA','BOLT.BA','BPAT.BA',
              'BYMA.BA','CADO.BA','CAPX.BA','CARC.BA','CECO2.BA','CELU.BA','CEPU.BA','CGPA2.BA','COME.BA',
              'CRES.BA','CTIO.BA','CVH.BA','DGCU2.BA','EDN.BA','FERR.BA','GAMI.BA','GARO.BA',
              'GCLA.BA','GGAL.BA','GRIM.BA','HARG.BA','HAVA.BA','INVJ.BA','IRS2W.BA',
              'IRSA.BA','LEDE.BA','LOMA.BA','LONG.BA','METR.BA','MIRG.BA','MOLA.BA','MOLI.BA','MORI.BA',
              'MTR.BA','OEST.BA','PAMP.BA','PATA.BA','POLL.BA','REGE.BA','RICH.BA','RIGO.BA','ROSE.BA','SAMI.BA','SEMI.BA','SUPV.BA',
              'TECO2.BA','TGNO4.BA','TGSU2.BA','TRAN.BA','TXAR.BA','VALO.BA','YPFD.BA']

    # Descargar los datos de todas las acciones y calcular el volumen ponderado
    data = yf.download(tickers, period=f"{anios}y", interval="1d")
    vol = (data['Close'] * data['Volume'] / 1000000).rolling(ruedas).mean().tail(1).squeeze()

    # Filtrar las acciones por volumen
    tickers_volumen = list(vol.sort_values(ascending=False).head(cantidadAcciones).index)

    # Descargar solo los datos de las acciones seleccionadas
    data = yf.download(tickers_volumen, period=f"{anios}y", interval="1d")['Close'].abs()

    # Ajustar los precios a la métrica CCL
    dataCCL = data.div(ccl, axis=0)

    # Encontrar máximos y mínimos
    fechasMax = dataCCL.idxmax()
    preciosMax = dataCCL.max()
    fechasMin = dataCCL.idxmin()
    preciosHoy = dataCCL.tail(1).squeeze()

    # Calcular el upside y desde el máximo
    upside = ((preciosMax / preciosHoy - 1) * 100)
    desdeMax = ((preciosHoy / preciosMax - 1) * 100)

    # Crear la tabla con los resultados
    tabla = pd.concat([fechasMax, fechasMin, preciosMax, preciosHoy, desdeMax, upside], axis=1)
    tabla.columns = ['Fecha Px Max', 'Fecha Px Min', 'Px Max', 'Px Hoy', '%DesdeMax', '%HastaMax']
    tabla = tabla.sort_values('%HastaMax', ascending=False).round(2)

    # Formatear las fechas
    tabla['Fecha Px Max'] = pd.to_datetime(tabla['Fecha Px Max']).dt.strftime("%d-%m-%Y")
    tabla['Fecha Px Min'] = pd.to_datetime(tabla['Fecha Px Min']).dt.strftime("%d-%m-%Y")
    
  
   
    #st.write("###")
    st.write("---")
    # Mostrar la tabla con los resultados
    with st.status("Generando Listado de Acciones", expanded=True) as status:
        st.write("Buscando los datos...")
        time.sleep(2)
        st.write("Realizando los Calculos.")
        time.sleep(1)
        st.write("Listado casi finalizado...")
        time.sleep(1)
        status.update(
            label="Litado completo ....!", state="complete", expanded=True
        )
        with st.container(border=True):
          st.table(tabla)


# ---- CONTACT FOOTER----
#st.write("##")
st.write("---")
if st.container(border=True):
  st.write("&copy; - derechos reservados -  2024 -  Walter Gómez - FullStack Developer - Data Science - Business Intelligence")
  #st.write("##")
  left, right = st.columns(2, gap='medium', vertical_alignment="center")
  with left:
        url="https://www.linkedin.com/in/walter-gomez-fullstack-developer-datascience-businessintelligence-finanzas-python/"            
        st.link_button("Mi LinkedIn", url, use_container_width= True)
  with right: 
        url1= "https://walter-portfolio-animado.netlify.app/"      
        st.link_button("Mi Portfolio", url1, use_container_width= True)