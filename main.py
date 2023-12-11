import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import json
import seaborn as sns
from PIL import Image
import numpy as np
import pages.visualization as vs
import pages.prediction as pre

# constants
DATA = 'data/clients.csv'

@st.cache_data
def load_data():
    return pd.read_csv(DATA, index_col=0)

def process_main_page():
    show_main_page()

def show_main_page():
    image = Image.open('data/target.png')
    label = Image.open('data/label.png')
    st.set_page_config(
        layout="centered",
        initial_sidebar_state="collapsed",
        page_title="Demo bank targeting",
        page_icon=label,
        menu_items={
            'Report a bug': "mailto:shef.anastasia@gmail.com",
            'About': "#МОВС23 Applied Python. Собираем всю информацию о клиентах в одну таблицу. Представляем EDA. Делаем предсказания."
        }
    )
    st.markdown(
        """
    <style>
        [data-testid="collapsedControl"] {
            display: none
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.write(
        """
        # Вероятность отлика клиента на предложение банка
        Определяем, кто из клиентов положительно откликнется на предложение банка.
        """
    )
    st.image(image)

    # Load data
    #data_load_state = st.text('Loading data...')
    df = load_data()
    #data_load_state.text("Done! (using st.cache_data)")

    tab1,tab2 = st.tabs(["EDA", "Prediction"])

    with tab1:
      vs.show_page(df)
    with tab2:
        pre.show_page(df)


if __name__ == "__main__":
    process_main_page()




