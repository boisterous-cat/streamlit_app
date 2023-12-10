import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
import numpy as np
import pickle
import gzip
from typing import List, Optional
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler

with gzip.open('data/pickle_model.pkl', 'rb') as ifp:
    MODEL = pickle.load(ifp)


class Item(BaseModel):
    AGE: Optional[float]= None
    GENDER: Optional[bool]= False
    #EDUCATION: Optional[str]= None
    #MARITAL_STATUS: Optional[str]= None
    CHILD_TOTAL: Optional[float]= None
    DEPENDANTS: Optional[float]= None
    SOCSTATUS_WORK_FL: Optional[bool]= None
    SOCSTATUS_PENS_FL: Optional[bool]= None
    PERSONAL_INCOME: Optional[float]= None
    CREDIT_AMOUNT: Optional[float]= None
    CLOSED: Optional[bool]= False
    PROPERTY: Optional[bool]= False


class Items(BaseModel):
    objects: List[Item]


def show_page(df):
    st.header('_Predictions_', divider='rainbow')
    st.subheader('Страница находится в разработке, так как есть проблема с самой моделью и категориальными признаками:')

    # СЧИТЫВАЕМ ВЕС
    age = st.number_input("Введите ваш возраст", min_value=18, max_value=99, value="min")
    gender = st.checkbox("Мужской пол?")
    #EDUCATION = st.selectbox(
    #    'Ваше образование?',
    #    (['Неполное среднее', 'Среднее', 'Среднее специальное', 'Неоконченное высшее', 'Высшее', 'Ученая степень']))
    #MARITAL_STATUS = st.selectbox(
    #    'Ваше семейное положение?',
    #    (['Состою в браке', 'Не состоял в браке', 'Разведен(а)', 'Вдовец/Вдова', 'Гражданский брак']))
    child = st.number_input("Введите количество детей", min_value=0, max_value=20, value="min",step=1)
    dep = st.number_input("Введите количество иждивенцев", min_value=0, max_value=20, value="min",step=1)
    work = st.checkbox("Вы работаете?")
    pens = st.checkbox("Вы на пенсии?")
    income = st.number_input("Введите ваш ежемесячный доход", min_value=0.0, value="min")
    credit = st.number_input("Введите количество кредитов", min_value=0, max_value=20, value="min",step=1)
    closed = st.checkbox("Есть закрытые кредиты?")
    property = st.checkbox("Есть собственность?")


    if (st.button('Рассчитать')):
        item = Item()
        if age !=None:
            item.AGE = float(age)
        else:
            item.AGE =0.0
        item.GENDER = gender
        if child !=None:
            item.CHILD_TOTAL = float(child)
        else:
            item.CHILD_TOTAL=0.0
        if dep!=None:
            item.DEPENDANTS = float(dep)
        else:
            item.DEPENDANTS=0.0
        item.SOCSTATUS_WORK_FL = work
        item.SOCSTATUS_PENS_FL = pens
        if income!=None:
            item.PERSONAL_INCOME = float(income)
        else:
            item.PERSONAL_INCOME=0.0
        if credit!=None:
            item.CREDIT_AMOUNT = float(credit)
        else:
            item.CREDIT_AMOUNT =0
        item.CLOSED = closed
        item.PROPERTY = property

        df_test = pd.DataFrame([item.dict()], columns=item.dict().keys())
        pred = np.exp(MODEL.predict(df_test))

        prediction = pred

        st.text(prediction)
