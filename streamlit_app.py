import altair as alt
import pandas as pd
import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wget
import sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Configuration Page
st.set_page_config(
    page_title="Classification titanic", page_icon="🤖", layout="centered"
)

# Titre de l'app
st.title("Classification binaire du titanic - 2024")


st.markdown(
    '<div style="text-align: center;"><img src="https://upload.wikimedia.org/wikipedia/en/1/18/Titanic_%281997_film%29_poster.png" alt="Italian Trulli"></div>',
    unsafe_allow_html=True,
)

st.markdown("")
st.markdown("")



# load the saved model
model = joblib.load('model_titanic.joblib')

with st.form("my_form"):

    AGE = st.slider("Age", 0, 2, 95)

    st.markdown("")
    st.markdown("")

    SEX = st.radio("Sexe", ("Homme", "Femme"))

    st.markdown("")
    st.markdown("")

    PCLASS = st.selectbox(
        "Séletionez la classe", ("Première", "Deuxième", "Troisème")
    )


    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")

    st.write(
        "Cette personne avait ",
        AGE,
        "ans,",
        " était un/une",
        SEX,
        "et était dans la",
        PCLASS,
        "classe",
    )

    if SEX == "Homme":
        SEX = 1
    else:
        SEX = 0

    if PCLASS == "Première":
        PCLASS = 1
    elif PCLASS == "Deuxième":
        PCLASS = 2
    else:
        PCLASS = 3

    # if EMBARKED == "C":
    #     EMBARKED = 1
    # elif EMBARKED == "S":
    #     EMBARKED = 2
    # else:
    #     EMBARKED = 3

    # PREDICTIONS 0 ou 1
    pred = model.predict(
        [[PCLASS, SEX, AGE]]
    )

    if pred == 0:
        pred = "mort"
    else:
        pred = "survie"

    st.metric(" ", pred)
    
    proba = model.predict_proba([[PCLASS, SEX, AGE]])

    st.write(f"Probabilité de survie : {proba[0][1]*100:.2f}%")

    st.write(f"Probabilité de décès : {proba[0][0]*100:.2f}%")

    submitted = st.form_submit_button("Prédire")
