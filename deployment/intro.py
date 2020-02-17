import streamlit as st
import numpy as np 
import pandas as pd
from catboost import *

print("Imports done")
@st.cache
def load_model():
    model = CatBoost()
    model.load_model(fname = "C:\\Users\\Ankur Bhatkalkar\\Documents\\GitHub\\SFCrimePrediction\\models\\final_model")
    return model

st.title("Know what crimes happen in your area!!")
st.subheader("Using San Francisco Open Crime Data this a model designed to predict which crimes\n"+
        "are most likely to happen depending on factors used in the dataset")


