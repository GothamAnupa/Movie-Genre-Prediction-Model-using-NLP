# genre_predictor_app.py
import streamlit as st
import pickle
import pandas as pd

# Load trained model and MultiLabelBinarizer
model = pickle.load(open("genre_model.pkl", "rb"))
mlb = pickle.load(open("genre_mlb.pkl", "rb"))

st.title("🎬 Movie Genre Predictor (NLP)")

# Input fields
title = st.text_input("Movie Title", "")
description = st.text_area("Movie Description", "")
hero = st.text_input("Lead Actor", "")

if st.button("Predict Genre"):
    if title and description and hero:
        text = f"{title} {description} {hero}"
        prediction = model.predict([text])
        genres = mlb.inverse_transform(prediction)
        
        if genres[0]:
            st.success(f"🎯 Predicted Genres: {', '.join(genres[0])}")
        else:
            st.warning("Couldn't confidently predict any genres.")
    else:
        st.warning("Please fill in all fields.")
