import streamlit as st
import joblib
import numpy as np

# Load your model
model_path = r'C:\Users\Pendyala Pranav\Desktop\IPL_PROJECT\score_predictor_model.pkl'
model = joblib.load(model_path)

st.title('Cricket Score Predictor')

def user_input_features():
    player_form = st.sidebar.slider('Player Form', 0, 100, 50)
    pitch_condition = st.sidebar.slider('Pitch Condition', 0, 100, 50)
    opposition_strength = st.sidebar.slider('Opposition Strength', 0, 100, 50)
    weather_condition = st.sidebar.slider('Weather Condition', 0, 100, 50)
    player_fatigue = st.sidebar.slider('Player Fatigue', 0, 100, 50)
    home_advantage = st.sidebar.selectbox('Home Advantage', [0, 1], index=0)
    match_importance = st.sidebar.slider('Match Importance', 0, 100, 50)
    recent_team_form = st.sidebar.slider('Recent Team Form', 0, 100, 50)

    features = [player_form, pitch_condition, opposition_strength, weather_condition, player_fatigue, home_advantage, match_importance, recent_team_form]
    return np.array(features).reshape(1, -1)

input_data = user_input_features()

if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write('Predicted Score:', prediction[0])
