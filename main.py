import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Prediction App")
st.markdown("Enter passenger details to see if they would have survived:")

# Input form
pclass = st.selectbox("Ticket Class", [1, 2, 3], index=2)
sex = st.selectbox("Sex", ['male', 'female'])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
fare = st.slider("Fare Paid", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

# Predict button
if st.button("Predict Survival"):
    # Create DataFrame with same column names as in training
    input_df = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.success(f"🎉 Survived! (Confidence: {prob:.2f})")
    else:
        st.error(f"💀 Did not survive. (Confidence: {prob:.2f})")
