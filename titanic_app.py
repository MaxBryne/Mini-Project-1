import streamlit as st
import joblib
import pandas as pd

# Load the trained Titanic model
model = joblib.load(open('titanic_model.joblib', 'rb'))

# Title for the app
st.title('Titanic Survival Prediction App')

# User input form
st.sidebar.header('Passenger Details')

# Collecting user input for the features
Pclass = st.sidebar.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
Sex = st.sidebar.selectbox('Gender (0 = Female, 1 = Male)', [0, 1])
Age = st.sidebar.slider('Age', 0, 80, 25)
SibSp = st.sidebar.slider('Number of Siblings/Spouses Aboard', 0, 8, 0)
Parch = st.sidebar.slider('Number of Parents/Children Aboard', 0, 6, 0)
Fare = st.sidebar.slider('Fare ($)', 0.0, 500.0, 50.0)
Embarked = st.sidebar.selectbox('Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)', ['C', 'Q', 'S'])

# Mapping 'Embarked' to numerical values (adjust according to your model's encoding)
embarked_map = {'C': 0, 'Q': 1, 'S': 2}
Embarked = embarked_map[Embarked]

# Calculate Family Size
family_size = SibSp + Parch + 1  # +1 to include the passenger themselves

# Input features as a DataFrame
input_data = pd.DataFrame({
    'Pclass': [Pclass],
    'Sex': [Sex],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Embarked': [Embarked],
    'FamilySize': [family_size]
})

# Displaying user input
st.write("### Passenger Details:")
st.write(input_data)

# Prediction
if st.button('Predict Survival'):
    with st.spinner('Predicting...'):
        prediction = model.predict(input_data)
        result = 'Survived' if prediction[0] == 1 else 'Did Not Survive'
    st.success(f"The passenger is predicted to: **{result}**")