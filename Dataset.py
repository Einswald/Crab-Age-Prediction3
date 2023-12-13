import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import os
import numpy as np
import altair as alt

data = pd.read_csv('CrabAgePrediction.csv')

X = data[['Length', 'Diameter', 'Height', 'Weight']]
y = data['Age']

page_names = {
    "Dataset": "dataset",
    "Visualization": "visualization",
    "Prediction": "prediction",
}

# Create sidebar with navigation links
st.sidebar.markdown("""<div style="text-align: center;"><h1>Main Menu</h1></div>""", unsafe_allow_html=True)

# Display content based on the selected page
page = st.sidebar.radio("Pilih Menu", list(page_names.keys()))

# Dataset page
if page == "Dataset":
    st.markdown("""<div style="text-align: center;"><h1>Data Crab</h1></div>""", unsafe_allow_html=True)
    data = pd.read_csv('CrabAgePrediction.csv')

    st.dataframe(data)
    st.markdown(data.info())

# Visualization page
elif page == "Visualization":
    st.markdown("""<div style="text-align: center;"><h1>Visualization Data</h1></div>""", unsafe_allow_html=True)
    data = pd.read_csv('CrabAgePrediction.csv')

    # Graph Bar
    st.write("Graph Bar Visualization")
    plt.bar(data['Age'].value_counts().index, data['Age'].value_counts())
    plt.title("Distribusi Umur Plot Kepiting")
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())

    # Scatterplot
    st.write("Scatterplot Visualizaation")
    chart = alt.Chart(data).mark_point().encode(
        x="Length",
        y="Weight",
        color="Age",
    )
    st.altair_chart(chart)

# Prediction page
elif page == "Prediction":
    st.markdown("""<div style="text-align: center;"><h1>Crab Age Prediction</h1></div>""", unsafe_allow_html=True)
    Length = st.number_input("Length : ", value=None, placeholder="In Feet")
    Diameter = st.number_input("Diameter : ", value=None, placeholder="In Feet")
    Height = st.number_input("Height : ", value=None, placeholder="In Feet")
    Weight = st.number_input("Weight: ", value=None, placeholder="In Ounce")

    if st.button('Prediksi'):
        new_data = pd.DataFrame({'Length': [Length], 'Diameter': [Diameter], 'Height': [Height], 'Weight': Weight})
        model = DecisionTreeRegressor(random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model.fit(X_train, y_train)
        predicted_cc = model.predict(new_data)

        # Visualisasi prediksi
        st.markdown(f'Umur Kepiting : {predicted_cc} bulan')

        # Tampilkan scatterplot dengan titik yang menunjukkan prediksi umur kepiting
        chart = alt.Chart(data).mark_point().encode(
            x="Length",
            y="Weight",
            color="Age",
            size="Age",
        )
        chart = chart.transform_calculate(
            "Age",
            f"{predicted_cc}",
        )
        st.altair_chart(chart.mark_circle(size=60).encode(
            x="Length",
            y="Weight",
            color="Age",
        ))

