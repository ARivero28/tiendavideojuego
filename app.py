import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado
modelo = joblib.load('modelo-reg-tree-knn-nn.pkl')

st.title('🎮 Predicción de Éxito de Videojuego')

# Entradas del usuario
plataforma = st.selectbox('Plataforma', ['PC', 'PlayStation', 'Xbox', 'Nintendo'])
genero = st.selectbox('Género', ['Acción', 'Aventura', 'RPG', 'Estrategia', 'Deportes'])
duracion = st.slider('Duración estimada (horas)', 1, 100, 10)

# Crear DataFrame para predicción
entrada = pd.DataFrame({
    'plataforma': [plataforma],
    'genero': [genero],
    'duracion': [duracion]
})

# Predicción
if st.button('Predecir'):
    prediccion = modelo.predict(entrada)
    st.success(f'Resultado de predicción: {prediccion[0]}')
    

