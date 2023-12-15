import streamlit as st
import requests

SERVER_URL = 'https://map-model-service-icec-yoangelcruz.cloud.okteto.net/v1/models/map-model:predict'

def make_prediction(inputs):
    predict_request = {'instances': inputs}
    response = requests.post(SERVER_URL, json=predict_request)
    
    if response.status_code == 200:
        prediction = response.json()
        return prediction
    else:
        st.error("Failed to get predictions. Please check your inputs and try again.")
        return None

def main():
    st.title('Predictor de ubicaciones geográficas')

    st.header('Coordenadas para Paris')
    kazakhstan_lat = st.number_input('Ingrese la latitud de Paris:', value=48.0196)
    kazakhstan_lon = st.number_input('Ingrese la longitud de Paris:', value=66.9237)

    st.header('Coordenadas para Los Angeles')
    brasilia_lat = st.number_input('Ingrese la latitud de Los Angeles:', value=-15.7801)
    brasilia_lon = st.number_input('Ingrese la longitud de Los Angeles:', value=-47.9292)

    if st.button('Predecir'):
        inputs = [
            [kazakhstan_lon, kazakhstan_lat],
            [brasilia_lon, brasilia_lat]
        ]
        predictions = make_prediction(inputs)

        if predictions:
            st.write("\nPredicciones para Paris:")
            st.write(predictions['predictions'][0])

            st.write("\nPredicciones para Los Angeles:")
            st.write(predictions['predictions'][1])

if __name__ == '__main__':
    main()
