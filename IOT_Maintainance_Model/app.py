import streamlit as st
import pandas as pd
import pickle

# Load the trained model pipeline
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Set up the app
st.set_page_config(page_title="IoT Equipment Maintenance ML Predictor", layout="centered", page_icon="⚙️")

st.title("⚙️ IoT Maintenance Predictor")
st.write("""
This app predicts whether a machine requires maintenance based on its real-time IoT sensor readings.
Please input the current machine readings below.
""")

st.header("Input Sensor Readings")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=200.0, value=75.0, step=0.1)
    vibration = st.number_input("Vibration Level", min_value=0.0, max_value=200.0, value=45.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    pressure = st.number_input("Pressure (bar)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

with col2:
    energy_consumption = st.number_input("Energy Consumption (kWh)", min_value=0.0, max_value=20.0, value=2.5, step=0.1)
    predicted_remaining_life = st.number_input("Predicted Remaining Life (Days)", min_value=0, max_value=1000, value=200, step=1)
    
    st.write("Current Machine Flags")
    machine_status_option = st.selectbox("Machine Status", options=["0 - Off", "1 - Normal", "2 - Warning"], index=1)
    machine_status = int(machine_status_option.split(" ")[0])
    
    anomaly_flag_option = st.selectbox("Anomaly Flag", options=["0 - No Anomaly", "1 - Anomaly Detected"], index=0)
    anomaly_flag = int(anomaly_flag_option.split(" ")[0])

# Button to predict
if st.button("Predict Maintenance Requirement", type="primary"):
    
    # Create a DataFrame for the single prediction (matching exactly to model training features)
    input_data = pd.DataFrame({
        'temperature': [temperature],
        'vibration': [vibration],
        'humidity': [humidity],
        'pressure': [pressure],
        'energy_consumption': [energy_consumption],
        'machine_status': [machine_status],
        'anomaly_flag': [anomaly_flag],
        'predicted_remaining_life': [predicted_remaining_life]
    })
    
    # Predict
    try:
        prediction = model.predict(input_data)[0]
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.error("🚨 **Maintenance Required!** The model strongly suggests scheduling maintenance for this machine.")
        else:
            st.success("✅ **No Maintenance Required.** The machine is currently operating within expected parameters.")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
