import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import joblib

class StarClassification:
    def __init__(self):
        self.model = None
        self.scaler = None
    
    def load_model(self, model_path, scaler_path):
        """Load pre-trained model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
    
    def _collect_user_input(self):
        """Collect inputs from the user for star classification"""
        st.markdown("### Star Characteristics")

        # Collecting inputs for the features based on your dataset
        l = st.number_input("Luminosity (L)", min_value=0, max_value=1000, value=1)
        r = st.number_input("Radius (R)", min_value=0.1, max_value=10.0, value=1.0)
        a_m = st.number_input("Absolute Magnitude (A_M)", min_value=-10, max_value=20, value=5)
        color = st.selectbox("Color", ["Blue", "White", "Yellow", "Red", "Orange"])
        spectral_class = st.selectbox("Spectral Class", ["O", "B", "A", "F", "G", "K", "M"])

        # Convert categorical features to numeric if needed
        color_map = {"Blue": 1, "White": 2, "Yellow": 3, "Red": 4, "Orange": 5}
        spectral_class_map = {"O": 1, "B": 2, "A": 3, "F": 4, "G": 5, "K": 6, "M": 7}

        # Return only the features (without 'Temperature' column)
        return pd.DataFrame(
            [
                {
                    "L": l,
                    "R": r,
                    "A_M": a_m,
                    "Color": color_map[color],
                    "Spectral_Class": spectral_class_map[spectral_class],
                    "Type": 0  # Adding the 'Type' column with a dummy value (assuming it was used during training)
                }
            ]
        )
    
    def _preprocess_data(self, input_data):
        """Preprocess input data for model prediction"""
        df = input_data.copy()

        # Ensure the input data has the same columns as the training data (with 'Type')
        expected_columns = ["L", "R", "A_M", "Color", "Spectral_Class", "Type"]

        # Reorder the columns to match the order the model expects
        df = df[expected_columns]

        # Scale the features if a scaler exists
        if self.scaler is not None:
            scaled_data = self.scaler.transform(df)
            df = pd.DataFrame(scaled_data, columns=df.columns)

        return df
    
    def _predict_with_model(self, input_data):
        """Predict star classification using the trained model"""
        processed_data = self._preprocess_data(input_data)

        # Make predictions using the model
        predictions = self.model.predict(processed_data)
        return predictions
    
    def app(self):
        """Run the app"""
        st.title("Star Temperature Prediction")

        # Collect user input
        input_data = self._collect_user_input()

        # Predict with the model
        if st.button("Predict"):
            prediction = self._predict_with_model(input_data)*1000
            st.write(f"Predicted Star Temperature: {prediction[0]} K")

# Initialize the StarClassification app
if __name__ == "__main__":
    model_path = "exported_model/rf/rf_Budget_model.pkl"  # Replace with your actual model file path
    scaler_path = "exported_model/rf/rf_Budget_scaler.pkl"  # Replace with your actual scaler file path

    star_classifier = StarClassification()
    star_classifier.load_model(model_path, scaler_path)
    star_classifier.app()
