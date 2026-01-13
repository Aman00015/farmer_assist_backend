import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

class YieldPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.is_trained = False
        self.df = None
    
    def load_data(self, data_path):
        """Load and prepare data"""
        try:
            self.df = pd.read_csv(data_path)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def train_model(self, data_path, model_path):
        """Train the yield prediction model"""
        try:
            if not self.load_data(data_path):
                return {"status": "error", "message": "Failed to load data"}
            
            # Prepare features and target
            X = self.df[["Crop", "State", "Area", "Fertilizer", "Pesticide"]].copy()
            y = self.df["Yield"].copy()
            
            # Clean data
            for col in ["Crop", "State"]:
                X[col] = X[col].astype(str).str.strip().str.title()
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            
            # Save model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump({
                'model': self.model,
                'label_encoders': self.label_encoders
            }, model_path)
            
            self.is_trained = True
            
            return {
                "status": "success",
                "message": "Model trained successfully",
                "rmse": round(rmse, 2)
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            if not os.path.exists(model_path):
                return False
            
            loaded_data = joblib.load(model_path)
            self.model = loaded_data['model']
            self.label_encoders = loaded_data['label_encoders']
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, input_data):
        """Make yield prediction"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        try:
            # Prepare input
            crop_input = input_data['crop'].strip().title()
            state_input = input_data['state'].strip().title()
            area = float(input_data['area'])
            fertilizer = float(input_data['fertilizer'])
            pesticide = float(input_data['pesticide'])
            
            # Validate inputs
            if area <= 0:
                return {"error": "Area must be positive"}
            
            # Encode categorical variables
            try:
                crop_enc = self.label_encoders["Crop"].transform([crop_input])[0]
                state_enc = self.label_encoders["State"].transform([state_input])[0]
            except ValueError:
                available_crops = list(self.label_encoders["Crop"].classes_)
                available_states = list(self.label_encoders["State"].classes_)
                return {"error": f"Invalid crop or state. Available crops: {available_crops}"}
            
            # Prepare input for prediction
            farmer_input = pd.DataFrame([[crop_enc, state_enc, area, fertilizer, pesticide]],
                                      columns=["Crop", "State", "Area", "Fertilizer", "Pesticide"])
            
            # Make prediction
            predicted_yield = self.model.predict(farmer_input)[0]
            total_production = predicted_yield * area
            
            return {
                "predicted_yield_per_hectare": round(float(predicted_yield), 2),
                "total_production": round(float(total_production), 2),
                "units": "tons"
            }
            
        except Exception as e:
            return {"error": str(e)}

# Global instance
yield_predictor = YieldPredictor()