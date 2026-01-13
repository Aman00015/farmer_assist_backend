import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

class CropRecommender:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.is_trained = False
        self.df = None
    
    def load_data(self, data_path):
        """Load and prepare data"""
        try:
            self.df = pd.read_csv(data_path)
            print(f"✅ Data loaded: {self.df.shape[0]} rows")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def train_model(self, data_path, model_path):
        """Train the crop recommendation model"""
        try:
            if not self.load_data(data_path):
                return {"status": "error", "message": "Failed to load data"}
            
            # Prepare features and target
            X = self.df[["State", "Area", "Fertilizer", "Pesticide"]].copy()
            y = self.df["Crop"].copy()
            
            # Clean data
            X['State'] = X['State'].astype(str).str.strip().str.title()
            y = y.astype(str).str.strip().str.title()
            
            # Encode state
            le = LabelEncoder()
            X['State'] = le.fit_transform(X['State'])
            self.label_encoders['State'] = le
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump({
                'model': self.model,
                'label_encoders': self.label_encoders,
                'df': self.df  # Save data for reference
            }, model_path)
            
            self.is_trained = True
            
            return {
                "status": "success",
                "message": "Model trained successfully",
                "accuracy": round(accuracy, 4)
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
            self.df = loaded_data.get('df', None)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def recommend_crops(self, input_data):
        """Make crop recommendations"""
        if not self.is_trained:
            return {"error": "Crop recommendation model not trained"}
        
        try:
            # Prepare input
            state_input = input_data['state'].strip().title()
            area = float(input_data['area'])
            fertilizer = float(input_data['fertilizer'])
            pesticide = float(input_data['pesticide'])
            
            # Validate inputs
            if area <= 0:
                return {"error": "Area must be positive"}
            if fertilizer < 0 or pesticide < 0:
                return {"error": "Fertilizer and pesticide must be non-negative"}
            
            # Encode state
            try:
                state_enc = self.label_encoders["State"].transform([state_input])[0]
            except ValueError:
                available_states = list(self.label_encoders["State"].classes_)
                return {"error": f"Invalid state '{state_input}'. Available: {available_states}"}
            
            # Prepare input for prediction
            farmer_input = pd.DataFrame([[state_enc, area, fertilizer, pesticide]],
                                      columns=["State", "Area", "Fertilizer", "Pesticide"])
            
            # Get predictions
            crop_probabilities = self.model.predict_proba(farmer_input)[0]
            crop_names = self.model.classes_
            
            # Sort by probability and get top 5
            sorted_indices = crop_probabilities.argsort()[::-1]
            recommendations = []
            
            for i in sorted_indices[:5]:
                confidence = float(crop_probabilities[i])
                crop_name = str(crop_names[i])
                
                # Calculate suitability percentage
                suitability = int(confidence * 100)
                
                # Get expected yield
                expected_yield = self._get_expected_yield(crop_name)
                
                # Get cultivation tips
                tips = self._get_crop_tips(crop_name)
                
                recommendations.append({
                    "name": crop_name,
                    "confidence": round(confidence, 3),
                    "suitability": suitability,
                    "expectedYield": expected_yield,
                    "tips": tips
                })
            
            return {
                "success": True,
                "recommendations": recommendations,
                "top_recommendation": recommendations[0] if recommendations else None
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_expected_yield(self, crop_name):
        """Get expected yield based on historical data"""
        try:
            if self.df is not None:
                crop_data = self.df[self.df['Crop'].str.strip().str.title() == crop_name]
                if not crop_data.empty:
                    avg_yield = crop_data['Yield'].mean()
                    return f"{avg_yield:.1f} tons/hectare"
        except:
            pass
        
        # Fallback to default ranges
        yield_ranges = {
            "Rice": "4-6 tons/hectare",
            "Wheat": "3-4 tons/hectare", 
            "Maize": "5-7 tons/hectare",
            "Cotton": "2-3 tons/hectare",
            "Sugarcane": "70-100 tons/hectare",
            "Soybean": "2-3 tons/hectare",
            "Potato": "20-25 tons/hectare",
            "Tomato": "25-30 tons/hectare",
            "Corn": "5-7 tons/hectare",
            "Barley": "2-3 tons/hectare"
        }
        return yield_ranges.get(crop_name, "3-5 tons/hectare")
    
    def _get_crop_tips(self, crop_name):
        """Get cultivation tips for crop"""
        tips = {
            "Rice": "Requires abundant water and warm climate. Ideal for water-rich regions.",
            "Wheat": "Best in cool, dry climates. Requires well-drained fertile soil.",
            "Maize": "Needs moderate rainfall. Suitable for various soil types.",
            "Cotton": "Requires warm climate and well-drained black cotton soil.",
            "Sugarcane": "Needs tropical climate with abundant water supply.",
            "Soybean": "Adaptable to various soils. Good for crop rotation.",
            "Potato": "Grows best in cool climate with well-drained soil.",
            "Tomato": "Requires full sun and well-drained soil. Regular watering needed.",
            "Corn": "Needs deep, fertile soil with good drainage.",
            "Barley": "Grows well in cool climates with moderate rainfall."
        }
        return tips.get(crop_name, "Requires proper irrigation and soil nutrients.")

# Global instance
crop_recommender = CropRecommender()