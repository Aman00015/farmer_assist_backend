import os
import numpy as np
from PIL import Image
import io
import random

class DiseaseDetector:
    def __init__(self):
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the disease detection model"""
        try:
            # For demo purposes, we'll create a mock model
            self.model = {
                "name": "Plant_Disease_Detector",
                "version": "1.0",
                "supported_crops": ["Tomato", "Potato", "Corn", "Grape", "Apple", "Cherry", "Peach", "Pepper", "Strawberry"]
            }
            self.is_loaded = True
            
            return {"status": "success", "message": "Disease detection model loaded successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def predict_disease(self, image_file):
        """Predict disease from uploaded image"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        try:
            # Read and validate image - use await for async file reading
            image_data = await image_file.read()
            if len(image_data) == 0:
                return {"error": "Empty image file"}
            
            # Reset file pointer for potential reuse
            await image_file.seek(0)
            
            image = Image.open(io.BytesIO(image_data))
            
            # Validate image
            if image.mode not in ['RGB', 'RGBA', 'L']:
                return {"error": f"Unsupported image mode: {image.mode}"}
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get image info
            width, height = image.size
            
            # Mock predictions for demo
            crops_diseases = [
                {"crop": "Tomato", "disease": "Late Blight", "confidence": random.uniform(85, 95)},
                {"crop": "Tomato", "disease": "Early Blight", "confidence": random.uniform(80, 90)},
                {"crop": "Tomato", "disease": "Bacterial Spot", "confidence": random.uniform(75, 85)},
                {"crop": "Potato", "disease": "Late Blight", "confidence": random.uniform(80, 90)},
                {"crop": "Potato", "disease": "Early Blight", "confidence": random.uniform(85, 95)},
                {"crop": "Corn", "disease": "Northern Leaf Blight", "confidence": random.uniform(70, 85)},
                {"crop": "Corn", "disease": "Common Rust", "confidence": random.uniform(75, 88)},
                {"crop": "Grape", "disease": "Black Rot", "confidence": random.uniform(80, 92)},
                {"crop": "Grape", "disease": "Powdery Mildew", "confidence": random.uniform(82, 94)},
                {"crop": "Apple", "disease": "Apple Scab", "confidence": random.uniform(78, 90)},
                {"crop": "Apple", "disease": "Cedar Apple Rust", "confidence": random.uniform(75, 88)},
                {"crop": "Cherry", "disease": "Cherry Leaf Spot", "confidence": random.uniform(70, 85)},
                {"crop": "Peach", "disease": "Bacterial Spot", "confidence": random.uniform(72, 87)},
                {"crop": "Pepper", "disease": "Bacterial Spot", "confidence": random.uniform(76, 89)},
                {"crop": "Strawberry", "disease": "Leaf Spot", "confidence": random.uniform(78, 91)},
                {"crop": "Soybean", "disease": "Bacterial Blight", "confidence": random.uniform(73, 86)},
                {"crop": "Healthy", "disease": "No Disease Detected", "confidence": random.uniform(90, 99)},
            ]
            
            # Pick a random prediction for demo
            prediction = random.choice(crops_diseases)
            
            # Get treatment recommendations
            treatment = self._get_treatment_recommendations(prediction["crop"], prediction["disease"])
            
            return {
                "success": True,
                "prediction": {
                    "crop": prediction["crop"],
                    "disease": prediction["disease"],
                    "confidence": round(prediction["confidence"], 2)
                },
                "treatment": treatment,
                "image_info": {
                    "width": width,
                    "height": height,
                    "format": image.format,
                    "mode": image.mode
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_treatment_recommendations(self, crop, disease):
        """Get treatment recommendations based on crop and disease"""
        recommendations = []
        
        # If healthy, return maintenance tips
        if disease == "No Disease Detected":
            return [
                "Continue current maintenance practices",
                "Monitor plants regularly for early signs of disease",
                "Maintain proper watering schedule",
                "Ensure adequate sunlight and air circulation"
            ]
        
        # General recommendations for all diseases
        general_recommendations = [
            "Remove and destroy infected plant parts",
            "Ensure proper spacing for air circulation",
            "Avoid overhead watering to reduce moisture",
            "Apply appropriate fungicides/pesticides as needed",
            "Maintain proper plant nutrition"
        ]
        
        recommendations.extend(general_recommendations)
        
        # Crop-specific recommendations
        crop_recommendations = {
            "Tomato": [
                "Rotate crops annually (3-4 year rotation)",
                "Use disease-resistant varieties",
                "Apply copper-based fungicides preventively",
                "Stake plants to improve air circulation"
            ],
            "Potato": [
                "Use certified disease-free seed potatoes",
                "Practice crop rotation (3-4 years away from potatoes)",
                "Apply fungicides during flowering",
                "Destroy volunteer potatoes and nightshade weeds"
            ],
            "Corn": [
                "Plant resistant hybrids when available",
                "Apply fungicide at silking stage if disease pressure is high",
                "Remove crop debris after harvest to reduce overwintering pathogens",
                "Avoid continuous corn planting"
            ],
            "Grape": [
                "Prune vines properly for good air circulation",
                "Apply sulfur or copper sprays during growing season",
                "Remove infected leaves and fruits promptly",
                "Use drip irrigation instead of overhead"
            ],
            "Apple": [
                "Apply fungicides during bud break and petal fall",
                "Prune to improve air circulation",
                "Remove fallen leaves and fruit in autumn",
                "Use resistant varieties when possible"
            ]
        }
        
        if crop in crop_recommendations:
            recommendations.extend(crop_recommendations[crop])
        
        # Disease-specific recommendations
        disease_recommendations = {
            "Late Blight": [
                "Apply fungicides containing chlorothalonil, mancozeb, or copper",
                "Remove infected plants immediately to prevent spread",
                "Avoid working in fields when plants are wet",
                "Destroy cull piles and volunteer plants"
            ],
            "Early Blight": [
                "Apply fungicides every 7-10 days during favorable conditions",
                "Mulch soil to prevent soil splashing onto leaves",
                "Remove lower infected leaves to improve air circulation",
                "Use balanced fertilization (avoid excess nitrogen)"
            ],
            "Bacterial Spot": [
                "Use copper-based bactericides preventively",
                "Avoid overhead irrigation",
                "Use pathogen-free seeds and transplants",
                "Rotate with non-host crops for 2-3 years"
            ],
            "Powdery Mildew": [
                "Apply sulfur, potassium bicarbonate, or horticultural oils",
                "Improve air circulation around plants",
                "Avoid excessive nitrogen fertilization",
                "Plant in full sun locations"
            ],
            "Leaf Spot": [
                "Apply copper-based fungicides at first sign of disease",
                "Water plants at the base (not on leaves)",
                "Remove and destroy fallen leaves in autumn",
                "Space plants properly to reduce humidity"
            ]
        }
        
        # Check for any matching disease keywords
        for disease_key, treatment_list in disease_recommendations.items():
            if disease_key.lower() in disease.lower():
                recommendations.extend(treatment_list)
                break
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations

# Global instance
disease_detector = DiseaseDetector()