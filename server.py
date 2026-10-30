"""
Airline Passenger Satisfaction — FastAPI Backend
=================================================
Provides real-time passenger satisfaction inference, model benchmarking,
and pre-built sample passenger test points for the React UI.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Initialize App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="NeuralFlight AI — Passenger Satisfaction Prediction API",
    description="High-performance machine learning backend powered by LightGBM, XGBoost, and Random Forest",
    version="1.0.0"
)

# Enable CORS for React frontend (Vite default port 5173 and all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
PROC_DIR = os.path.join(BASE_DIR, "processed")

# ─── Load Scaler Info ─────────────────────────────────────────────────────────
SCALER_FILE = os.path.join(PROC_DIR, "scaler_info.json")
if not os.path.exists(SCALER_FILE):
    raise FileNotFoundError(f"Scaler info not found at {SCALER_FILE}")

with open(SCALER_FILE, "r") as f:
    scaler_data = json.load(f)

scale_cols = scaler_data["scale_cols"]
scaler_mean = dict(zip(scale_cols, scaler_data["scaler_mean_"]))
scaler_scale = dict(zip(scale_cols, scaler_data["scaler_scale_"]))

# ─── Load Machine Learning Models ─────────────────────────────────────────────
models = {}
model_metadata = {
    "LightGBM": {
        "file": "best_model.pkl",
        "accuracy": 96.16,
        "f1_score": 0.9552,
        "roc_auc": 0.9947,
        "description": "Best Overall Model (Light Gradient Boosting Machine)"
    },
    "XGBoost": {
        "file": "xgboost.pkl",
        "accuracy": 95.97,
        "f1_score": 0.9531,
        "roc_auc": 0.9940,
        "description": "Extreme Gradient Boosting with L2 Regularization"
    },
    "Random Forest": {
        "file": "random_forest.pkl",
        "accuracy": 95.17,
        "f1_score": 0.9437,
        "roc_auc": 0.9915,
        "description": "Bagging Ensemble of 250 Decision Trees"
    }
}

for name, meta in model_metadata.items():
    file_path = os.path.join(MODELS_DIR, meta["file"])
    if os.path.exists(file_path):
        try:
            models[name] = joblib.load(file_path)
            print(f"[OK] Loaded {name} from {meta['file']}")
        except Exception as e:
            print(f"[WARN] Failed to load {name}: {e}")
    else:
        print(f"[WARN] Model file {meta['file']} not found on disk")

# Fallback: best_model is LightGBM
if "LightGBM" not in models and os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")):
    models["LightGBM"] = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))

# ─── Pydantic Input Schema ───────────────────────────────────────────────────
class PassengerInput(BaseModel):
    # Demographics
    gender: str = Field(default="Female", description="Male or Female")
    customer_type: str = Field(default="Returning", description="Returning or First-time")
    age: int = Field(default=38, ge=7, le=95, description="Passenger age (7-95)")
    
    # Flight details
    type_of_travel: str = Field(default="Business", description="Business or Personal")
    travel_class: str = Field(default="Business", description="Business, Economy, or Economy Plus")
    flight_distance: float = Field(default=1500.0, ge=30, le=5500, description="Distance in miles")
    departure_delay: float = Field(default=0.0, ge=0, le=1500, description="Departure delay in minutes")
    arrival_delay: float = Field(default=0.0, ge=0, le=1500, description="Arrival delay in minutes")
    
    # Survey Service Ratings (1 to 5 scale)
    online_boarding: int = Field(default=5, ge=1, le=5)
    inflight_wifi: int = Field(default=5, ge=1, le=5)
    seat_comfort: int = Field(default=4, ge=1, le=5)
    legroom_service: int = Field(default=4, ge=1, le=5)
    cleanliness: int = Field(default=4, ge=1, le=5)
    food_and_drink: int = Field(default=3, ge=1, le=5)
    inflight_service: int = Field(default=4, ge=1, le=5)
    inflight_entertainment: int = Field(default=4, ge=1, le=5)
    baggage_handling: int = Field(default=4, ge=1, le=5)
    onboard_service: int = Field(default=4, ge=1, le=5)
    checkin_service: int = Field(default=4, ge=1, le=5)
    gate_location: int = Field(default=3, ge=1, le=5)
    ease_of_booking: int = Field(default=4, ge=1, le=5)
    time_convenience: int = Field(default=3, ge=1, le=5)

    # Optional selected model
    model_name: Optional[str] = Field(default="LightGBM")

# ─── Feature Vector Transformer ──────────────────────────────────────────────
def transform_passenger_to_features(data: PassengerInput) -> pd.DataFrame:
    """Transforms raw passenger inputs into the exact 29 preprocessed features."""
    
    # 1. Total Delay & Log Flight Distance
    total_delay = float(data.departure_delay + data.arrival_delay)
    log_distance = float(np.log1p(data.flight_distance))
    
    # 2. Composite Domain Scores
    ground_score = float(np.mean([
        data.time_convenience,
        data.ease_of_booking,
        data.checkin_service,
        data.gate_location
    ]))
    
    inflight_score = float(np.mean([
        data.online_boarding,
        data.onboard_service,
        data.seat_comfort,
        data.legroom_service,
        data.cleanliness,
        data.food_and_drink,
        data.inflight_service,
        data.inflight_wifi,
        data.inflight_entertainment,
        data.baggage_handling
    ]))
    
    # 3. Standardize Numerical Features
    def scale(col_name: str, val: float) -> float:
        m = scaler_mean.get(col_name, 0.0)
        s = scaler_scale.get(col_name, 1.0)
        return (val - m) / s
    
    # 4. Age Group OHE
    age = data.age
    age_child = (age <= 17)
    age_young_adult = (18 <= age <= 25)
    age_adult = (26 <= age <= 40)
    age_middle = (41 <= age <= 60)
    age_senior = (age > 60)
    
    # 5. Travel Class OHE
    tc = data.travel_class.strip().lower()
    class_business = (tc == "business")
    class_economy = (tc == "economy")
    class_eco_plus = ("plus" in tc)
    
    # 6. Binary Encodings
    gender_val = 1 if data.gender.lower() == "female" else 0
    cust_val = 1 if "return" in data.customer_type.lower() else 0
    travel_type_val = 1 if data.type_of_travel.lower() == "business" else 0
    
    features = {
        "Gender": gender_val,
        "Customer Type": cust_val,
        "Type of Travel": travel_type_val,
        "Departure and Arrival Time Convenience": scale("Departure and Arrival Time Convenience", data.time_convenience),
        "Ease of Online Booking": scale("Ease of Online Booking", data.ease_of_booking),
        "Check-in Service": scale("Check-in Service", data.checkin_service),
        "Online Boarding": scale("Online Boarding", data.online_boarding),
        "Gate Location": scale("Gate Location", data.gate_location),
        "On-board Service": scale("On-board Service", data.onboard_service),
        "Seat Comfort": scale("Seat Comfort", data.seat_comfort),
        "Leg Room Service": scale("Leg Room Service", data.legroom_service),
        "Cleanliness": scale("Cleanliness", data.cleanliness),
        "Food and Drink": scale("Food and Drink", data.food_and_drink),
        "In-flight Service": scale("In-flight Service", data.inflight_service),
        "In-flight Wifi Service": scale("In-flight Wifi Service", data.inflight_wifi),
        "In-flight Entertainment": scale("In-flight Entertainment", data.inflight_entertainment),
        "Baggage Handling": scale("Baggage Handling", data.baggage_handling),
        "Total Delay": scale("Total Delay", total_delay),
        "Log Flight Distance": scale("Log Flight Distance", log_distance),
        "Ground Service Score": scale("Ground Service Score", ground_score),
        "In-flight Service Score": scale("In-flight Service Score", inflight_score),
        "Class_Business": class_business,
        "Class_Economy": class_economy,
        "Class_Economy Plus": class_eco_plus,
        "AgeGrp_Child": age_child,
        "AgeGrp_Young Adult": age_young_adult,
        "AgeGrp_Adult": age_adult,
        "AgeGrp_Middle-Aged": age_middle,
        "AgeGrp_Senior": age_senior
    }
    
    return pd.DataFrame([features])

# ─── API Routes ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "online",
        "service": "NeuralFlight AI - Airline Satisfaction Prediction Service",
        "available_models": list(models.keys()),
        "docs_url": "/docs"
    }

@app.get("/models")
def get_models():
    """Returns available models with their benchmark accuracy and description."""
    return [
        {
            "name": name,
            "accuracy": meta["accuracy"],
            "f1_score": meta["f1_score"],
            "roc_auc": meta["roc_auc"],
            "description": meta["description"],
            "is_loaded": name in models
        }
        for name, meta in model_metadata.items()
    ]

@app.get("/presets")
def get_presets():
    """Returns 3 pre-configured sample passenger test points."""
    return [
        {
            "id": "satisfied_business",
            "title": "VIP Business Traveler (Satisfied)",
            "description": "Business class, seamless boarding, fast in-flight wifi, no delays.",
            "data": {
                "gender": "Female",
                "customer_type": "Returning",
                "age": 42,
                "type_of_travel": "Business",
                "travel_class": "Business",
                "flight_distance": 2100.0,
                "departure_delay": 0.0,
                "arrival_delay": 0.0,
                "online_boarding": 5,
                "inflight_wifi": 5,
                "seat_comfort": 5,
                "legroom_service": 5,
                "cleanliness": 5,
                "food_and_drink": 4,
                "inflight_service": 5,
                "inflight_entertainment": 5,
                "baggage_handling": 5,
                "onboard_service": 5,
                "checkin_service": 5,
                "gate_location": 4,
                "ease_of_booking": 5,
                "time_convenience": 4
            }
        },
        {
            "id": "dissatisfied_economy",
            "title": "Frustrated Economy Passenger (Dissatisfied)",
            "description": "Economy class, 75m departure delay, broken wifi, poor legroom.",
            "data": {
                "gender": "Male",
                "customer_type": "First-time",
                "age": 24,
                "type_of_travel": "Personal",
                "travel_class": "Economy",
                "flight_distance": 850.0,
                "departure_delay": 75.0,
                "arrival_delay": 60.0,
                "online_boarding": 2,
                "inflight_wifi": 1,
                "seat_comfort": 2,
                "legroom_service": 1,
                "cleanliness": 2,
                "food_and_drink": 1,
                "inflight_service": 2,
                "inflight_entertainment": 2,
                "baggage_handling": 2,
                "onboard_service": 2,
                "checkin_service": 2,
                "gate_location": 2,
                "ease_of_booking": 2,
                "time_convenience": 2
            }
        },
        {
            "id": "borderline_traveler",
            "title": "Casual Economy Plus Flyer (Borderline)",
            "description": "Economy Plus, average wifi and services with moderate satisfaction potential.",
            "data": {
                "gender": "Male",
                "customer_type": "Returning",
                "age": 34,
                "type_of_travel": "Business",
                "travel_class": "Economy Plus",
                "flight_distance": 1400.0,
                "departure_delay": 15.0,
                "arrival_delay": 10.0,
                "online_boarding": 3,
                "inflight_wifi": 3,
                "seat_comfort": 3,
                "legroom_service": 3,
                "cleanliness": 4,
                "food_and_drink": 3,
                "inflight_service": 4,
                "inflight_entertainment": 3,
                "baggage_handling": 4,
                "onboard_service": 4,
                "checkin_service": 3,
                "gate_location": 3,
                "ease_of_booking": 3,
                "time_convenience": 3
            }
        }
    ]

@app.post("/predict")
def predict_satisfaction(passenger: PassengerInput):
    """
    Accepts passenger survey data and predicts satisfaction probability
    across the primary model and all available benchmark models.
    """
    if not models:
        raise HTTPException(status_code=500, detail="No models loaded on server")
    
    # 1. Transform raw inputs into 29 features
    df_features = transform_passenger_to_features(passenger)
    
    # 2. Determine target primary model
    chosen_name = passenger.model_name if passenger.model_name in models else "LightGBM"
    primary_model = models.get(chosen_name) or list(models.values())[0]
    
    # 3. Run primary prediction
    pred_raw = int(primary_model.predict(df_features)[0])
    prob_raw = float(primary_model.predict_proba(df_features)[0][1])
    
    is_satisfied = bool(pred_raw == 1)
    confidence = (prob_raw * 100.0) if is_satisfied else ((1.0 - prob_raw) * 100.0)
    
    # 4. Run cross-comparison on all loaded models
    comparison_results = {}
    for name, mdl in models.items():
        p = int(mdl.predict(df_features)[0])
        prob = float(mdl.predict_proba(df_features)[0][1])
        conf = (prob * 100.0) if p == 1 else ((1.0 - prob) * 100.0)
        comparison_results[name] = {
            "prediction": "Satisfied" if p == 1 else "Neutral or Dissatisfied",
            "is_satisfied": (p == 1),
            "confidence_percent": round(conf, 2),
            "probability_satisfied": round(prob * 100.0, 2),
            "benchmark_accuracy": model_metadata.get(name, {}).get("accuracy", 95.0)
        }
    
    # 5. Extract dominant driver insights
    top_drivers = []
    if passenger.online_boarding >= 4:
        top_drivers.append({"factor": "Online Boarding", "sentiment": "positive", "detail": f"High boarding rating ({passenger.online_boarding}/5)"})
    elif passenger.online_boarding <= 2:
        top_drivers.append({"factor": "Online Boarding", "sentiment": "negative", "detail": f"Poor boarding experience ({passenger.online_boarding}/5)"})

    if passenger.inflight_wifi >= 4:
        top_drivers.append({"factor": "In-flight Wifi", "sentiment": "positive", "detail": f"Fast & reliable connection ({passenger.inflight_wifi}/5)"})
    elif passenger.inflight_wifi <= 2:
        top_drivers.append({"factor": "In-flight Wifi", "sentiment": "negative", "detail": f"Sub-standard wifi connectivity ({passenger.inflight_wifi}/5)"})

    if (passenger.departure_delay + passenger.arrival_delay) > 40:
        top_drivers.append({"factor": "Flight Delay", "sentiment": "negative", "detail": f"Significant flight delay ({int(passenger.departure_delay + passenger.arrival_delay)} mins total)"})
    elif (passenger.departure_delay + passenger.arrival_delay) == 0:
        top_drivers.append({"factor": "Punctuality", "sentiment": "positive", "detail": "Punctual departure and arrival"})

    if passenger.travel_class.lower() == "business":
        top_drivers.append({"factor": "Travel Class", "sentiment": "positive", "detail": "Business Class cabin amenities"})
    
    return {
        "status": "success",
        "primary_model": chosen_name,
        "benchmark_accuracy": model_metadata.get(chosen_name, {}).get("accuracy", 96.16),
        "prediction_label": "Satisfied" if is_satisfied else "Neutral or Dissatisfied",
        "is_satisfied": is_satisfied,
        "confidence_percentage": round(confidence, 2),
        "probability_satisfied": round(prob_raw * 100.0, 2),
        "probability_dissatisfied": round((1.0 - prob_raw) * 100.0, 2),
        "key_drivers": top_drivers,
        "model_comparison": comparison_results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
