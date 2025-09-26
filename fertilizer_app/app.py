from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# ----------------- Flask Config -----------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# ----------------- Load & Train Model -----------------
CSV_PATH = "data/fertilizer_dataset.csv"  # keep your dataset inside a data/ folder


def load_data(path):
    df = pd.read_csv(path)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Normalize casing
    df["Soil Type"] = df["Soil Type"].str.strip().str.title()
    df["Crop Type"] = df["Crop Type"].str.strip().str.title()
    df["Fertilizer Name"] = df["Fertilizer Name"].str.strip()

    return df


def build_pipeline(df):
    X = df.drop(columns=["Fertilizer Name"])
    y = df["Fertilizer Name"]

    numeric_cols = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
    cat_cols = ["Soil Type", "Crop Type"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = KNeighborsClassifier(n_neighbors=5, weights="distance")
    pipe = Pipeline(steps=[("prep", pre), ("model", model)])
    pipe.fit(X, y)

    return pipe


# Load dataset + model
df = load_data(CSV_PATH)
pipeline = build_pipeline(df)

# Dropdown options
SOIL_TYPES = sorted(df["Soil Type"].unique().tolist())
CROP_TYPES = sorted(df["Crop Type"].unique().tolist())

# ----------------- Routes -----------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", soils=SOIL_TYPES, crops=CROP_TYPES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        def to_float(x):
            return float(x) if x and str(x).strip() != "" else np.nan

        payload = pd.DataFrame([{
            "Temparature": to_float(form.get("temperature")),
            "Humidity": to_float(form.get("humidity")),
            "Moisture": to_float(form.get("moisture")),
            "Soil Type": form.get("soil_type", "").strip().title(),
            "Crop Type": form.get("crop_type", "").strip().title(),
            "Nitrogen": to_float(form.get("nitrogen")),
            "Potassium": to_float(form.get("potassium")),
            "Phosphorous": to_float(form.get("phosphorous")),
        }])

        # Validation
        if payload.isna().any().any():
            return render_template(
                "index.html",
                soils=SOIL_TYPES,
                crops=CROP_TYPES,
                error="⚠️ Please fill all fields with valid numbers.",
            )

        pred = pipeline.predict(payload)[0]
        proba = max(pipeline.predict_proba(payload)[0])
        confidence = round(float(proba) * 100, 1)

        return render_template(
            "index.html",
            soils=SOIL_TYPES,
            crops=CROP_TYPES,
            result=pred,
            confidence=confidence
        )

    except Exception as e:
        return render_template(
            "index.html",
            soils=SOIL_TYPES,
            crops=CROP_TYPES,
            error=f"Error: {str(e)}"
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API for predictions"""
    data = request.get_json(force=True)
    df_in = pd.DataFrame([{
        "Temparature": float(data["temperature"]),
        "Humidity": float(data["humidity"]),
        "Moisture": float(data["moisture"]),
        "Soil Type": str(data["soil_type"]).title(),
        "Crop Type": str(data["crop_type"]).title(),
        "Nitrogen": float(data["nitrogen"]),
        "Potassium": float(data["potassium"]),
        "Phosphorous": float(data["phosphorous"]),
    }])

    pred = pipeline.predict(df_in)[0]
    proba = max(pipeline.predict_proba(df_in)[0])
    return jsonify({
        "fertilizer": pred,
        "confidence": round(float(proba) * 100, 1)
    })


# ----------------- Run App -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
