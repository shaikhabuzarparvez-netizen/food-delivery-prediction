import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score
)


# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv("Food_Delivery_Time_Prediction.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ============================================================
# FUNCTION TO CREATE PREPROCESSOR
# ============================================================

def create_preprocessor(X):

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# 1. DELIVERY TIME PREDICTION MODEL
# ============================================================

print("\nTraining Delivery Time Prediction Model...")

delivery_features = [
    "Order_Hour",
    "Is_Weekend",
    "Is_Festival",
    "Weather",
    "Vehicle_Type",
    "Rider_Experience_Years",
    "Rider_Rating",
    "Restaurant_Rating",
    "Cuisine_Type",
    "Order_Items",
    "Restaurant_Load",
    "Preparation_Time_Min",
    "Road_Distance_km",
    "Delivery_Distance_Category",
    "Traffic_Level",
    "Number_of_Signals",
    "Average_Speed_kmph",
    "Delivery_Priority"
]

X_delivery = data[delivery_features]
y_delivery = data["Time_taken_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X_delivery,
    y_delivery,
    test_size=0.20,
    random_state=42
)

delivery_preprocessor = create_preprocessor(X_delivery)

delivery_model = Pipeline(
    steps=[
        ("preprocessor", delivery_preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=60,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

delivery_model.fit(X_train, y_train)

delivery_predictions = delivery_model.predict(X_test)

delivery_mae = mean_absolute_error(
    y_test,
    delivery_predictions
)

delivery_rmse = mean_squared_error(
    y_test,
    delivery_predictions
) ** 0.5

delivery_r2 = r2_score(
    y_test,
    delivery_predictions
)

print("\nDelivery Time Model Results:")
print("MAE :", round(delivery_mae, 2), "minutes")
print("RMSE:", round(delivery_rmse, 2), "minutes")
print("R2  :", round(delivery_r2, 3))


joblib.dump(
    delivery_model,
    "delivery_model.pkl",
    compress=3
)

print("Delivery model saved as delivery_model.pkl")


# ============================================================
# 2. TRAFFIC LEVEL PREDICTION MODEL
# ============================================================

print("\nTraining Traffic Level Prediction Model...")

traffic_features = [
    "Order_Hour",
    "Is_Weekend",
    "Is_Festival",
    "Weather",
    "Pickup_Zone",
    "Dropoff_Zone",
    "Vehicle_Type",
    "Rider_Experience_Years",
    "Rider_Rating",
    "Restaurant_Rating",
    "Cuisine_Type",
    "Order_Items",
    "Restaurant_Load",
    "Preparation_Time_Min",
    "Road_Distance_km",
    "Delivery_Distance_Category",
    "Number_of_Signals",
    "Delivery_Priority"
]

X_traffic = data[traffic_features]
y_traffic = data["Traffic_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X_traffic,
    y_traffic,
    test_size=0.20,
    random_state=42,
    stratify=y_traffic
)

traffic_preprocessor = create_preprocessor(X_traffic)

traffic_model = Pipeline(
    steps=[
        ("preprocessor", traffic_preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=60,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
        )
    ]
)

traffic_model.fit(X_train, y_train)

traffic_predictions = traffic_model.predict(X_test)

traffic_accuracy = accuracy_score(
    y_test,
    traffic_predictions
)

print("\nTraffic Level Model Results:")
print(
    "Accuracy:",
    round(traffic_accuracy * 100, 2),
    "%"
)


joblib.dump(
    traffic_model,
    "traffic_model.pkl",
    compress=3
)

print("Traffic model saved as traffic_model.pkl")


# ============================================================
# FINISHED
# ============================================================

print("\n======================================")
print("BOTH MODELS TRAINED SUCCESSFULLY!")
print("======================================")