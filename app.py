import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Food Delivery Prediction",
    page_icon="🍔",
    layout="wide"
)


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("Food_Delivery_Time_Prediction.csv")


@st.cache_resource
def load_models():
    delivery_model = joblib.load("delivery_model.pkl")
    traffic_model = joblib.load("traffic_model.pkl")

    return delivery_model, traffic_model


data = load_data()
delivery_model, traffic_model = load_models()


# ============================================================
# TITLE
# ============================================================

st.title("🍔 Food Delivery Management & Prediction System")

st.write(
    "Machine learning based system for predicting "
    "traffic level and food delivery time."
)
# ============================================================
# DATA ANALYSIS DASHBOARD
# ============================================================

st.header("📊 Data Analysis Dashboard")

st.write(
    "Explore the food delivery dataset using statistical "
    "analysis and data visualization."
)


tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Delivery Time",
    "🚦 Traffic Analysis",
    "📏 Distance Analysis",
    "🔥 Correlation"
])


# ============================================================
# TAB 1 - DELIVERY TIME DISTRIBUTION
# ============================================================

with tab1:

    st.subheader("Distribution of Delivery Time")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        data["Time_taken_min"],
        bins=20
    )

    ax.set_title("Distribution of Food Delivery Time")
    ax.set_xlabel("Delivery Time (minutes)")
    ax.set_ylabel("Number of Orders")

    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average",
            f"{data['Time_taken_min'].mean():.1f} min"
        )

    with col2:
        st.metric(
            "Median",
            f"{data['Time_taken_min'].median():.0f} min"
        )

    with col3:
        st.metric(
            "Maximum",
            f"{data['Time_taken_min'].max():.0f} min"
        )


# ============================================================
# TAB 2 - TRAFFIC ANALYSIS
# ============================================================

with tab2:

    st.subheader("Traffic Level vs Delivery Time")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=data,
        x="Traffic_Level",
        y="Time_taken_min",
        ax=ax
    )

    ax.set_title("Traffic Level vs Delivery Time")
    ax.set_xlabel("Traffic Level")
    ax.set_ylabel("Delivery Time (minutes)")

    st.pyplot(fig)

    st.write(
        "The chart shows how delivery time varies across "
        "different traffic levels."
    )


# ============================================================
# TAB 3 - DISTANCE ANALYSIS
# ============================================================

with tab3:

    st.subheader("Road Distance vs Delivery Time")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        data["Road_Distance_km"],
        data["Time_taken_min"],
        alpha=0.3
    )

    ax.set_title(
        "Road Distance vs Delivery Time"
    )

    ax.set_xlabel(
        "Road Distance (km)"
    )

    ax.set_ylabel(
        "Delivery Time (minutes)"
    )

    st.pyplot(fig)

    st.write(
        "Longer delivery distances generally show "
        "higher delivery times."
    )


# ============================================================
# TAB 4 - CORRELATION
# ============================================================

with tab4:

    st.subheader("Correlation Heatmap")

    numeric_data = data.select_dtypes(
        include="number"
    )

    correlation = numeric_data.corr()

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title(
        "Correlation Between Numerical Variables"
    )

    st.pyplot(fig)

    st.write(
        "Correlation indicates the strength and direction "
        "of a linear relationship between numerical variables."
    )
# ============================================================
# MACHINE LEARNING MODEL PERFORMANCE
# ============================================================

st.header("🤖 Machine Learning Model Performance")

st.write(
    "Performance of the machine learning models evaluated "
    "on the test dataset."
)


# ============================================================
# DELIVERY TIME MODEL
# ============================================================

st.subheader("🚚 Delivery Time Prediction Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        "2.62 min"
    )

with col2:
    st.metric(
        "RMSE",
        "3.38 min"
    )

with col3:
    st.metric(
        "R² Score",
        "0.991"
    )

st.info(
    "MAE represents the average absolute prediction error. "
    "RMSE gives more weight to larger errors. "
    "R² indicates how well the model explains variation "
    "in delivery time."
)


# ============================================================
# TRAFFIC MODEL
# ============================================================

st.subheader("🚦 Traffic Level Prediction Model")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Test Accuracy",
        "96.0%"
    )

with col2:
    st.metric(
        "Model",
        "Random Forest"
    )

st.info(
    "The traffic model classifies orders into Low, Moderate, "
    "High, and Severe traffic levels."
)


# ============================================================
# MODEL SUMMARY
# ============================================================

st.subheader("📋 Model Summary")

model_summary = pd.DataFrame({
    "Prediction": [
        "Delivery Time",
        "Traffic Level"
    ],
    "Algorithm": [
        "Random Forest Regressor",
        "Random Forest Classifier"
    ],
    "Evaluation Metric": [
        "MAE / RMSE / R²",
        "Accuracy"
    ],
    "Test Performance": [
        "MAE: 2.61 min | R²: 0.991",
        "Accuracy: 96.0%"
    ]
})

st.dataframe(
    model_summary,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ============================================================
# ORDER INPUT
# ============================================================

st.divider()

# ============================================================
# PROJECT INFORMATION
# ============================================================

st.header("📘 Project Information")

st.write(
    "Food Delivery Management and Delivery Time Prediction "
    "is a Python-based data science and machine learning project "
    "that analyzes food delivery orders and predicts traffic "
    "level and estimated delivery time."
)


# ============================================================
# PROJECT OBJECTIVE
# ============================================================

st.subheader("🎯 Project Objective")

st.write(
    "The main objective of this project is to analyze food "
    "delivery data, identify important factors affecting delivery "
    "time, predict traffic conditions, and estimate the time "
    "required to complete a food delivery order."
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.subheader("📂 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Records",
        "50,000"
    )

with col2:
    st.metric(
        "Total Features",
        "24"
    )

with col3:
    st.metric(
        "Target Predictions",
        "2"
    )

st.write(
    "The project uses a Kaggle food delivery dataset containing "
    "50,000 delivery orders and 24 features. The dataset contains "
    "information related to orders, riders, restaurants, weather, "
    "distance, traffic, preparation time, and delivery time."
)


# ============================================================
# TECHNOLOGIES
# ============================================================

st.subheader("🛠️ Technologies Used")

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:
    st.markdown(
        """
        **Programming**
        
        🐍 Python
        
        📊 Pandas
        
        🔢 NumPy
        """
    )

with tech_col2:
    st.markdown(
        """
        **Data Visualization**
        
        📈 Matplotlib
        
        📊 Seaborn
        """
    )

with tech_col3:
    st.markdown(
        """
        **Machine Learning**
        
        🤖 Scikit-learn
        
        🌐 Streamlit
        """
    )


# ============================================================
# MACHINE LEARNING
# ============================================================

st.subheader("🤖 Machine Learning Approach")

st.write(
    "Two Random Forest models are used in this project."
)

st.markdown(
    """
    **1. Traffic Level Prediction**
    
    A Random Forest Classifier predicts the traffic level as:
    Low, Moderate, High, or Severe.
    
    **2. Delivery Time Prediction**
    
    A Random Forest Regressor estimates the delivery time in minutes.
    """
)


# ============================================================
# PROJECT WORKFLOW
# ============================================================

st.subheader("🔄 Project Workflow")

st.code(
    """
Kaggle Dataset
      ↓
Data Loading
      ↓
Data Inspection & Cleaning
      ↓
Exploratory Data Analysis
      ↓
Data Visualization
      ↓
Feature Preparation
      ↓
Machine Learning
      ↓
Traffic Prediction
      ↓
Delivery Time Prediction
      ↓
Streamlit Web Application
""",
    language="text"
)


# ============================================================
# LIMITATIONS
# ============================================================

st.subheader("⚠️ Project Limitations")

st.write(
    "The predictions depend on the quality and characteristics "
    "of the Kaggle dataset. Model performance on this dataset "
    "does not guarantee the same performance on real-world "
    "delivery data."
)

st.write(
    "The application is developed as an academic data science "
    "project and should be considered a prediction prototype "
    "rather than a production food delivery system."
)


st.divider()


# ============================================================
# ORDER INPUT
# ============================================================
# ============================================================
# ORDER INPUT
# ============================================================

st.divider()


# ============================================================
# PROJECT STATISTICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Orders", "50,000")

with col2:
    st.metric("Dataset Features", "24")

with col3:
    st.metric("ML Models", "2")

with col4:
    st.metric("Traffic Accuracy", "96.0%")


st.divider()


# ============================================================
# ORDER INPUT
# ============================================================

st.header("📋 Enter Order Information")

st.write(
    "Enter the details of the order below. "
    "The system will first predict traffic level and "
    "then estimate delivery time."
)


# ============================================================
# BASIC ORDER INFORMATION
# ============================================================

st.subheader("🕐 Order Information")

col1, col2, col3 = st.columns(3)

with col1:
    order_hour = st.slider(
        "Order Hour",
        min_value=0,
        max_value=23,
        value=13
    )

with col2:
    is_weekend = st.checkbox(
        "Weekend",
        value=False
    )

with col3:
    is_festival = st.checkbox(
        "Festival",
        value=False
    )


# ============================================================
# LOCATION AND DELIVERY INFORMATION
# ============================================================

st.subheader("📍 Delivery Information")

col1, col2, col3 = st.columns(3)

with col1:
    weather = st.selectbox(
        "Weather",
        sorted(data["Weather"].unique())
    )

with col2:
    pickup_zone = st.selectbox(
        "Pickup Zone",
        sorted(data["Pickup_Zone"].unique())
    )

with col3:
    dropoff_zone = st.selectbox(
        "Drop-off Zone",
        sorted(data["Dropoff_Zone"].unique())
    )


col1, col2, col3 = st.columns(3)

with col1:
    vehicle_type = st.selectbox(
        "Vehicle Type",
        sorted(data["Vehicle_Type"].unique())
    )

with col2:
    distance_category = st.selectbox(
        "Delivery Distance Category",
        sorted(data["Delivery_Distance_Category"].unique())
    )

with col3:
    delivery_priority = st.selectbox(
        "Delivery Priority",
        sorted(data["Delivery_Priority"].unique())
    )


# ============================================================
# RIDER AND RESTAURANT INFORMATION
# ============================================================

st.subheader("👨‍🍳 Rider & Restaurant Information")

col1, col2, col3 = st.columns(3)

with col1:
    rider_experience = st.number_input(
        "Rider Experience (Years)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        step=0.5
    )

with col2:
    rider_rating = st.number_input(
        "Rider Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.5,
        step=0.1
    )

with col3:
    restaurant_rating = st.number_input(
        "Restaurant Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.2,
        step=0.1
    )


col1, col2, col3 = st.columns(3)

with col1:
    cuisine_type = st.selectbox(
        "Cuisine Type",
        sorted(data["Cuisine_Type"].unique())
    )

with col2:
    restaurant_load = st.selectbox(
        "Restaurant Load",
        sorted(data["Restaurant_Load"].unique())
    )

with col3:
    order_items = st.number_input(
        "Number of Items",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )


# ============================================================
# TIME AND ROAD INFORMATION
# ============================================================

st.subheader("🛣️ Time & Road Information")

col1, col2, col3 = st.columns(3)

with col1:
    preparation_time = st.number_input(
        "Preparation Time (Minutes)",
        min_value=1,
        max_value=120,
        value=20,
        step=1
    )

with col2:
    road_distance = st.number_input(
        "Road Distance (km)",
        min_value=0.1,
        max_value=100.0,
        value=8.0,
        step=0.5
    )

with col3:
    number_of_signals = st.number_input(
        "Number of Signals",
        min_value=0,
        max_value=50,
        value=10,
        step=1
    )


average_speed = st.number_input(
    "Average Speed (km/h)",
    min_value=5.0,
    max_value=100.0,
    value=32.0,
    step=1.0
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🚀 Predict Delivery",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # TRAFFIC PREDICTION
    # --------------------------------------------------------

    traffic_input = pd.DataFrame([{
        "Order_Hour": order_hour,
        "Is_Weekend": int(is_weekend),
        "Is_Festival": int(is_festival),
        "Weather": weather,
        "Pickup_Zone": pickup_zone,
        "Dropoff_Zone": dropoff_zone,
        "Vehicle_Type": vehicle_type,
        "Rider_Experience_Years": rider_experience,
        "Rider_Rating": rider_rating,
        "Restaurant_Rating": restaurant_rating,
        "Cuisine_Type": cuisine_type,
        "Order_Items": order_items,
        "Restaurant_Load": restaurant_load,
        "Preparation_Time_Min": preparation_time,
        "Road_Distance_km": road_distance,
        "Delivery_Distance_Category": distance_category,
        "Number_of_Signals": number_of_signals,
        "Delivery_Priority": delivery_priority
    }])


    predicted_traffic = traffic_model.predict(
        traffic_input
    )[0]


    # --------------------------------------------------------
    # DELIVERY TIME PREDICTION
    # --------------------------------------------------------

    delivery_input = pd.DataFrame([{
        "Order_Hour": order_hour,
        "Is_Weekend": int(is_weekend),
        "Is_Festival": int(is_festival),
        "Weather": weather,
        "Vehicle_Type": vehicle_type,
        "Rider_Experience_Years": rider_experience,
        "Rider_Rating": rider_rating,
        "Restaurant_Rating": restaurant_rating,
        "Cuisine_Type": cuisine_type,
        "Order_Items": order_items,
        "Restaurant_Load": restaurant_load,
        "Preparation_Time_Min": preparation_time,
        "Road_Distance_km": road_distance,
        "Delivery_Distance_Category": distance_category,
        "Traffic_Level": predicted_traffic,
        "Number_of_Signals": number_of_signals,
        "Average_Speed_kmph": average_speed,
        "Delivery_Priority": delivery_priority
    }])


    predicted_time = delivery_model.predict(
        delivery_input
    )[0]


    predicted_time = round(predicted_time)


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    st.divider()

    st.header("🎯 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🚦 Traffic Level")

        if predicted_traffic == "Low":
            st.success(f"Traffic: {predicted_traffic}")

        elif predicted_traffic == "Moderate":
            st.warning(f"Traffic: {predicted_traffic}")

        elif predicted_traffic == "High":
            st.warning(f"Traffic: {predicted_traffic}")

        else:
            st.error(f"Traffic: {predicted_traffic}")


    with col2:

        st.subheader("🚚 Estimated Delivery Time")

        st.success(
            f"{predicted_time} minutes"
        )


    st.info(
        "The traffic level is predicted first. "
        "The predicted traffic level is then used by the "
        "delivery-time model."
    )
