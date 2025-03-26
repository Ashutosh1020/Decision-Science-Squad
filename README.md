
<div align="center">
<h2> Hello Everyone <img src="https://raw.githubusercontent.com/ABSphreak/ABSphreak/master/gifs/Hi.gif" width="33"></h2>
<h1>✈️ Flight Price Predictor Dashboard</h1>
</div>

<div align="center">
  
### Block 43 - Construct Week Project  
### Project Code: **B43_DA_043_Decision Science Squad**  
### Project Name: **Decision Science Squad**
### 🌐 Streamlit Link : [Click to Open Streamlit App](https://decision-science-squad.streamlit.app/)

</div>

<div align="center">
  <img src="https://github.com/Ashutosh1020/Decision-Science-Squad/blob/main/flight_price_prediction_logo.jpg" width='300'>
</div>

---
<h3 align="left"> About This Project 📖</h3>

The **Flight Price Predictor** is an intelligent web application designed to help travelers estimate airfare costs with machine learning. Born from the frustration of unpredictable flight pricing, this tool analyzes historical trends and key factors to deliver accurate price forecasts.

<h3>About the Dataset:</h3> 
The flight price dataset (processed_flight_prices.csv) contains comprehensive pricing information for domestic flights across major Indian airlines. Each entry represents a specific flight route with associated pricing and operational details.<br>

<h3>Key Attributes:</h3>

✈️ Flight Details:
Airline, Origin, Destination, Class (Economy/Business), Number_of_Stops

⏱️ Timing Metrics:
Flight_Date, Duration_Minutes, Departure_Hour, Arrival_Hour

💰 Pricing Data:
Price_(₹) (Target Variable)

📍 Geographical Coverage:
6 major Indian cities (Delhi, Mumbai, Bangalore, Hyderabad, Chennai, Kolkata)

<h3>Dataset Specifications:</h3>

Rows: 30,000+ flight records

Columns: 12 key features

Time Period: 2022-2023 bookings

Price Range: ₹2,000 - ₹1,24,000

<h3>The project aims to:</h3>

1. **Understand price distribution** across airlines, routes, and travel seasons

2. **Identify pricing trends** based on flight timing (advance booking, day-of-week, peak hours)

3. **Visualize cost drivers** to help travelers optimize booking decisions and budgets


---

## 📂 Repository Structure
```
📂 Decision-Science_Squad
├── 📂 data/
│   ├── processed_flight_prices.csv       # Cleaned flight dataset
│   ├── optimized_flight_model.pkl       # Trained ML model
│   └── flight_features.pkl              # Feature encodings
│
├── 📜 app.py                            # Streamlit application
├── 📜 requirements.txt                  # Dependencies
├── 📜 README.md                         # Project documentation
│
├── 📂 assets/                           # Visual resources
│   ├── app_screenshot.png               # Dashboard preview
│   └── flight_routes_map.png            # Route visualization
│
└── 📂 notebooks/                        # Jupyter notebooks
    ├── data_cleaning.ipynb              # Data preprocessing
    └── model_training.ipynb             # ML development
```

## 🛠️ Installation
To run this project locally, follow these steps:

1️⃣ Clone the repository:
   ```bash
   https://github.com/Ashutosh1020/Decision-Science-Squad.git
   cd Decision-Science_Squad
   ```
2️⃣ Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3️⃣ Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
---

## 🛠️ Technologies Used
- 🐍 **Python** (Pandas, NumPy, Scikit-learn, XGBoost)
- 🎨 **Streamlit** (For web app deployment)
- 📊 **Matplotlib & Seaborn** (For visualization)
- 🌐 **Flask** (Optional API integration)

---

Video Walkthrough of the project :
  [▶️ Click here To watch the video](https://youtu.be/5GXCzPnPYBI)

### ScreenShots 📷

<h4>» Dashboard </h4>
<img src="https://github.com/Ashutosh1020/Decision-Science-Squad/blob/main/Project_ScreenShots/Screenshot%20(3809).png" width="750" height="310" alt="Dashboard">

<h4>» Average Price By Airline </h4>
<img src="https://github.com/Ashutosh1020/Decision-Science-Squad/blob/main/Project_ScreenShots/Screenshot%20(3810).png" width="750" height="310" alt="Filter Data">

<h4>» Flight Price Predictor </h4> 
<img src="https://github.com/Ashutosh1020/Decision-Science-Squad/blob/main/Project_ScreenShots/Screenshot%20(3812).png" width="750" height="310" alt="Filter Data">

<h4>» Prediction Result</h4> 
<img src="https://github.com/Ashutosh1020/Decision-Science-Squad/blob/main/Project_ScreenShots/Screenshot%20(3813).png" width="750" height="310" alt="Filter Data">

<h4>» Some Code Snippet </h4> 
<img src="https://github.com/Ashutosh1020/Decision-Science-Squad/blob/main/Project_ScreenShots/Screenshot%20(3814).png" width="750" height="310" alt="Filter Data">
<img src="https://github.com/Ashutosh1020/Decision-Science-Squad/blob/main/Project_ScreenShots/Screenshot%20(3815).png" width="750" height="310" alt="Filter Data">
