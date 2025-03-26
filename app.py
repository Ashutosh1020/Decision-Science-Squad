import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime
import numpy as np
import os
import gdown

# Page configuration
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
  /* Main app theme */
.main {
    background-color: #DCE2E6;
    color: #333344;
}

/* Headers */
h1, h2, h3 {
    color: #2a4480;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #1a365d, #2563eb);
    padding: 1.2rem;
}


[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: white;
}

/* Logo styling */
.logo {
    text-align: center;
    margin-bottom: 1.5rem;
}

.logo img {
    width: 90px;
    height: auto;
    filter: drop-shadow(0 2px 5px rgba(0, 0, 0, 0.2));
}

/* Cards for metrics */
.metric-card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    margin-bottom: 1.2rem;
    border-left: 4px solid #3b82f6;
}

/* Buttons */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 6px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    border: none;
    transition: all 0.3s;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    box-shadow: 0 4px 8px rgba(37, 99, 235, 0.25);
    transform: translateY(-2px);
}

/* Prediction result */
.prediction-result {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: white;
    padding: 5px 10px;
    border-radius: 12px;
    text-align: center;
    font-size: 1.6rem;
    margin: 1.5rem 0;
    box-shadow: 0 8px 16px rgba(37, 99, 235, 0.2);
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid #d1d5db;
    font-size: 0.85rem;
    color: #6b7280;
}

/* KPI cards */
.kpi-container {
    display: flex;
    flex-wrap: wrap;
    gap: 1.2rem;
    justify-content: space-between;
}

.kpi-card {
    background: linear-gradient(to bottom right, #ffffff, #f9fafb);
    border-radius: 12px;
    padding: 1.2rem;
    flex: 1;
    min-width: 220px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    text-align: center;
    transition: transform 0.3s ease;
    border: 1px solid #e5e7eb;
}

.kpi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
}

.kpi-value {
    font-size: 2rem;
    font-weight: bold;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.kpi-label {
    font-size: 1.1rem;
    color: #4b5563;
    font-weight: 515;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.2rem;
}

.stTabs [data-baseweb="tab"] {
    background-color: #26262396;
    border-radius: 6px 6px 0 0;
    padding: 0.5rem 1rem;
}

.stTabs [aria-selected="true"] {
    background-color: #2563eb;
    color: white;
}

/* Data tables */
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
}

/* Widget labels */
.stSelectbox label, .stNumberInput label {
    color: #2a4480;
    font-weight: 500;
}

/* Streamlit expander */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #2a4480;
}

/* Selection boxes */
.stSelectbox div[data-baseweb="select"] {
    border-radius: 8px;
    border-color: #d1d5db;
}

/* Number inputs */
.stNumberInput div[data-baseweb="input"] {
    border-radius: 8px;
    border-color: #d1d5db;
}

/* Change the font color of 'Select Flight Date' */
.stDateInput label {
    color: black !important;
    font-weight: 500;
}
/* Sidebar select box labels */
section[data-testid="stSidebar"] label {
    color: white !important;
    font-weight: bold;
}
.prediction-result h4 {
        color: white;
    }
[data-testid="stSidebar"] h1 {
    font-size: 18px; /* Adjust size as needed */
}

/* Increase font size for Sidebar radio button labels */
[data-testid="stSidebar"] .stRadio label {
    font-size: 28px !important;  /* Adjust the size as needed */
    font-weight: bold !important;
}
/* Target the label of the time input widget */
    .stTimeInput label {
        color: black !important;
        font-weight: bold !important;
    }


</style>
""", unsafe_allow_html=True)


# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("processed_flights_price.csv")
        # Ensure flight date is in datetime format
        if 'Flight Date' in df.columns:
            df['Flight Date'] = pd.to_datetime(df['Flight Date'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Using sample data instead.")
        # Create sample data
        airlines = ['IndiGo', 'Air India', 'Vistara', 'SpiceJet', 'AirAsia', 'GoAir']
        origins = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
        destinations = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata']
        stops = [0, 1, 2]
        
        # Create a sample dataframe
        sample_data = []
        for _ in range(300):
            origin = np.random.choice(origins)
            # Ensure destination is different from origin
            dest_options = [d for d in destinations if d != origin]
            destination = np.random.choice(dest_options)
            
            airline = np.random.choice(airlines)
            stop = np.random.choice(stops)
            duration = np.random.randint(30, 300)
            price = np.random.randint(2000, 15000)
            flight_class = np.random.choice(['Economy', 'Business'])
            
            # Create a random date in 2023
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            date_str = f"2023-{month:02d}-{day:02d}"
            
            sample_data.append({
                'Airline': airline,
                'Origin': origin,
                'Destination': destination,
                'Number of Stops': stop,
                'Duration (Minutes)': duration,
                'Price (₹)': price,
                'Class': flight_class,
                'Flight Date': pd.to_datetime(date_str)
            })
        
        df = pd.DataFrame(sample_data)
        return df

# Configure Google Drive link
MODEL_URL = "https://drive.google.com/uc?id=1g241oRYF554q1grX4bOfBzlJJo1gGLxq"
MODEL_PATH = "optimized_flight_model.pkl"

# Load model and features
@st.cache_resource
def load_model():
    try:
        # Download model from Google Drive if it doesn't exist
        if not os.path.exists(MODEL_PATH):
            st.info("Downloading model file from Google Drive...")
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        
        # Load local files as before
        model = joblib.load(MODEL_PATH)
        features = joblib.load("flight_features.pkl")
        return model, features
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Main function to run the app
def main():
    # Load data and model
    df = load_data()
    model, feature_columns = load_model()
    

##############################################################################################################################################################

    
    # Navigation in sidebar
    st.sidebar.title("✈️ Flight Price Predictor")
    
    # Add logo to sidebar
    st.sidebar.image("flight_price_prediction_logo.jpg", width=230)
    
    nav = st.sidebar.radio("Navigate", ["Dashboard", "Price Prediction", "Data Exploration", "About"])
    
    if nav == "Dashboard":
        show_dashboard(df)
    elif nav == "Price Prediction":
        show_prediction(df, model, feature_columns)
    elif nav == "Data Exploration":
        show_exploration(df)
    else:
        show_about()

def show_dashboard(df):
    st.title("✈️ Flight Price Predictor Dashboard")
    
    # Add logo to main page
    
###############################################################################################################################################################


    # Calculate KPIs
    total_flights = len(df)
    num_airlines = df['Airline'].nunique()
    avg_price = int(df['Price (₹)'].mean())
    min_price = int(df['Price (₹)'].min())
    max_price = int(df['Price (₹)'].max())
    most_frequent_destination = df['Destination'].mode()[0]
    most_frequent_departure = df['Origin'].mode()[0]
    
    # Display KPIs
    st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
    
    # First row of KPIs
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_flights}</div>
            <div class="kpi-label">🛬 Total Flights</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{num_airlines}</div>
            <div class="kpi-label">🛩️ Total Airlines</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">₹{avg_price:,}</div>
            <div class="kpi-label">💰 Average Ticket Price</div>
        </div>
        """, unsafe_allow_html=True)

    # Add space between rows
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        

    # Second row of KPIs
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">₹{min_price:,}</div>
            <div class="kpi-label">💰 Minimum Ticket Price</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">₹{max_price:,}</div>
            <div class="kpi-label">💰 Maximum Ticket Price</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{most_frequent_departure} → {most_frequent_destination}</div>
            <div class="kpi-label">🔁 Most Frequent Route</div>
        </div>
        """, unsafe_allow_html=True)
    
    
    st.markdown('</div>', unsafe_allow_html=True)


################################################################################################################################################################

        # Interactive charts
    st.header("Flight Price Analysis")
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Price by Airline", "Price by Route", "Price Trends"])
    
    with tab1:
        st.subheader("Average Price by Airline")
        airline_avg = df.groupby('Airline')['Price (₹)'].mean().sort_values(ascending=False).reset_index()
        
        fig = px.bar(
            airline_avg, 
            x='Airline', 
            y='Price (₹)', 
            title="Average Price by Airline",
            color='Price (₹)',
            color_continuous_scale='blues',
            labels={'Price (₹)': 'Average Price (₹)'}
        )
        
        fig.update_layout(
            xaxis_title="Airline",
            yaxis_title="Average Price (₹)",
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Insight:** Compare the average prices across different airlines to find the most economical options for your travel.
        """)
    
        # Move the heatmap inside tab1
        st.header("Price Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        corr_cols = [col for col in numeric_cols if col != 'Price (₹)' and col in ['Duration (Minutes)', 'Number of Stops']]
        
        corr_cols.append('Price (₹)')
        
        if len(corr_cols) > 1:
            corr_df = df[corr_cols].corr()
            
            fig = px.imshow(
                corr_df, 
                text_auto=True, 
                color_continuous_scale='blues',
                title="Correlation Between Price and Other Factors"
            )
            
            fig.update_layout(
                width=800,
                height=500,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Insight:** This heatmap shows how different factors correlate with flight prices. A stronger positive correlation indicates that as one factor increases, prices tend to increase as well.
            """)
        else:
            st.warning("Not enough numerical columns for correlation analysis.")
    
    with tab2:
        st.subheader("Average Price by Route")
        
        route_counts = df.groupby(['Origin', 'Destination']).size().reset_index(name='Count')
        top_routes = route_counts.sort_values('Count', ascending=False).head(10)
        
        top_routes_df = df[df.apply(lambda x: (x['Origin'], x['Destination']) in 
                                   zip(top_routes['Origin'], top_routes['Destination']), axis=1)]
        
        route_avg = top_routes_df.groupby(['Origin', 'Destination'])['Price (₹)'].mean().reset_index()
        route_avg['Route'] = route_avg['Origin'] + ' → ' + route_avg['Destination']
        
        fig = px.bar(
            route_avg.sort_values('Price (₹)', ascending=False), 
            x='Route', 
            y='Price (₹)', 
            title="Average Price by Popular Routes",
            color='Price (₹)',
            color_continuous_scale='blues',
            labels={'Price (₹)': 'Average Price (₹)'}
        )
        
        fig.update_layout(
            xaxis_title="Route",
            yaxis_title="Average Price (₹)",
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Insight:** Some routes are consistently more expensive than others due to factors like demand, distance, and competition.
        """)
    
    with tab3:
        st.subheader("Price Trends Over Time")
        
        if 'Flight Date' in df.columns:
            df['Month'] = df['Flight Date'].dt.month
            df['Year'] = df['Flight Date'].dt.year
            df['YearMonth'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
            
            monthly_avg = df.groupby('YearMonth')['Price (₹)'].mean().reset_index()
            monthly_avg = monthly_avg.sort_values('YearMonth')
            
            fig = px.line(
                monthly_avg, 
                x='YearMonth', 
                y='Price (₹)', 
                title="Average Price Trend Over Time",
                markers=True,
                line_shape='linear',
                labels={'Price (₹)': 'Average Price (₹)', 'YearMonth': 'Month'}
            )
            
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Average Price (₹)",
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                xaxis=dict(gridcolor='rgba(0,0,0,0.1)')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Insight:** Flight prices tend to fluctuate throughout the year, with peaks during holiday seasons and dips during off-peak travel periods.
            """)
        else:
            st.warning("Flight date information is not available in the dataset.")
        
    

##########################################################################################################################################################
def show_prediction(df, model, feature_columns):
    st.title("✈️ Flight Price Predictor")
    st.write("Predict flight prices based on various factors.")

    # Check if model and scaler are loaded
    if model is None or feature_columns is None:
        st.error("Model or feature files not found. Cannot perform predictions.")
        return

    # Load the fitted scaler
    try:
        scaler = joblib.load("flight_scaler.pkl")
    except FileNotFoundError:
        st.error("Scaler file not found. Please ensure 'flight_scaler.pkl' exists.")
        return

    # Create a form for input
    with st.form("prediction_form"):
        st.subheader("Enter Flight Details")

        col1, col2 = st.columns(2)

        with col1:
            # Get unique values from dataset
            airlines = sorted(df['Airline'].unique().tolist())
            origins = sorted(df['Origin'].unique().tolist())
            destinations = sorted(df['Destination'].unique().tolist())

            airline = st.selectbox("Select Airline", airlines)
            origin = st.selectbox("Select Origin", origins)
            destination = st.selectbox("Select Destination", destinations)
            flight_class = st.selectbox("Select Class", ["Economy", "Business"])

        with col2:
            stops = st.number_input("Select Number of Stops", min_value=0, max_value=5, step=1)
            duration = st.number_input("Select Duration (Minutes)", min_value=30, max_value=1500, step=1)
            date = st.date_input("Select Flight Date", value=datetime.now())

            # Extract date components
            day = date.day
            month = date.month
            year = date.year

            # Extract departure and arrival hours (example logic)
            departure_time = st.time_input("Select Departure Time", value=datetime.now().time())
            arrival_time = st.time_input("Select Arrival Time", value=datetime.now().time())
            departure_hour = departure_time.hour
            arrival_hour = arrival_time.hour

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        # Prepare input data
        input_data = {
            "Number of Stops": stops,
            "Duration (Minutes)": duration,
            "Date": day,
            "Month": month,
            "Year": year,
            "Departure Hour": departure_hour,
            "Arrival Hour": arrival_hour,
            "Class": 0 if flight_class == "Economy" else 1  # Encode class
        }

        # One-hot encoding for categorical features
        for col in feature_columns:
            if col.startswith("Airline_"):
                input_data[col] = 1 if f"Airline_{airline}" == col else 0
            elif col.startswith("Origin_"):
                input_data[col] = 1 if f"Origin_{origin}" == col else 0
            elif col.startswith("Destination_"):
                input_data[col] = 1 if f"Destination_{destination}" == col else 0

        # Create DataFrame from input data
        input_df = pd.DataFrame([input_data])

        # Fill missing columns with 0
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Ensure columns are in the right order
        input_df = input_df[feature_columns]

        # Debugging: Print input data
        # st.write("Input Data for Prediction:")
        # st.write(input_df)

        # Identify numerical columns
        numerical_cols = ["Number of Stops", "Duration (Minutes)", "Date", "Month", "Year", "Departure Hour", "Arrival Hour"]

        # Scale only numerical features
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Predict price
        st.subheader("Prediction Result")
        with st.spinner("Predicting..."):
            predicted_price = model.predict(input_df)[0]

        # Display prediction result
        st.markdown(f"""
        <div class="prediction-result">
            <h4>💰 Estimated Price: ₹{predicted_price:,.2f}</h4>
        </div>
        """, unsafe_allow_html=True)
#----------------------------------------------------------------------------------------------------------------------
        
        # Compare with other airlines
        st.subheader("Price Comparison with Other Airlines")
        
        # Create a DataFrame for comparison
        comparison_data = []
        
        for comp_airline in airlines[:5]:  # Limit to top 5 airlines
            temp_data = input_data.copy()
            
            # Update airline encoding
            for col in feature_columns:
                if col.startswith("Airline_"):
                    temp_data[col] = 1 if f"Airline_{comp_airline}" == col else 0
            
            temp_df = pd.DataFrame([temp_data])
            
            # Fill missing columns with 0
            for col in feature_columns:
                if col not in temp_df.columns:
                    temp_df[col] = 0
            
            # Ensure columns are in the right order
            temp_df = temp_df[feature_columns]
            
            # Predict price
            comp_price = model.predict(temp_df)[0]
            
            comparison_data.append({
                "Airline": comp_airline,
                "Predicted Price (₹)": comp_price
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        fig = px.bar(
            comp_df, 
            x='Airline', 
            y='Predicted Price (₹)', 
            color='Predicted Price (₹)',
            color_continuous_scale='blues',
            title="Price Comparison Across Airlines"
        )
        
        fig.update_layout(
            xaxis_title="Airline",
            yaxis_title="Predicted Price (₹)",
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)



#############################################################################################################################################################

def show_exploration(df):
    st.title("✈️ Flight Data Exploration")
    
    # Filters in the sidebar
    st.sidebar.header("Filters")
    
    # Filter by airline
    airlines = ['All'] + sorted(df['Airline'].unique().tolist())
    selected_airline = st.sidebar.selectbox("Airline", airlines)
    
    # Filter by origin
    origins = ['All'] + sorted(df['Origin'].unique().tolist())
    selected_origin = st.sidebar.selectbox("Origin", origins)
    
    # Filter by destination
    destinations = ['All'] + sorted(df['Destination'].unique().tolist())
    selected_destination = st.sidebar.selectbox("Destination", destinations)
    
    # Filter by price range
    min_price = int(df['Price (₹)'].min())
    max_price = int(df['Price (₹)'].max())
    price_range = st.sidebar.slider(
        "Price Range (₹)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price)
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_airline != 'All':
        filtered_df = filtered_df[filtered_df['Airline'] == selected_airline]
    
    if selected_origin != 'All':
        filtered_df = filtered_df[filtered_df['Origin'] == selected_origin]
    
    if selected_destination != 'All':
        filtered_df = filtered_df[filtered_df['Destination'] == selected_destination]
    
    filtered_df = filtered_df[
        (filtered_df['Price (₹)'] >= price_range[0]) & 
        (filtered_df['Price (₹)'] <= price_range[1])
    ]
    
    # Display filtered data
    st.header("Filtered Flight Data")
    st.dataframe(filtered_df)

    
    
    
    # Visualizations
    st.header("Data Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Price Distribution", "Price vs Duration", "Stops Analysis"])
    
    with tab1:
        st.subheader("Price Distribution")
        
        fig = px.histogram(
            filtered_df, 
            x='Price (₹)', 
            nbins=30,
            title="Distribution of Flight Prices",
            color_discrete_sequence=['#185adb']
        )
        
        fig.update_layout(
            xaxis_title="Price (₹)",
            yaxis_title="Count",
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Insight:** This histogram shows the distribution of flight prices. The most common price range and any outliers can be identified from this visualization.
        """)
    
    with tab2:
        st.subheader("Price vs Duration")
        
        fig = px.scatter(
            filtered_df, 
            x='Duration (Minutes)', 
            y='Price (₹)', 
            color='Airline',
            hover_data=['Origin', 'Destination'],
            title="Price vs Duration by Airline"
        )
        
        fig.update_layout(
            xaxis_title="Duration (Minutes)",
            yaxis_title="Price (₹)",
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Insight:** This scatter plot shows the relationship between flight duration and price. Generally, longer flights tend to be more expensive, but there are exceptions.
        """)
    
    with tab3:
        st.subheader("Price by Number of Stops")
        
        # Box plot
        fig = px.box(
            filtered_df, 
            x='Number of Stops', 
            y='Price (₹)', 
            color='Number of Stops',
            title="Price Distribution by Number of Stops"
        )
        
        fig.update_layout(
            xaxis_title="Number of Stops",
            yaxis_title="Price (₹)",
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Insight:** This box plot shows how prices vary with the number of stops. Generally, non-stop flights are more expensive, but they save time.
        """)



################################################################################################################################################################

def show_about():
    st.title("About Flight Price Predictor")
    
    st.markdown("""
    ## ✈️ About the App
    
    The Flight Price Predictor is a data-driven application that helps travelers estimate the cost of flights based on various factors. 
    Using machine learning algorithms, it provides price predictions and insights into factors that affect flight prices.
    
    ### Features:
    
    - **Price Prediction**: Get estimated prices for flights based on airline, origin, destination, and other factors.
    - **Data Exploration**: Explore flight data to understand pricing patterns and trends.
    - **Dashboard**: Visualize key metrics and insights about flight prices.
    
    ### How it Works:
    
    The app uses a machine learning model trained on historical flight data to predict prices. The model takes into account:
    
    - Airline
    - Origin and destination
    - Flight duration
    - Number of stops
    - Travel class
    - Date of travel
    
    ### Data Source:
    
    The data used in this application includes flight information from various Indian airlines, covering domestic routes.
    
    
    ## 📧 Contact
    
    For questions or feedback, please contact:  
    Email: uashutosh309@gmail.com  
    GitHub: [Project Repository](https://github.com/Ashutosh1020/Decision-Science-Squad)
    """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2023 Flight Price Predictor | Created with passion and ❤️ </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()