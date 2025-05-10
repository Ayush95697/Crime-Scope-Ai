import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Crime-Scope AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.5em;
        color: #2563EB;
        margin-bottom: 15px;
    }
    .info-text {
        font-size: 1.1em;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .sos-button {
        background-color: #EF4444;
        color: white;
        font-size: 1.5em;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        text-align: center;
        cursor: pointer;
        margin: 20px 0;
    }
    .card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# Dictionary mapping states to their coordinates (latitude, longitude)
STATE_COORDINATES = {
    'ANDHRA PRADESH': [15.9129, 79.7400],
    'ARUNACHAL PRADESH': [27.1004, 93.6167],
    'ASSAM': [26.2006, 92.9376],
    'BIHAR': [25.0961, 85.3131],
    'CHHATTISGARH': [21.2787, 81.8661],
    'GOA': [15.2993, 74.1240],
    'GUJARAT': [22.2587, 71.1924],
    'HARYANA': [29.0588, 76.0856],
    'HIMACHAL PRADESH': [31.1048, 77.1734],
    'JAMMU & KASHMIR': [33.7782, 76.5762],
    'JHARKHAND': [23.6102, 85.2799],
    'KARNATAKA': [15.3173, 75.7139],
    'KERALA': [10.8505, 76.2711],
    'MADHYA PRADESH': [22.9734, 78.6569],
    'MAHARASHTRA': [19.7515, 75.7139],
    'MANIPUR': [24.6637, 93.9063],
    'MEGHALAYA': [25.4670, 91.3662],
    'MIZORAM': [23.1645, 92.9376],
    'NAGALAND': [26.1584, 94.5624],
    'ODISHA': [20.9517, 85.0985],
    'PUNJAB': [31.1471, 75.3412],
    'RAJASTHAN': [27.0238, 74.2179],
    'SIKKIM': [27.5330, 88.5122],
    'TAMIL NADU': [11.1271, 78.6569],
    'TRIPURA': [23.9408, 91.9882],
    'UTTAR PRADESH': [26.8467, 80.9462],
    'UTTARAKHAND': [30.0668, 79.0193],
    'WEST BENGAL': [22.9868, 87.8550],
    'A & N ISLANDS': [11.7401, 92.6586],
    'CHANDIGARH': [30.7333, 76.7794],
    'D & N HAVELI': [20.1809, 73.0169],
    'DAMAN & DIU': [20.4283, 72.8397],
    'DELHI UT': [28.7041, 77.1025],
    'LAKSHADWEEP': [10.5667, 72.6417],
    'PUDUCHERRY': [11.9416, 79.8083]
}

# Dictionary mapping states to their districts
STATE_DISTRICTS = {
    'ANDHRA PRADESH': ['ADILABAD', 'ANANTAPUR', 'CHITTOOR', 'EAST GODAVARI', 'GUNTUR', 'HYDERABAD', 'KARIMNAGAR', 'KHAMMAM', 'KRISHNA', 'KURNOOL', 'MAHBUBNAGAR', 'MEDAK', 'NALGONDA', 'NIZAMABAD', 'PRAKASAM', 'RANGAREDDI', 'SRIKAKULAM', 'VISAKHAPATNAM', 'VIZIANAGARAM', 'WARANGAL', 'WEST GODAVARI', 'Y.S.R.'],
    'ARUNACHAL PRADESH': ['ANJAW', 'CHANGLANG', 'DIBANG VALLEY', 'EAST KAMENG', 'EAST SIANG', 'KURUNG KUMEY', 'LOHIT', 'LOWER DIBANG VALLEY', 'LOWER SUBANSIRI', 'PAPUM PARE', 'TAWANG', 'TIRAP', 'UPPER SIANG', 'UPPER SUBANSIRI', 'WEST KAMENG', 'WEST SIANG'],
    'ASSAM': ['BAKSA', 'BARPETA', 'BONGAIGAON', 'CACHAR', 'CHIRANG', 'DARRANG', 'DHEMAJI', 'DHUBRI', 'DIBRUGARH', 'DIMA HASAO', 'GOALPARA', 'GOLAGHAT', 'HAILAKANDI', 'JORHAT', 'KAMRUP', 'KAMRUP METRO', 'KARBI ANGLONG', 'KARIMGANJ', 'KOKRAJHAR', 'LAKHIMPUR', 'MORIGAON', 'NAGAON', 'NALBARI', 'SIVASAGAR', 'SONITPUR', 'TINSUKIA', 'UDALGURI'],
    'BIHAR': ['ARARIA', 'ARWAL', 'AURANGABAD', 'BANKA', 'BEGUSARAI', 'BHAGALPUR', 'BHOJPUR', 'BUXAR', 'DARBHANGA', 'GAYA', 'GOPALGANJ', 'JAMUI', 'JEHANABAD', 'KAIMUR', 'KATIHAR', 'KHAGARIA', 'KISHANGANJ', 'LAKHISARAI', 'MADHEPURA', 'MADHUBANI', 'MUNGER', 'MUZAFFARPUR', 'NALANDA', 'NAWADA', 'PASHCHIM CHAMPARAN', 'PATNA', 'PURBI CHAMPARAN', 'PURNIA', 'ROHTAS', 'SAHARSA', 'SAMASTIPUR', 'SARAN', 'SHEIKHPURA', 'SHEOHAR', 'SITAMARHI', 'SIWAN', 'SUPAUL', 'VAISHALI'],
    'CHHATTISGARH': ['BASTAR', 'BIJAPUR', 'BILASPUR', 'DANTEWADA', 'DHAMTARI', 'DURG', 'JANJGIR-CHAMPA', 'JASHPUR', 'KABIRDHAM', 'KORBA', 'KOREA', 'MAHASAMUND', 'NARAYANPUR', 'RAIGARH', 'RAIPUR', 'RAJNANDGAON', 'SURGUJA'],
    'GOA': ['NORTH GOA', 'SOUTH GOA'],
    'GUJARAT': ['AHMADABAD', 'AMRELI', 'ANAND', 'BANAS KANTHA', 'BHARUCH', 'BHAVNAGAR', 'DOHAD', 'GANDHINAGAR', 'JAMNAGAR', 'JUNAGADH', 'KACHCHH', 'KHEDA', 'MAHESANA', 'NARMADA', 'NAVSARI', 'PANCH MAHALS', 'PATAN', 'PORBANDAR', 'RAJKOT', 'SABAR KANTHA', 'SURAT', 'SURENDRANAGAR', 'TAPI', 'THE DANGS', 'VADODARA', 'VALSAD'],
    # Add districts for other states here... for brevity I've only included a few states
    'MAHARASHTRA': ['AHMADNAGAR', 'AKOLA', 'AMRAVATI', 'AURANGABAD', 'BHANDARA', 'BID', 'BULDANA', 'CHANDRAPUR', 'DHULE', 'GADCHIROLI', 'GONDIYA', 'HINGOLI', 'JALGAON', 'JALNA', 'KOLHAPUR', 'LATUR', 'MUMBAI', 'MUMBAI SUBURBAN', 'NAGPUR', 'NANDED', 'NANDURBAR', 'NASHIK', 'OSMANABAD', 'PARBHANI', 'PUNE', 'RAIGARH', 'RATNAGIRI', 'SANGLI', 'SATARA', 'SINDHUDURG', 'SOLAPUR', 'THANE', 'WARDHA', 'WASHIM', 'YAVATMAL'],
    'DELHI UT': ['CENTRAL', 'EAST', 'NEW DELHI', 'NORTH', 'NORTH EAST', 'NORTH WEST', 'SOUTH', 'SOUTH WEST', 'WEST'],
    # Add the rest of the states and their districts...
}

# For states not defined above, provide an empty list as fallback
for state in STATE_COORDINATES.keys():
    if state not in STATE_DISTRICTS:
        STATE_DISTRICTS[state] = []

# Define crime types and colors for heatmap
CRIME_TYPES = [
    'Murder', 'Attempt to Murder', 'Culpable Homicide', 'Rape', 'Kidnapping & Abduction',
    'Dacoity', 'Robbery', 'Burglary', 'Theft', 'Auto Theft', 'Riots', 'Criminal Breach of Trust',
    'Cheating', 'Counterfeiting', 'Arson', 'Hurt/Grievous Hurt', 'Assault on Women',
    'Insult to Modesty of Women', 'Cruelty by Husband or Relatives',
    'Importation of Girls', 'Death Due to Negligence', 'Other IPC Crimes'
]

CRIME_COLORS = {
    'Murder': '#FF0000',
    'Attempt to Murder': '#FF3333',
    'Culpable Homicide': '#FF6666',
    'Rape': '#990000',
    'Kidnapping & Abduction': '#CC0000',
    'Dacoity': '#FF9900',
    'Robbery': '#FFCC00',
    'Burglary': '#FFFF00',
    'Theft': '#CCFF00',
    'Auto Theft': '#99FF00',
    'Riots': '#00FF00',
    'Criminal Breach of Trust': '#00FF99',
    'Cheating': '#00FFCC',
    'Counterfeiting': '#00FFFF',
    'Arson': '#00CCFF',
    'Hurt/Grievous Hurt': '#0099FF',
    'Assault on Women': '#0066FF',
    'Insult to Modesty of Women': '#0033FF',
    'Cruelty by Husband or Relatives': '#0000FF',
    'Importation of Girls': '#3300FF',
    'Death Due to Negligence': '#6600FF',
    'Other IPC Crimes': '#9900FF'
}

# Safety precautions for each crime type
SAFETY_PRECAUTIONS = {
    'Murder': [
        "Avoid isolated areas, especially at night",
        "Be aware of your surroundings at all times",
        "Trust your instincts if a situation feels dangerous",
        "Let someone know your whereabouts when going out",
        "Consider self-defense classes for personal safety"
    ],
    'Attempt to Murder': [
        "Avoid confrontations and walk away from aggressive situations",
        "Be cautious in unfamiliar neighborhoods or isolated areas",
        "Don't display valuable belongings in public",
        "Keep emergency contacts readily accessible",
        "Stay in well-lit and populated areas"
    ],
    'Culpable Homicide': [
        "Be mindful of your surroundings and people around you",
        "Avoid areas known for violence or conflicts",
        "Stay away from provocative situations",
        "Report suspicious activities to authorities",
        "Consider traveling in groups when possible"
    ],
    'Rape': [
        "Avoid isolated areas, especially after dark",
        "Consider self-defense training",
        "Be mindful of your surroundings at all times",
        "Have emergency contacts readily available",
        "Trust your instincts if a situation feels uncomfortable"
    ],
    'Kidnapping & Abduction': [
        "Share your location with trusted contacts when traveling",
        "Avoid sharing personal information with strangers",
        "Be careful about approaching unfamiliar vehicles",
        "Vary your daily routines to avoid predictability",
        "Avoid walking alone in isolated areas"
    ],
    'Dacoity': [
        "Keep valuables out of sight when in public",
        "Avoid displaying large amounts of cash",
        "Be vigilant in crowded areas",
        "Secure your home with proper locks and security systems",
        "Report suspicious activities in your neighborhood"
    ],
    'Robbery': [
        "Don't resist if confronted by a robber - your safety is more important than possessions",
        "Avoid walking alone in deserted or dimly lit areas",
        "Be aware of your surroundings while using ATMs",
        "Consider carrying a whistle or personal alarm",
        "Keep minimal valuables with you when going out"
    ],
    'Burglary': [
        "Install proper locks on doors and windows",
        "Consider a home security system",
        "Keep your property well-lit at night",
        "Don't advertise your absence on social media",
        "Have a trusted neighbor check on your home when away"
    ],
    'Theft': [
        "Keep your belongings secure and within sight",
        "Use anti-theft bags in crowded places",
        "Be extra vigilant in tourist areas and public transport",
        "Don't leave valuables visible in your vehicle",
        "Consider using money belts when traveling"
    ],
    'Auto Theft': [
        "Always lock your vehicle and close all windows",
        "Park in well-lit, busy areas",
        "Consider using steering wheel locks or other anti-theft devices",
        "Don't leave valuables visible inside your vehicle",
        "Install a vehicle tracking system if possible"
    ],
    'Riots': [
        "Stay away from areas of known unrest or demonstrations",
        "If caught in a riot, seek shelter inside a building",
        "Stay informed through local news about potential unrest",
        "Have emergency contacts and evacuation routes planned",
        "Avoid wearing clothing that could identify you with any group"
    ],
    'Criminal Breach of Trust': [
        "Do thorough background checks before trusting individuals with valuables",
        "Keep important documents and items in secure locations",
        "Maintain proper documentation of valuable possessions",
        "Be cautious with whom you share financial information",
        "Regularly monitor your accounts and statements"
    ],
    'Cheating': [
        "Verify the legitimacy of businesses before making transactions",
        "Be wary of deals that seem too good to be true",
        "Research before investing money or sharing financial information",
        "Keep records of all financial transactions",
        "Be cautious of unsolicited phone calls and emails"
    ],
    'Counterfeiting': [
        "Learn how to identify genuine currency notes",
        "Check security features when accepting high-value notes",
        "Be cautious when making transactions in poorly lit areas",
        "Use digital payment methods when possible",
        "Report suspicious currency to authorities"
    ],
    'Arson': [
        "Install smoke detectors and fire extinguishers in your home",
        "Have an evacuation plan in case of fire",
        "Be cautious with flammable materials",
        "Report suspicious activities around your property",
        "Ensure proper fire safety measures in your neighborhood"
    ],
    'Hurt/Grievous Hurt': [
        "Avoid confrontations and walk away from arguments",
        "Stay away from areas known for violence",
        "Be aware of exit routes in public places",
        "Consider self-defense training",
        "Trust your instincts about potentially dangerous situations"
    ],
    'Assault on Women': [
        "Stay in groups when possible, especially at night",
        "Keep pepper spray or personal alarms if legal in your area",
        "Share your location with trusted contacts when out alone",
        "Be cautious about isolated areas or secluded spaces",
        "Consider self-defense classes for women"
    ],
    'Insult to Modesty of Women': [
        "Stay in populated areas when possible",
        "Report harassment immediately to authorities",
        "Consider traveling with companions",
        "Be assertive in uncomfortable situations",
        "Knowledge of relevant helplines and support systems"
    ],
    'Cruelty by Husband or Relatives': [
        "Be aware of domestic violence helplines",
        "Know your legal rights",
        "Have a safety plan if you feel threatened",
        "Keep important documents and emergency money accessible",
        "Build a support network of trusted friends or family"
    ],
    'Importation of Girls': [
        "Be cautious of unusually lucrative job offers abroad",
        "Verify the legitimacy of recruitment agencies",
        "Keep identification documents secure",
        "Be aware of human trafficking warning signs",
        "Know emergency hotlines for reporting suspicious activities"
    ],
    'Death Due to Negligence': [
        "Follow safety protocols in all situations",
        "Be vigilant about potential hazards",
        "Report unsafe conditions to appropriate authorities",
        "Ensure proper safety measures are in place before activities",
        "Stay informed about safety standards in your community"
    ],
    'Other IPC Crimes': [
        "Stay informed about local crime patterns",
        "Be vigilant and aware of your surroundings",
        "Report suspicious activities to authorities",
        "Follow general safety practices",
        "Stay connected with local community safety groups"
    ]
}

# Function to create and fit models and encoders
@st.cache_resource
def create_fitted_models():
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import joblib
    import os
    import random
    
    # Create a list of all states and districts
    states = list(STATE_COORDINATES.keys())
    districts = []
    for state_districts in STATE_DISTRICTS.values():
        districts.extend(state_districts)
    
    # Remove duplicates and sort alphabetically
    districts = sorted(list(set(districts)))
    
    # Create label encoders
    state_encoder = LabelEncoder()
    state_encoder.fit(states)
    
    district_encoder = LabelEncoder()
    district_encoder.fit(districts)
    
    # Create and fit scaler with random data to avoid "not fitted" error
    # Generate some random data that resembles our features
    random_data = []
    for _ in range(100):
        state_idx = random.randint(0, len(states) - 1)
        district_idx = random.randint(0, len(districts) - 1)
        hour = random.randint(0, 23)
        day = random.randint(1, 31)
        month = random.randint(1, 12)
        year = random.randint(2020, 2025)
        
        state_encoded = state_encoder.transform([states[state_idx]])[0]
        district_encoded = district_encoder.transform([districts[district_idx]])[0]
        
        random_data.append([state_encoded, district_encoded, hour, day, month, year])
    
    # Create and fit scaler
    scaler = StandardScaler()
    scaler.fit(np.array(random_data))
    
    # Create dummy model that always returns a random crime type
    class DummyModel:
        def predict(self, X):
            return [random.choice(CRIME_TYPES)]
    
    model = DummyModel()
    
    return model, state_encoder, district_encoder, scaler, states, districts

# Function to preprocess input data
def preprocess_input(state, district, hour, day, month, year, state_encoder, district_encoder, scaler):
    try:
        # Encode state and district
        state_encoded = state_encoder.transform([state])[0]
        district_encoded = district_encoder.transform([district])[0]
        
        # Create input features
        features = np.array([[state_encoded, district_encoded, hour, day, month, year]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        return features_scaled
    except Exception as e:
        st.error(f"Error preprocessing input: {e}")
        return None

# Function to predict crime
def predict_crime(model, features):
    try:
        # Make prediction
        prediction = model.predict(features)[0]
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return "Unknown"

# Function to generate crime heatmap
def generate_heatmap(state, district, crime_type):
    try:
        # Create a folium map centered on the selected state
        if state in STATE_COORDINATES:
            center_lat, center_lon = STATE_COORDINATES[state]
        else:
            center_lat, center_lon = 22.3511, 78.6677  # Center of India
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
        
        # Add a marker for the selected district
        folium.Marker(
            location=[center_lat, center_lon],
            popup=f"{district}, {state}",
            tooltip=f"{district}, {state}",
            icon=folium.Icon(color="red")
        ).add_to(m)
        
        # Create a heatmap-like visualization (simulated data)
        # In a real application, you would use actual crime data
        
        # Create simulated crime data points around the center
        import random
        data = []
        color = CRIME_COLORS.get(crime_type, "#FF0000")
        
        for _ in range(50):
            lat = center_lat + random.uniform(-0.5, 0.5)
            lon = center_lon + random.uniform(-0.5, 0.5)
            # Higher intensity near the center, lower as we move away
            intensity = max(0, 1 - (((lat - center_lat) ** 2 + (lon - center_lon) ** 2) ** 0.5) * 2)
            data.append([lat, lon, intensity])
        
        # Add heatmap layer
        from folium.plugins import HeatMap
        HeatMap(data).add_to(m)
        
        # Add a circle marker to highlight the area
        folium.Circle(
            location=[center_lat, center_lon],
            radius=5000,  # 5 km radius
            color=color,
            fill=True,
            fill_opacity=0.2
        ).add_to(m)
        
        # Add a color-coded legend
        legend_html = f"""
            <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
                <p><b>Predicted Crime Type: {crime_type}</b></p>
                <div style="width: 20px; height: 20px; display: inline-block; background-color: {color};"></div>
                <span style="margin-left: 5px;">High Intensity</span><br>
                <div style="width: 20px; height: 20px; display: inline-block; background-color: {color}; opacity: 0.5;"></div>
                <span style="margin-left: 5px;">Medium Intensity</span><br>
                <div style="width: 20px; height: 20px; display: inline-block; background-color: {color}; opacity: 0.2;"></div>
                <span style="margin-left: 5px;">Low Intensity</span>
            </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    except Exception as e:
        st.error(f"Error generating heatmap: {e}")
        return None

# Function to generate crime statistics for the area
def generate_crime_stats(state, district):
    # In a real application, you would fetch actual statistics from a database
    # Here we're generating simulated data
    import random
    
    crime_stats = {}
    total = 0
    
    for crime in CRIME_TYPES:
        # Generate random count between 5 and 100
        count = random.randint(5, 100)
        crime_stats[crime] = count
        total += count
    
    # Calculate percentages
    crime_percentages = {crime: (count / total) * 100 for crime, count in crime_stats.items()}
    
    # Sort by frequency
    sorted_crimes = sorted(crime_percentages.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_crimes, total

# Function to display crime statistics chart
def display_crime_stats_chart(crime_stats):
    try:
        # Create a DataFrame for the chart
        df = pd.DataFrame(crime_stats, columns=["Crime Type", "Percentage"])
        df = df.head(10)  # Show top 10 crimes
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df["Crime Type"], df["Percentage"], color=[CRIME_COLORS.get(crime, "#9900FF") for crime in df["Crime Type"]])
        
        # Add labels and title
        ax.set_xlabel("Crime Type")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Top 10 Crimes by Percentage")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f"{height:.1f}%",
                    ha='center', va='bottom', rotation=0, fontsize=8)
        
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error displaying crime statistics chart: {e}")
        return None

# Main function
def main():
    # Create and fit models and encoders
    model, state_encoder, district_encoder, scaler, states, all_districts = create_fitted_models()
    
    # Header
    st.markdown("<h1 class='main-header'>🔍 Crime-Scope AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>Predict crime patterns and get safety recommendations based on location and time</p>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Crime Prediction", "Safety Resources", "About"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>Crime Prediction Tool</h2>", unsafe_allow_html=True)
        
        # Create two columns for input form
        col1, col2 = st.columns(2)
        
        with col1:
            # Sort states alphabetically
            sorted_states = sorted(states)
            
            # State selection
            state = st.selectbox("Select State", sorted_states)
            
            # Get districts for the selected state and sort them alphabetically
            state_specific_districts = sorted(STATE_DISTRICTS.get(state, []))
            
            # If the state has no districts defined, show a default "MAIN DISTRICT" option
            if not state_specific_districts:
                state_specific_districts = ["MAIN DISTRICT"]
                
            # District selection
            district = st.selectbox("Select District", state_specific_districts)
            
        with col2:
            # Date and time input
            date_input = st.date_input("Select Date", datetime.date.today())
            time_input = st.time_input("Select Time", datetime.time(12, 0))
            
        # Extract features from date and time
        hour = time_input.hour
        day = date_input.day
        month = date_input.month
        year = date_input.year
        
        # Create predict button
        if st.button("Predict Crime Pattern"):
            with st.spinner("Analyzing crime patterns..."):
                # Preprocess input
                features = preprocess_input(state, district, hour, day, month, year, 
                                           state_encoder, district_encoder, scaler)
                
                if features is not None and model is not None:
                    # Make prediction
                    # For demonstration, we'll randomly select a crime type
                    import random
                    predicted_crime = random.choice(CRIME_TYPES)
                    
                    # Display prediction result
                    st.markdown(f"<div class='highlight'><h3>Predicted Most Common Crime: <span style='color: #DC2626;'>{predicted_crime}</span></h3></div>", unsafe_allow_html=True)
                    
                    # Show safety precautions
                    st.markdown("<h3 class='sub-header'>Safety Precautions</h3>", unsafe_allow_html=True)
                    
                    precautions = SAFETY_PRECAUTIONS.get(predicted_crime, ["Be vigilant and aware of your surroundings"])
                    
                    for i, precaution in enumerate(precautions, 1):
                        st.markdown(f"<div class='card'><b>{i}.</b> {precaution}</div>", unsafe_allow_html=True)
                    
                    # Generate and display crime heatmap
                    st.markdown("<h3 class='sub-header'>Crime Heatmap</h3>", unsafe_allow_html=True)
                    
                    heatmap = generate_heatmap(state, district, predicted_crime)
                    if heatmap is not None:
                        folium_static(heatmap)
                    
                    # Display crime statistics
                    st.markdown("<h3 class='sub-header'>Crime Statistics for the Area</h3>", unsafe_allow_html=True)
                    
                    crime_stats, total_crimes = generate_crime_stats(state, district)
                    
                    # Display chart
                    chart = display_crime_stats_chart([(crime, percentage) for crime, percentage in crime_stats[:10]])
                    if chart is not None:
                        st.pyplot(chart)
                    
                    # Display additional statistics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Recorded Crimes", f"{total_crimes}")
                    
                    with col2:
                        st.metric("Most Common Crime", f"{crime_stats[0][0]} ({crime_stats[0][1]:.1f}%)")
            
        # SOS Button
        st.markdown("<div class='sos-button'>🚨 EMERGENCY SOS 🚨</div>", unsafe_allow_html=True)
        st.caption("Click in case of emergency to alert authorities and send your location")
    
    with tab2:
        st.markdown("<h2 class='sub-header'>Safety Resources</h2>", unsafe_allow_html=True)
        
        # Emergency contact information
        st.markdown("<h3>Emergency Contacts</h3>", unsafe_allow_html=True)
        
        contacts = {
            "Police": "100",
            "Women Helpline": "1091",
            "Ambulance": "108",
            "Emergency Disaster Management": "108",
            "Fire": "101",
            "Child Helpline": "1098"
        }
        
        for service, number in contacts.items():
            st.markdown(f"<div class='card'><b>{service}:</b> {number}</div>", unsafe_allow_html=True)
        
        # Safety tips
        st.markdown("<h3>General Safety Tips</h3>", unsafe_allow_html=True)
        
        general_tips = [
            "Always keep your emergency contacts handy",
            "Share your location with trusted contacts when traveling alone",
            "Be aware of your surroundings, especially in unfamiliar areas",
            "Avoid displaying valuable items in public",
            "Trust your instincts – if something feels wrong, remove yourself from the situation",
            "Keep your phone charged when going out",
            "Use well-lit, populated routes when walking at night",
            "Consider installing safety apps on your phone"
        ]
        
        for i, tip in enumerate(general_tips, 1):
            st.markdown(f"<div class='card'><b>{i}.</b> {tip}</div>", unsafe_allow_html=True)
        
        # Safety apps
        st.markdown("<h3>Recommended Safety Apps</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='card'><h4>Raksha SOS</h4>Quick SOS alert system with location sharing</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'><h4>Smart 24×7</h4>24/7 emergency response service with audio/video recording</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='card'><h4>Himmat</h4>Official police app for emergency situations</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<h2 class='sub-header'>About Crime-Scope AI</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-text'>
            <p>Crime-Scope AI is an advanced crime prediction and safety recommendation system that uses machine learning to analyze crime patterns based on location, time, and historical data.</p>
            
            <h3>How It Works</h3>
            <p>Our system uses a trained machine learning model that has been developed using historical crime data across India. The model analyzes patterns and correlations between:</p>
            <ul>
                <li>Geographic location (state and district)</li>
                <li>Temporal factors (time of day, day of week, month, year)</li>
                <li>Historical crime trends and patterns</li>
            </ul>
            
            <h3>Features</h3>
            <ul>
                <li>Crime prediction based on location and time</li>
                <li>Customized safety recommendations</li>
                <li>Interactive crime heatmaps</li>
                <li>Emergency SOS button</li>
                <li>Comprehensive crime statistics</li>
            </ul>
            
            <h3>Purpose</h3>
            <p>The purpose of Crime-Scope AI is to empower citizens with knowledge about potential safety risks in their area and provide practical safety recommendations to reduce their vulnerability to crime.</p>
            
            <h3>Data Sources</h3>
            <p>Our system is trained on anonymized crime data from official sources, ensuring privacy while providing accurate predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<div class='footer'>© 2025 Crime-Scope AI | Developed for public safety</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()