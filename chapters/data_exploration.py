import time
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pydeck as pdk
import requests
import plotly.graph_objects as go

from chapters.overview import df

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.image("./assets/DataExploration.jpg", use_container_width=True)

st.title("Data Exploration")

# Display Dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Show Basic Info
st.write(f"Total Records: **{df.shape[0]}** |" f" Total Features: **{df.shape[1]}**")

# Summary Statistics
st.subheader("Summary Statistics")
st.write(df.describe(include='all'))

# Check for Missing Values
st.write("Missing Values")
missing_values = df.isnull().sum()
st.write(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")

# Visualizations
st.header("Data Visualizations")

col1, col2 = st.columns(2)




################ GENDER DISTRIBUTION ################





with col1:
    st.subheader("Gender Distribution")
    # Get gender counts
    gender_counts = df['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']

    # Create pie chart using plotly express with custom colors
    fig = px.pie(
        gender_counts,
        values='Count',
        names='Gender',
        #title='Gender Distribution',
        hover_data=['Count'],
        labels={'Count': 'Number of Records'},
        color='Gender',
        color_discrete_map={'Male': 'blue', 'Female': 'pink'}
    )

    # Customize the chart
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label+value',
        hole=0.3,  # Creates a donut chart effect - remove if you prefer a regular pie chart
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)






##################### DEPRESSION STATUS #####################






with col2:
    st.subheader("Depression Status Distribution")
    # Get depression counts
    depression_counts = df['Depression'].value_counts().reset_index()
    depression_counts.columns = ['Depression', 'Count']

    # Map the numeric values to descriptive labels
    depression_counts['Depression'] = depression_counts['Depression'].map({
        0: "Do not Have Depression",
        1: "Have Depression"
    })

    # Create pie chart using plotly express with custom colors
    fig = px.pie(
        depression_counts,
        values='Count',
        names='Depression',
        #title='Depression Status Distribution',
        hover_data=['Count'],
        labels={'Count': 'Number of Records'},
        color='Depression',
        color_discrete_map={
            "Do not Have Depression": "white",
            "Have Depression": "black"
        }
    )

    # Customize the chart
    fig.update_traces(
        textposition='inside',
        textinfo='percent+value',
        hole=0.3,  # Creates a donut chart effect - remove if you prefer a regular pie chart
        marker=dict(line=dict(color='#333333', width=2))
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)







#################### AGE DISTRIBUTION ####################





# Calculate age distribution
age_counts = df['Age'].value_counts().sort_index()
age_distribution = pd.DataFrame({
    'Age': age_counts.index,
    'Count': age_counts.values
})

# Set up the sidebar elements
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

# Title for the chart
st.subheader("Age Distribution Chart")

# Create the initial empty chart
chart = st.bar_chart()

# Prepare data for animation
steps = 20  # Number of animation steps
max_count = age_distribution['Count'].max()

# Start with empty data frame
display_data = pd.DataFrame(index=age_distribution['Age'])

# Animate the chart building
for i in range(1, steps + 1):
    # Calculate the proportion of counts to show in this step
    proportion = i / steps

    # Create new data with partial counts
    new_data = age_distribution.copy()
    new_data['Count'] = age_distribution['Count'] * proportion

    # Update chart
    chart.add_rows(new_data.set_index('Age'))

    # Update progress
    status_text.text(f"Building age distribution: {int(proportion * 100)}% complete")
    progress_bar.progress(i / steps)

    # Short pause for animation effect
    time.sleep(0.1)

# Clear the progress elements
progress_bar.empty()
status_text.text("Age distribution complete!")







##################### SLEEP DURATION DISTRIBUTION #####################





st.subheader("Sleep Duration Distribution")
# Get sleep duration counts
sleep_counts = df['Sleep Duration'].value_counts().reset_index()
sleep_counts.columns = ['Sleep Duration', 'Count']

# Sort by sleep duration (assuming it's numeric)
sleep_counts = sleep_counts.sort_values('Sleep Duration')

# Create bar chart using plotly express
fig = px.bar(
    sleep_counts,
    x='Sleep Duration',
    y='Count',
    #title='Sleep Duration Distribution',
    labels={'Sleep Duration': 'Hours of Sleep', 'Count': 'Number of Records'},
    text='Count',  # Display count on bars
    color='Sleep Duration',
    color_continuous_scale='RdBu'  # Red to Blue color scale
)

# Customize the chart
fig.update_traces(
    textposition='outside',
    texttemplate='%{text}',
    marker_line_width=1,
    marker_line_color='white'
)

fig.update_layout(
    xaxis_title='Sleep Duration (hours)',
    yaxis_title='Count',
    coloraxis_showscale=False  # Hide the color scale
)

# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Create a two-column layout
col1, col2 = st.columns(2)



##################### ACADEMIC PRESSURE DISTRIBUTION #####################



with col1:
    st.subheader("Academic Pressure Distribution")
    # Get counts for Academic Pressure
    pressure_counts = df["Academic Pressure"].value_counts().reset_index()
    pressure_counts.columns = ['Pressure Level', 'Count']

    # Sort by pressure level to ensure correct order (assuming it's a numeric scale)
    pressure_counts = pressure_counts.sort_values('Pressure Level')

    # Create the bar chart for Academic Pressure
    fig_pressure = px.bar(
        pressure_counts,
        x='Pressure Level',
        y='Count',
        text='Count',
        labels={'Pressure Level': 'Academic Pressure Level', 'Count': 'Number of Students'},
        #title='Academic Pressure Distribution'
    )

    # Customize the chart with a red gradient (higher pressure = darker red)
    pressure_levels = len(pressure_counts)
    red_gradient = [
        f'rgba({180 + i * 75 / pressure_levels}, {50 - i * 50 / pressure_levels}, {50 - i * 50 / pressure_levels}, 0.8)'
        for i in range(pressure_levels)]

    fig_pressure.update_traces(
        marker_color=red_gradient,
        marker_line_color='white',
        marker_line_width=1,
        textposition='outside'
    )

    fig_pressure.update_layout(
        xaxis=dict(
            title="Academic Pressure Level",
            tickmode='linear'
        ),
        yaxis_title="Number of Students"
    )

    # Display the chart
    st.plotly_chart(fig_pressure, use_container_width=True)



##################### STUDY SATISFACTION DISTRIBUTION #####################


with col2:
    st.subheader("Study Satisfaction Distribution")
    # Get counts for Study Satisfaction
    satisfaction_counts = df["Study Satisfaction"].value_counts().reset_index()
    satisfaction_counts.columns = ['Satisfaction Level', 'Count']

    # Sort by satisfaction level to ensure correct order (assuming it's a numeric scale)
    satisfaction_counts = satisfaction_counts.sort_values('Satisfaction Level')

    # Create the bar chart for Study Satisfaction
    fig_satisfaction = px.bar(
        satisfaction_counts,
        x='Satisfaction Level',
        y='Count',
        text='Count',
        labels={'Satisfaction Level': 'Study Satisfaction Level', 'Count': 'Number of Students'},
        #title='Study Satisfaction Distribution'
    )

    # Customize the chart with a green gradient (higher satisfaction = darker green)
    satisfaction_levels = len(satisfaction_counts)
    green_gradient = [
        f'rgba({50 - i * 50 / satisfaction_levels}, {150 + i * 105 / satisfaction_levels}, {50 - i * 50 / satisfaction_levels}, 0.8)'
        for i in range(satisfaction_levels)]

    fig_satisfaction.update_traces(
        marker_color=green_gradient,
        marker_line_color='white',
        marker_line_width=1,
        textposition='outside'
    )

    fig_satisfaction.update_layout(
        xaxis=dict(
            title="Study Satisfaction Level",
            tickmode='linear'
        ),
        yaxis_title="Number of Students"
    )

    # Display the chart
    st.plotly_chart(fig_satisfaction, use_container_width=True)








##################### WORK/STUDY HOURS DISTRIBUTION #####################






st.subheader("Work/Study Hours Distribution")
# Get counts for each integer value (0-12)
hour_counts = df["Work/Study Hours"].value_counts().reset_index()
hour_counts.columns = ['Hours', 'Count']

# Sort by hours to ensure correct order
hour_counts = hour_counts.sort_values('Hours')

# Make sure we have all values from 0-12
all_hours = pd.DataFrame({'Hours': range(13)})
hour_counts = pd.merge(all_hours, hour_counts, on='Hours', how='left').fillna(0)
hour_counts['Count'] = hour_counts['Count'].astype(int)

# Create a custom blue color gradient (starting with a visible light blue)
# We'll use a range from #D4E6F1 (light blue) to #0D47A1 (dark blue)
n_colors = 13  # 0-12 hours
blue_gradient = [
    f'rgba({max(30, int(30 + (200-i*17)))}, {max(100, int(144 - i*10))}, {min(255, int(180 + i*6))}, 0.9)'
    for i in range(n_colors)
]

# Create bar chart
fig = go.Figure()
fig.add_trace(go.Bar(
    x=hour_counts['Hours'],
    y=hour_counts['Count'],
    text=hour_counts['Count'],
    textposition='outside',
    marker_color=blue_gradient,
    marker_line_color='white',
    marker_line_width=1,
))

# Update layout
fig.update_layout(
    #title="Work/Study Hours Distribution",
    xaxis=dict(
        #title="Work/Study Hours",
        tickmode='linear',  # Force all ticks to show
        tick0=0,
        dtick=1,  # Step size of 1
        range=[-0.5, 12.5]  # Add some padding
    ),
    yaxis_title="Count",
    bargap=0.2  # Gap between bars
)

# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)





######################### CGPA DISTRIBUTION #########################

st.subheader("CGPA Distribution")
from scipy import stats

# Create figure
fig = go.Figure()

# Add histogram
fig.add_trace(go.Histogram(
    x=df["CGPA"],
    autobinx=False,
    xbins=dict(start=0, end=10, size=0.5),
    name="CGPA Frequency",
    marker_color="#3366CC",
    opacity=0.7
))

# Calculate KDE
kde_x = np.linspace(0, 10, 100)
kde = stats.gaussian_kde(df["CGPA"].dropna())
kde_y = kde(kde_x) * len(df["CGPA"]) * 0.5  # Scale to match histogram height

# Add KDE line
fig.add_trace(go.Scatter(
    x=kde_x,
    y=kde_y,
    mode='lines',
    name='Density',
    line=dict(color='red', width=2)
))

# Update layout
fig.update_layout(
    #title="CGPA Distribution with Density Curve",
    xaxis=dict(
        title="CGPA Value",
        range=[0, 10],
        dtick=1,
        tickmode='linear'
    ),
    yaxis_title="Frequency",
    bargap=0.05,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)









##################### CORRELATION HEATMAP #####################









# Correlation Heatmap
st.subheader("Correlation Heatmap")
num_df = df.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)





########################## CITY DISTRIBUTION ##########################




# Assuming df is your dataframe with 'City' and 'Depression' columns
st.subheader("City Distribution by Depression Status")

# Cache the geocoding function to speed up loading
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_coordinates(city_name):
    # Add country suffix to improve accuracy
    search_query = f"{city_name}, India"

    # We can remove the sleep delay since we're caching the results
    try:
        response = requests.get(
            f"https://nominatim.openstreetmap.org/search?q={search_query}&format=json&limit=1",
            headers={"User-Agent": "Streamlit City Visualization"}
        )
        data = response.json()

        if data and len(data) > 0:
            return float(data[0]["lat"]), float(data[0]["lon"])
        return None
    except:
        return None


# Cache the processing of city data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def process_city_data(df):
    # List of valid Indian cities
    valid_cities = [
        "Visakhapatnam", "Bangalore", "Srinagar", "Varanasi", "Jaipur", "Pune",
        "Thane", "Chennai", "Nagpur", "Nashik", "Vadodara", "Kalyan", "Rajkot",
        "Ahmedabad", "Kolkata", "Mumbai", "Lucknow", "Indore", "Surat",
        "Ludhiana", "Bhopal", "Meerut", "Agra", "Ghaziabad", "Hyderabad",
        "Vasai-Virar", "Kanpur", "Patna", "Faridabad", "Delhi"
    ]

    # Filter your DataFrame to include only valid cities
    filtered_df = df[df['City'].isin(valid_cities)].copy()

    # Group by City and Depression status to get counts
    city_depression_counts = filtered_df.groupby(['City', 'Depression']).size().reset_index(name='Count')

    # Map depression values to descriptive labels
    city_depression_counts['Depression_Status'] = city_depression_counts['Depression'].map({
        0: "Do not Have Depression",
        1: "Have Depression"
    })

    # Get coordinates for each city
    city_coords = {}
    for city in valid_cities:
        coords = get_coordinates(city)
        if coords:
            city_coords[city] = coords

    # Create data for visualization
    city_data = []
    for _, row in city_depression_counts.iterrows():
        city = row['City']
        if city in city_coords:
            lat, lon = city_coords[city]

            # Add a small offset for visualization based on depression status
            offset = 0.01 if row['Depression'] == 1 else -0.01

            city_data.append({
                "City": city,
                "Count": row['Count'],
                "Depression": row['Depression'],
                "Depression_Status": row['Depression_Status'],
                "lat": lat,
                "lon": lon + offset,
                "color": [255, 0, 0] if row['Depression'] == 1 else [0, 128, 0]
            })

    return pd.DataFrame(city_data)


# Process the data using cached function
city_df = process_city_data(df)

if not city_df.empty:
    # Create a PyDeck layer for depression status visualization
    layer = pdk.Layer(
        "ColumnLayer",
        data=city_df,
        get_position=["lon", "lat"],
        get_elevation="Count * 800",
        elevation_scale=0.5,
        radius=8000,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    # Set the viewport location (centered on India)
    view_state = pdk.ViewState(
        latitude=20.5937,
        longitude=78.9629,
        zoom=4,
        pitch=40,
    )

    # Create the PyDeck chart with enhanced tooltip
    tooltip = {
        "html": "<b>{City}</b><br/><b>{Depression_Status}</b>: {Count} records",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v9",
    )

    # Display the chart in Streamlit
    st.pydeck_chart(deck)

    # Add a legend
    st.markdown("""
    <div style="display: flex; align-items: center; margin-top: 10px;">
        <div style="width: 15px; height: 15px; background-color: red; margin-right: 5px;"></div>
        <span style="margin-right: 20px;">Have Depression</span>
        <div style="width: 15px; height: 15px; background-color: green; margin-right: 5px;"></div>
        <span>Do not Have Depression</span>
    </div>
    """, unsafe_allow_html=True)

    # Prepare data for the pivot table with cities as columns
    pivot_data = city_df[["City", "Depression_Status", "Count"]].copy()

    # Create a pivot table with Depression_Status as rows and City as columns
    pivot_table = pivot_data.pivot_table(
        index="Depression_Status",
        columns="City",
        values="Count",
        fill_value=0
    )

    # Rename the index for better display
    pivot_table = pivot_table.rename(
        index={
            "Have Depression": "Yes",
            "Do not Have Depression": "No"
        }
    )

    # Reset the index to make Depression_Status a regular column
    pivot_table = pivot_table.reset_index()

    # Rename the first column
    pivot_table = pivot_table.rename(columns={"Depression_Status": "Depression Status"})

    # Remove the column name for the city columns
    pivot_table.columns.name = None

    # Display the restructured table
    st.write(pivot_table)
else:
    st.write("No valid city data found for visualization.")











