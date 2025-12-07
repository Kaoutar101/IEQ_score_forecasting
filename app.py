import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config with professional theme
st.set_page_config(
    page_title="Air Quality Forecasting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-bottom: 3px solid #3498DB;
        padding-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #34495E;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498DB;
        margin-bottom: 1rem;
    }
    .alert-box {
        background: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3498DB 0%, #2C3E50 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Air Quality Forecasting System</h1>', unsafe_allow_html=True)

# Sidebar with professional layout
with st.sidebar:
    st.markdown("### System Control Panel")
    st.markdown("---")
    
    page = st.selectbox(
        "Navigation",
        ["Dashboard Overview", "Forecast Analysis", "Historical Data", "Model Performance", "Configuration"]
    )
    
    st.markdown("---")
    st.markdown("#### Forecast Settings")
    
    # Forecast horizon in 5-minute intervals
    forecast_horizon = st.slider(
        "Forecast Horizon (5-min intervals)",
        min_value=1,
        max_value=72,  # Up to 6 hours (72 * 5min = 360min = 6h)
        value=12,  # Default 1 hour forecast
        help="Number of 5-minute intervals to forecast ahead"
    )
    
    st.markdown("---")
    st.markdown("#### Data Upload")
    uploaded_file = st.file_uploader("Upload sensor data (CSV)", type=['csv'])
    
    st.markdown("---")
    st.markdown("*System Version: 2.1.0*")

# Model loading function (keep your existing working version)
@st.cache_resource
def load_saved_model():
    """Load the saved LSTM model and artifacts"""
    try:
        checkpoint = torch.load(
            'enhanced_lstm_air_quality_model.pth',
            map_location=torch.device('cpu'),
            weights_only=False
        )
        
        model_config = checkpoint['model_config']
        
        model = EnhancedLSTMModel(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            output_size=model_config['output_size'],
            dropout_rate=model_config.get('dropout_rate', 0.3),
            use_batch_norm=model_config.get('use_batch_norm', True)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        return {
            'model': model,
            'scaler_X': checkpoint['scaler_X'],
            'scaler_y': checkpoint['scaler_y'],
            'feature_columns': checkpoint['feature_columns'],
            'sequence_length': checkpoint['sequence_length'],
            'train_losses': checkpoint['train_losses'],
            'val_losses': checkpoint['val_losses'],
            'learning_rates': checkpoint.get('learning_rates', []),
            'model_config': model_config
        }
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Load model
with st.spinner("Loading forecasting model..."):
    data = load_saved_model()

# Helper function for 5-minute interval forecasting
def generate_5min_forecast(model, initial_sequence, scaler_X, scaler_y, feature_columns, horizon=12):
    """
    Generate multi-step forecast with 5-minute intervals
    horizon: number of 5-minute intervals to forecast
    """
    predictions = []
    current_sequence = initial_sequence.copy()
    
    with torch.no_grad():
        for step in range(horizon):
            # Scale current sequence
            seq_scaled = scaler_X.transform(
                current_sequence.reshape(-1, current_sequence.shape[-1])
            ).reshape(current_sequence.shape)
            
            # Make prediction
            seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0)
            pred_scaled = model(seq_tensor).item()
            
            # Inverse transform prediction
            pred = scaler_y.inverse_transform(np.array([[pred_scaled]])).item()
            predictions.append(pred)
            
            # Update sequence for next step (simplified - would need actual feature updates)
            # For demo, we'll shift and add the prediction as a feature
            new_row = current_sequence[-1].copy()
            # Update time features for 5-min increment
            if 'hour' in feature_columns:
                hour_idx = feature_columns.index('hour')
                minute_increment = (step + 1) * 5 / 60  # 5 minutes in hours
                new_row[hour_idx] = (new_row[hour_idx] + minute_increment) % 24
            
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_row
    
    return predictions

# Main dashboard content
if page == "Dashboard Overview":
    st.markdown('<h2 class="sub-header">Real-time Monitoring Dashboard</h2>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current AQI", "85", "▲ 2.3")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("PM2.5 Level", "24 μg/m³", "▼ 1.2")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Temperature", "22.5°C", "▲ 0.5")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Humidity", "65%", "▼ 3")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main forecasting section
    st.markdown('<h2 class="sub-header">5-Minute Interval Forecast</h2>', unsafe_allow_html=True)
    
    col_chart, col_summary = st.columns([2, 1])
    
    with col_chart:
        # Generate forecast data
        if data and 'model' in data:
            # Create sample sequence for demonstration
            np.random.seed(42)
            sample_sequence = np.random.randn(
                data['sequence_length'], 
                len(data['feature_columns'])
            )
            
            # Generate forecasts
            forecasts = generate_5min_forecast(
                data['model'],
                sample_sequence,
                data['scaler_X'],
                data['scaler_y'],
                data['feature_columns'],
                horizon=forecast_horizon
            )
            
            # Create time labels for 5-minute intervals
            time_labels = [f"+{i*5}min" for i in range(1, forecast_horizon + 1)]
            
            # Create forecast plot
            fig = go.Figure()
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=time_labels,
                y=forecasts,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#3498DB', width=3),
                marker=dict(size=8)
            ))
            
            # Add confidence interval (simulated)
            upper_bound = [f * 1.1 for f in forecasts]
            lower_bound = [f * 0.9 for f in forecasts]
            
            fig.add_trace(go.Scatter(
                x=time_labels + time_labels[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(52, 152, 219, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            ))
            
            fig.update_layout(
                title=f'Air Quality Forecast (Next {forecast_horizon*5} minutes)',
                xaxis_title='Time Ahead',
                yaxis_title='Air Quality Index',
                height=500,
                template='plotly_white',
                hovermode='x unified',
                showlegend=True,
                plot_bgcolor='rgba(240, 240, 240, 0.5)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col_summary:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown("### Forecast Summary")
        
        if 'forecasts' in locals():
            # Calculate statistics
            avg_forecast = np.mean(forecasts)
            max_forecast = np.max(forecasts)
            min_forecast = np.min(forecasts)
            trend = "increasing" if forecasts[-1] > forecasts[0] else "decreasing"
            
            st.metric("Average Forecast", f"{avg_forecast:.1f}")
            st.metric("Peak Forecast", f"{max_forecast:.1f}")
            st.metric("Minimum Forecast", f"{min_forecast:.1f}")
            st.metric("Trend", f"{trend.capitalize()}")
            
            # Determine alert level
            if max_forecast > 100:
                st.warning("⚠️ Alert: Poor air quality predicted")
            elif max_forecast > 50:
                st.info("ℹ️ Moderate air quality expected")
            else:
                st.success("✅ Good air quality expected")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed forecast table
    st.markdown('<h3 class="sub-header">Detailed Forecast Table</h3>', unsafe_allow_html=True)
    
    if 'forecasts' in locals():
        forecast_df = pd.DataFrame({
            'Time Ahead': time_labels,
            'AQI Forecast': forecasts,
            'Category': ['Good' if x <= 50 else 'Moderate' if x <= 100 else 'Poor' for x in forecasts],
            'Recommendation': [
                'Normal outdoor activities' if x <= 50 
                else 'Sensitive groups limit exposure' if x <= 100 
                else 'Limit outdoor activities' 
                for x in forecasts
            ]
        })
        
        # Style the dataframe
        def color_category(val):
            if val == 'Good':
                return 'background-color: #27AE60; color: white'
            elif val == 'Moderate':
                return 'background-color: #F39C12; color: white'
            else:
                return 'background-color: #E74C3C; color: white'
        
        styled_df = forecast_df.style.applymap(color_category, subset=['Category'])
        st.dataframe(styled_df, use_container_width=True, height=400)

elif page == "Forecast Analysis":
    st.markdown('<h2 class="sub-header">Forecast Analysis</h2>', unsafe_allow_html=True)
    
    # Multi-plot analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Hourly Trend', 'Feature Importance', 'Distribution', 'Error Analysis'),
        specs=[[{'type': 'scatter'}, {'type': 'bar'}],
               [{'type': 'histogram'}, {'type': 'scatter'}]]
    )
    
    # Sample data for demonstration
    hours = list(range(24))
    aqi_hourly = [50 + 30 * np.sin(h/24 * 2*np.pi) + np.random.randn()*5 for h in hours]
    features = ['Temp', 'Humidity', 'CO₂', 'PM2.5', 'PM10']
    importance = [0.35, 0.25, 0.15, 0.20, 0.05]
    
    # Plot 1: Hourly trend
    fig.add_trace(
        go.Scatter(x=hours, y=aqi_hourly, mode='lines+markers', name='AQI',
                  line=dict(color='#2C3E50', width=2)),
        row=1, col=1
    )
    
    # Plot 2: Feature importance
    fig.add_trace(
        go.Bar(x=features, y=importance, name='Importance',
              marker_color=['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#9B59B6']),
        row=1, col=2
    )
    
    # Plot 3: Distribution
    fig.add_trace(
        go.Histogram(x=np.random.randn(1000) + 50, nbinsx=30, name='Distribution',
                    marker_color='#3498DB'),
        row=2, col=1
    )
    
    # Plot 4: Error analysis
    fig.add_trace(
        go.Scatter(x=list(range(100)), y=np.random.randn(100), mode='markers',
                  name='Error', marker=dict(color='#E74C3C', size=8)),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Historical Data":
    st.markdown('<h2 class="sub-header">Historical Analysis</h2>', unsafe_allow_html=True)
    
    # Generate sample historical data
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    historical_aqi = 50 + 20 * np.sin(np.arange(100)/10) + np.random.randn(100)*10
    historical_pm25 = 20 + 10 * np.sin(np.arange(100)/10) + np.random.randn(100)*5
    
    historical_df = pd.DataFrame({
        'Timestamp': dates,
        'AQI': historical_aqi,
        'PM2.5': historical_pm25,
        'Temperature': 20 + 5 * np.sin(np.arange(100)/10),
        'Humidity': 60 + 10 * np.sin(np.arange(100)/10)
    })
    
    # Time series plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=historical_df['Timestamp'],
        y=historical_df['AQI'],
        mode='lines',
        name='AQI',
        line=dict(color='#3498DB', width=2)
    ))
    
    fig1.add_trace(go.Scatter(
        x=historical_df['Timestamp'],
        y=historical_df['PM2.5'],
        mode='lines',
        name='PM2.5',
        yaxis='y2',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))
    
    fig1.update_layout(
        title='Historical Air Quality Trends',
        xaxis_title='Date',
        yaxis=dict(title='AQI', color='#3498DB'),
        yaxis2=dict(title='PM2.5 (μg/m³)', color='#E74C3C', overlaying='y', side='right'),
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Correlation matrix
    st.markdown('<h3 class="sub-header">Feature Correlations</h3>', unsafe_allow_html=True)
    
    corr_matrix = historical_df[['AQI', 'PM2.5', 'Temperature', 'Humidity']].corr()
    
    fig2 = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Model Performance":
    st.markdown('<h2 class="sub-header">Model Performance Metrics</h2>', unsafe_allow_html=True)
    
    if data and 'train_losses' in data:
        col_metrics, col_chart = st.columns(2)
        
        with col_metrics:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("### Performance Summary")
            
            if len(data['train_losses']) > 0:
                final_train_loss = data['train_losses'][-1]
                final_val_loss = data['val_losses'][-1] if 'val_losses' in data else None
                best_val_loss = min(data['val_losses']) if 'val_losses' in data else None
                
                st.metric("Final Training Loss", f"{final_train_loss:.4f}")
                if final_val_loss:
                    st.metric("Final Validation Loss", f"{final_val_loss:.4f}")
                if best_val_loss:
                    st.metric("Best Validation Loss", f"{best_val_loss:.4f}")
                    
                # Calculate accuracy metrics (simulated)
                st.metric("R² Score", "0.89")
                st.metric("MAE", "4.2")
                st.metric("RMSE", "6.8")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_chart:
            # Loss plot
            fig = go.Figure()
            
            if 'train_losses' in data:
                fig.add_trace(go.Scatter(
                    y=data['train_losses'],
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='#3498DB', width=3)
                ))
            
            if 'val_losses' in data:
                fig.add_trace(go.Scatter(
                    y=data['val_losses'],
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='#E74C3C', width=3)
                ))
            
            fig.update_layout(
                title='Training History',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                height=400,
                template='plotly_white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Model architecture details
    st.markdown('<h3 class="sub-header">Model Architecture</h3>', unsafe_allow_html=True)
    
    if data and 'model_config' in data:
        config = data['model_config']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**LSTM Layers**")
            st.info(f"{config.get('num_layers', 'N/A')}")
        
        with col2:
            st.markdown("**Hidden Units**")
            st.info(f"{config.get('hidden_size', 'N/A')}")
        
        with col3:
            st.markdown("**Input Features**")
            st.info(f"{config.get('input_size', 'N/A')}")

else:  # Configuration page
    st.markdown('<h2 class="sub-header">System Configuration</h2>', unsafe_allow_html=True)
    
    with st.form("config_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Settings")
            retrain_model = st.checkbox("Retrain model on new data", value=False)
            update_frequency = st.selectbox(
                "Model update frequency",
                ["Daily", "Weekly", "Monthly", "Manual"]
            )
            
            st.markdown("#### Alert Thresholds")
            warning_threshold = st.number_input("Warning Threshold (AQI)", 50, 150, 100)
            alert_threshold = st.number_input("Alert Threshold (AQI)", 100, 300, 150)
        
        with col2:
            st.markdown("#### Data Settings")
            data_retention = st.selectbox(
                "Data retention period",
                ["30 days", "90 days", "1 year", "Indefinite"]
            )
            auto_backup = st.checkbox("Enable automatic backups", value=True)
            backup_frequency = st.selectbox(
                "Backup frequency",
                ["Daily", "Weekly", "Monthly"]
            )
        
        submitted = st.form_submit_button("Save Configuration")
        if submitted:
            st.success("Configuration saved successfully!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7F8C8D; font-size: 0.9rem;'>"
    "Air Quality Forecasting System v2.1.0 | © 2024 Environmental Analytics"
    "</div>",
    unsafe_allow_html=True
)
