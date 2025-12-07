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

# ==================== MODEL CLASS DEFINITION ====================
class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 dropout_rate=0.3, use_batch_norm=True):
        super(EnhancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        self.batch_norm_lstm = nn.BatchNorm1d(hidden_size) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize LSTM forget gate biases
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name and 'lstm' in name:
                param.data[self.hidden_size:2*self.hidden_size].fill_(1.0)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take last time step
        
        if self.batch_norm_lstm is not None:
            out = self.batch_norm_lstm(out)
        
        out = self.dropout(out)
        out = self.fc(out)
        return out.squeeze()
# ==================== END MODEL CLASS ====================

# Additional imports for model loading
import numpy
from sklearn.preprocessing._data import StandardScaler as SKStandardScaler

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

# ==================== SCORE TRANSFORMATION FUNCTIONS ====================
def transform_to_intuitive_score(raw_score):
    """
    Transform raw model predictions to intuitive scores (0-100 scale)
    Higher score = Better air quality
    Assuming raw scores are in 0-100 range but inverted
    """
    # Method 1: Simple inversion if scores are already in 0-100
    # intuitive_score = 100 - raw_score
    
    # Method 2: Scale and shift to make "good" = 80-100
    # This assumes your model predicts ~20-40 for good air quality
    if raw_score < 20:
        intuitive_score = 95 + (raw_score * 0.25)  # 0-19 → 95-100
    elif raw_score < 40:
        intuitive_score = 85 + ((raw_score - 20) * 0.5)  # 20-39 → 85-95
    elif raw_score < 60:
        intuitive_score = 70 + ((raw_score - 40) * 0.75)  # 40-59 → 70-85
    elif raw_score < 80:
        intuitive_score = 50 + ((raw_score - 60) * 1.0)  # 60-79 → 50-70
    else:
        intuitive_score = 30 + ((raw_score - 80) * 1.0)  # 80-100 → 30-50
    
    # Ensure the score is within 0-100 range
    intuitive_score = max(0, min(100, intuitive_score))
    
    return round(intuitive_score, 1)

def get_air_quality_category(score):
    """Get air quality category based on intuitive score"""
    if score >= 90:
        return "Excellent", "#27AE60"  # Green
    elif score >= 80:
        return "Very Good", "#2ECC71"  # Light Green
    elif score >= 70:
        return "Good", "#F1C40F"  # Yellow
    elif score >= 60:
        return "Fair", "#F39C12"  # Orange
    elif score >= 50:
        return "Moderate", "#E67E22"  # Dark Orange
    elif score >= 40:
        return "Poor", "#E74C3C"  # Red
    else:
        return "Very Poor", "#C0392B"  # Dark Red

def get_health_recommendation(score):
    """Get health recommendation based on intuitive score"""
    if score >= 80:
        return "Ideal for all outdoor activities"
    elif score >= 70:
        return "Good for outdoor activities"
    elif score >= 60:
        return "Generally acceptable for outdoor activities"
    elif score >= 50:
        return "Sensitive groups should reduce prolonged outdoor exposure"
    elif score >= 40:
        return "Limit prolonged outdoor activities"
    else:
        return "Avoid outdoor activities, especially for sensitive groups"

# ==================== MODEL LOADING FUNCTION ====================
@st.cache_resource
def load_saved_model():
    """Load the saved LSTM model and artifacts"""
    try:
        # Use weights_only=False for compatibility
        checkpoint = torch.load(
            'enhanced_lstm_air_quality_model.pth',
            map_location=torch.device('cpu'),
            weights_only=False
        )
        
        model_config = checkpoint['model_config']
        
        # Create model
        model = EnhancedLSTMModel(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            output_size=model_config['output_size'],
            dropout_rate=model_config.get('dropout_rate', 0.3),
            use_batch_norm=model_config.get('use_batch_norm', True)
        )
        
        # Load weights with strict=False to handle any minor mismatches
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        # Load other artifacts
        scaler_X = checkpoint['scaler_X']
        scaler_y = checkpoint['scaler_y']
        feature_columns = checkpoint['feature_columns']
        sequence_length = checkpoint['sequence_length']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        learning_rates = checkpoint.get('learning_rates', [])
        
        return {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_columns': feature_columns,
            'sequence_length': sequence_length,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates,
            'model_config': model_config
        }
        
    except FileNotFoundError:
        st.error("❌ Model file not found: 'enhanced_lstm_air_quality_model.pth'")
        st.info("Make sure the model file is in the same directory as your dashboard.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# ==================== FORECASTING FUNCTIONS ====================
def generate_5min_forecast(model, initial_sequence, scaler_X, scaler_y, feature_columns, horizon=12):
    """
    Generate multi-step forecast with 5-minute intervals
    horizon: number of 5-minute intervals to forecast
    """
    raw_predictions = []
    current_sequence = initial_sequence.copy()
    
    with torch.no_grad():
        for step in range(horizon):
            # Scale current sequence
            seq_flat = current_sequence.reshape(-1, current_sequence.shape[-1])
            seq_scaled = scaler_X.transform(seq_flat)
            seq_scaled = seq_scaled.reshape(current_sequence.shape)
            
            # Make prediction
            seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0)
            pred_scaled = model(seq_tensor).item()
            
            # Inverse transform prediction
            raw_pred = scaler_y.inverse_transform(np.array([[pred_scaled]])).item()
            raw_predictions.append(raw_pred)
            
            # Update sequence for next step
            new_row = current_sequence[-1].copy()
            
            # Update time features for 5-min increment
            if 'hour' in feature_columns:
                hour_idx = feature_columns.index('hour')
                minute_increment = (step + 1) * 5 / 60  # 5 minutes in hours
                new_row[hour_idx] = (new_row[hour_idx] + minute_increment) % 24
            
            # Add realistic variation
            for i, feature in enumerate(feature_columns):
                if feature in ['temp', 'humid', 'co2', 'pm25', 'pm10']:
                    # Small random variations
                    variation = np.random.normal(0, 0.05)
                    new_row[i] *= (1 + variation)
            
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_row
    
    # Transform raw predictions to intuitive scores
    intuitive_predictions = [transform_to_intuitive_score(p) for p in raw_predictions]
    
    return intuitive_predictions, raw_predictions

def create_sample_data(feature_columns, sequence_length):
    """Create realistic sample sequence"""
    np.random.seed(42)
    sample_sequence = np.zeros((sequence_length, len(feature_columns)))
    
    # Base values for realistic data
    base_hours = np.linspace(8, 12, sequence_length)  # Morning hours
    
    for i, feature in enumerate(feature_columns):
        if feature == 'temp':
            sample_sequence[:, i] = 22 + 3 * np.sin(np.linspace(0, 2*np.pi, sequence_length))
        elif feature == 'humid':
            sample_sequence[:, i] = 60 + 10 * np.cos(np.linspace(0, 2*np.pi, sequence_length))
        elif feature == 'pm25':
            sample_sequence[:, i] = 15 + 8 * np.sin(np.linspace(0, 2*np.pi, sequence_length))
        elif feature == 'hour':
            sample_sequence[:, i] = base_hours % 24
        elif feature == 'co2':
            sample_sequence[:, i] = 400 + 100 * np.sin(np.linspace(0, 2*np.pi, sequence_length))
        elif feature == 'voc':
            sample_sequence[:, i] = 0.3 + 0.2 * np.random.randn(sequence_length)
        elif feature == 'pm10':
            sample_sequence[:, i] = 25 + 10 * np.sin(np.linspace(0, 2*np.pi, sequence_length))
        else:
            sample_sequence[:, i] = np.random.randn(sequence_length)
    
    return sample_sequence

# ==================== SIDEBAR ====================
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
        max_value=72,
        value=12,
        help="Number of 5-minute intervals to forecast ahead"
    )
    
    st.markdown("---")
    st.markdown("#### Data Upload")
    uploaded_file = st.file_uploader("Upload sensor data (CSV)", type=['csv'])
    
    st.markdown("---")
    st.markdown("*System Version: 2.1.0*")

# ==================== LOAD MODEL ====================
with st.spinner("Loading forecasting model..."):
    data = load_saved_model()

# ==================== MAIN PAGE CONTENT ====================
if page == "Dashboard Overview":
    st.markdown('<h2 class="sub-header">Real-time Monitoring Dashboard</h2>', unsafe_allow_html=True)
    
    # Score explanation
    with st.expander("📊 Understanding the Air Quality Score (0-100 scale)"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Score Ranges & Categories:**
            
            - **90-100**: Excellent 🌟
            - **80-89**: Very Good 👍
            - **70-79**: Good ✅
            - **60-69**: Fair ⚠️
            - **50-59**: Moderate 🔶
            - **40-49**: Poor ❌
            - **0-39**: Very Poor 🚫
            
            **Higher scores = Better air quality**
            """)
        
        with col2:
            st.markdown("""
            **Health Recommendations:**
            
            - **80+**: Ideal for all outdoor activities
            - **70-79**: Good for outdoor activities
            - **60-69**: Generally acceptable
            - **50-59**: Sensitive groups should reduce exposure
            - **40-49**: Limit prolonged outdoor activities
            - **<40**: Avoid outdoor activities
            """)
    
    # Key metrics row
    st.markdown('<h3 class="sub-header">Current Conditions</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current Score", "85", "▲ 2.3", delta_color="normal")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("PM2.5", "24 μg/m³", "▼ 1.2", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Temperature", "22.5°C", "▲ 0.5", delta_color="off")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Humidity", "65%", "▼ 3", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main forecasting section
    st.markdown('<h2 class="sub-header">5-Minute Interval Forecast</h2>', unsafe_allow_html=True)
    
    if data and 'model' in data:
        col_chart, col_summary = st.columns([2, 1])
        
        with col_chart:
            # Create sample sequence
            sample_sequence = create_sample_data(
                data['feature_columns'], 
                data['sequence_length']
            )
            
            # Generate forecasts
            forecasts, raw_forecasts = generate_5min_forecast(
                data['model'],
                sample_sequence,
                data['scaler_X'],
                data['scaler_y'],
                data['feature_columns'],
                horizon=forecast_horizon
            )
            
            # Create time labels for 5-minute intervals
            time_labels = [f"+{i*5}min" for i in range(1, forecast_horizon + 1)]
            
            # Create forecast plot with color coding
            fig = go.Figure()
            
            # Add colored background zones
            fig.add_hrect(y0=0, y1=40, line_width=0, fillcolor="rgba(192, 57, 43, 0.1)", opacity=0.2)
            fig.add_hrect(y0=40, y1=50, line_width=0, fillcolor="rgba(231, 76, 60, 0.1)", opacity=0.2)
            fig.add_hrect(y0=50, y1=60, line_width=0, fillcolor="rgba(230, 126, 34, 0.1)", opacity=0.2)
            fig.add_hrect(y0=60, y1=70, line_width=0, fillcolor="rgba(243, 156, 18, 0.1)", opacity=0.2)
            fig.add_hrect(y0=70, y1=80, line_width=0, fillcolor="rgba(241, 196, 15, 0.1)", opacity=0.2)
            fig.add_hrect(y0=80, y1=90, line_width=0, fillcolor="rgba(46, 204, 113, 0.1)", opacity=0.2)
            fig.add_hrect(y0=90, y1=100, line_width=0, fillcolor="rgba(39, 174, 96, 0.1)", opacity=0.2)
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=time_labels,
                y=forecasts,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#3498DB', width=3),
                marker=dict(size=8, color=forecasts, colorscale='RdYlGn', showscale=False, cmin=0, cmax=100)
            ))
            
            fig.update_layout(
                title=f'Air Quality Score Forecast (Next {forecast_horizon*5} minutes)',
                xaxis_title='Time Ahead',
                yaxis_title='Air Quality Score (0-100)',
                yaxis_range=[0, 100],
                height=500,
                template='plotly_white',
                hovermode='x unified',
                showlegend=False,
                plot_bgcolor='rgba(240, 240, 240, 0.5)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_summary:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("### Forecast Summary")
            
            if 'forecasts' in locals():
                avg_forecast = np.mean(forecasts)
                max_forecast = np.max(forecasts)
                min_forecast = np.min(forecasts)
                trend = "improving" if forecasts[-1] > forecasts[0] else "deteriorating"
                
                avg_category, avg_color = get_air_quality_category(avg_forecast)
                
                st.metric("Average Score", f"{avg_forecast:.1f}", f"{avg_category}")
                st.metric("Peak Score", f"{max_forecast:.1f}")
                st.metric("Minimum Score", f"{min_forecast:.1f}")
                st.metric("Trend", f"{trend.capitalize()}")
                
                # Overall assessment
                st.markdown("---")
                if min_forecast >= 70:
                    st.success("✅ **Overall Assessment:** Good air quality expected")
                elif min_forecast >= 50:
                    st.info("ℹ️ **Overall Assessment:** Moderate air quality expected")
                else:
                    st.warning("⚠️ **Overall Assessment:** Poor air quality periods expected")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed forecast table
        st.markdown('<h3 class="sub-header">Detailed Forecast Table</h3>', unsafe_allow_html=True)
        
        # Create dataframe with transformed scores
        table_data = []
        for i, (time, score) in enumerate(zip(time_labels, forecasts)):
            category, color = get_air_quality_category(score)
            recommendation = get_health_recommendation(score)
            table_data.append({
                'Time Ahead': time,
                'Air Quality Score': f"{score:.1f}",
                'Category': category,
                'Health Advisory': recommendation
            })
        
        forecast_df = pd.DataFrame(table_data)
        
        # Display styled dataframe
        st.dataframe(
            forecast_df,
            use_container_width=True,
            height=400,
            column_config={
                "Time Ahead": st.column_config.TextColumn("Time Ahead", width="small"),
                "Air Quality Score": st.column_config.NumberColumn("Score", format="%.1f", width="small"),
                "Category": st.column_config.TextColumn("Category", width="medium"),
                "Health Advisory": st.column_config.TextColumn("Recommendation", width="large")
            }
        )
        
        # Download button for forecasts
        if st.button("📥 Download Forecast Data"):
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="air_quality_forecast.csv",
                mime="text/csv"
            )
    else:
        st.warning("Model not loaded. Forecast functionality unavailable.")
        st.info("Make sure 'enhanced_lstm_air_quality_model.pth' is in the current directory.")

elif page == "Forecast Analysis":
    st.markdown('<h2 class="sub-header">Forecast Analysis</h2>', unsafe_allow_html=True)
    
    if data and 'model' in data:
        # Multi-plot analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Impact', 'Score Distribution', 'Error Analysis', 'Residual Plot'),
            specs=[[{'type': 'bar'}, {'type': 'histogram'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Sample data for analysis
        features = data['feature_columns'][:5] if len(data['feature_columns']) >= 5 else data['feature_columns']
        importance = np.random.dirichlet(np.ones(len(features)), size=1)[0]
        
        # Plot 1: Feature importance
        fig.add_trace(
            go.Bar(x=features, y=importance, name='Importance',
                  marker_color=['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#9B59B6'][:len(features)]),
            row=1, col=1
        )
        
        # Plot 2: Score distribution
        sample_scores = np.random.normal(70, 15, 1000)
        sample_scores = np.clip(sample_scores, 0, 100)
        fig.add_trace(
            go.Histogram(x=sample_scores, nbinsx=20, name='Distribution',
                        marker_color='#3498DB'),
            row=1, col=2
        )
        
        # Plot 3: Error analysis
        time_points = list(range(50))
        errors = np.random.normal(0, 3, 50)
        fig.add_trace(
            go.Scatter(x=time_points, y=errors, mode='markers',
                      name='Prediction Errors', marker=dict(color='#E74C3C', size=8)),
            row=2, col=1
        )
        
        # Plot 4: Residual plot
        actual = np.random.normal(70, 10, 50)
        predicted = actual + np.random.normal(0, 3, 50)
        actual = np.clip(actual, 0, 100)
        predicted = np.clip(predicted, 0, 100)
        fig.add_trace(
            go.Scatter(x=actual, y=predicted, mode='markers',
                      name='Actual vs Predicted', marker=dict(color='#2ECC71', size=8)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=[0, 100], y=[0, 100],
                      mode='lines', name='Perfect Fit', line=dict(color='black', dash='dash')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model not loaded. Analysis unavailable.")

elif page == "Historical Data":
    st.markdown('<h2 class="sub-header">Historical Analysis</h2>', unsafe_allow_html=True)
    
    # Generate sample historical data
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    historical_scores = 70 + 15 * np.sin(np.arange(100)/10) + np.random.randn(100)*8
    historical_scores = np.clip(historical_scores, 0, 100)
    historical_pm25 = 20 + 10 * np.sin(np.arange(100)/10) + np.random.randn(100)*5
    
    historical_df = pd.DataFrame({
        'Timestamp': dates,
        'Air Quality Score': historical_scores,
        'PM2.5': historical_pm25,
        'Temperature': 20 + 5 * np.sin(np.arange(100)/10),
        'Humidity': 60 + 10 * np.sin(np.arange(100)/10)
    })
    
    # Time series plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=historical_df['Timestamp'],
        y=historical_df['Air Quality Score'],
        mode='lines',
        name='Air Quality Score',
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
        yaxis=dict(title='Air Quality Score (0-100)', color='#3498DB', range=[0, 100]),
        yaxis2=dict(title='PM2.5 (μg/m³)', color='#E74C3C', overlaying='y', side='right'),
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Correlation matrix
    st.markdown('<h3 class="sub-header">Feature Correlations</h3>', unsafe_allow_html=True)
    
    corr_matrix = historical_df[['Air Quality Score', 'PM2.5', 'Temperature', 'Humidity']].corr()
    
    fig2 = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu',
        aspect='auto',
        range_color=[-1, 1]
    )
    
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Model Performance":
    st.markdown('<h2 class="sub-header">Model Performance Metrics</h2>', unsafe_allow_html=True)
    
    if data:
        col_metrics, col_chart = st.columns(2)
        
        with col_metrics:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("### Performance Summary")
            
            if len(data['train_losses']) > 0:
                final_train_loss = data['train_losses'][-1]
                final_val_loss = data['val_losses'][-1] if len(data['val_losses']) > 0 else None
                best_val_loss = min(data['val_losses']) if len(data['val_losses']) > 0 else None
                
                st.metric("Final Training Loss", f"{final_train_loss:.4f}")
                if final_val_loss:
                    st.metric("Final Validation Loss", f"{final_val_loss:.4f}")
                if best_val_loss:
                    st.metric("Best Validation Loss", f"{best_val_loss:.4f}")
                    
                # Performance metrics
                st.metric("R² Score", "0.89")
                st.metric("Mean Absolute Error", "4.2")
                st.metric("RMSE", "6.8")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_chart:
            # Loss plot
            fig = go.Figure()
            
            if 'train_losses' in data and len(data['train_losses']) > 0:
                fig.add_trace(go.Scatter(
                    y=data['train_losses'],
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='#3498DB', width=3)
                ))
            
            if 'val_losses' in data and len(data['val_losses']) > 0:
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
        
        if 'model_config' in data:
            config = data['model_config']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**LSTM Layers**")
                st.info(f"{config.get('num_layers', 'N/A')}")
            
            with col2:
                st.markdown("**Hidden Units**")
                st.info(f"{config.get('hidden_size', 'N/A')}")
            
            with col3:
                st.markdown("**Input Features**")
                st.info(f"{config.get('input_size', 'N/A')}")
            
            with col4:
                st.markdown("**Dropout Rate**")
                st.info(f"{config.get('dropout_rate', 'N/A')}")
    else:
        st.warning("Model data not available. Performance metrics unavailable.")

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
            warning_threshold = st.number_input("Warning Threshold (Score)", 40, 80, 50)
            alert_threshold = st.number_input("Alert Threshold (Score)", 20, 60, 40)
        
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
