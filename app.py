import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Air Quality LSTM Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌫️ Air Quality LSTM Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/air-quality.png", width=80)
    st.markdown("## Navigation")
    page = st.radio(
        "Select Page",
        ["📊 Model Performance", "📈 Predictions", "🔮 Make Prediction", "📋 Model Info"]
    )
    
    st.markdown("---")
    st.markdown("### Settings")
    show_raw_data = st.checkbox("Show raw data", value=False)
    
    # File upload for predictions
    uploaded_file = st.file_uploader("Upload test data (CSV)", type=['csv'])
    
    st.markdown("---")
    st.markdown("**Note**: Using saved LSTM model for predictions")

# Load the saved model
@st.cache_resource
def load_saved_model():
    """Load the saved LSTM model and artifacts"""
    try:
        # Load the checkpoint
        checkpoint = torch.load('enhanced_lstm_air_quality_model.pth', 
                               map_location=torch.device('cpu'))
        
        # Define the model class (must match your training code)
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
                
                self.batch_norm = nn.BatchNorm1d(hidden_size) if use_batch_norm else None
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
                out = out[:, -1, :]
                
                if self.batch_norm is not None:
                    out = self.batch_norm(out)
                
                out = self.dropout(out)
                out = self.fc(out)
                return out.squeeze()
        
        # Get model config
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
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
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
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model
data = load_saved_model()

if data is None:
    st.warning("Could not load model. Using demo mode.")
    # Create demo data
    data = {
        'train_losses': np.random.randn(100).cumsum() + 10,
        'val_losses': np.random.randn(100).cumsum() + 12,
        'learning_rates': np.linspace(0.001, 0.0001, 100),
        'feature_columns': ['temp', 'humid', 'co2', 'voc', 'pm25', 'pm10', 
                           'hour', 'day', 'month', 'year', 'dayofweek'],
        'sequence_length': 24
    }

# Page 1: Model Performance
if page == "📊 Model Performance":
    st.header("Model Training Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Training loss plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=data['train_losses'],
            mode='lines',
            name='Training Loss',
            line=dict(color='blue', width=3)
        ))
        fig.add_trace(go.Scatter(
            y=data['val_losses'],
            mode='lines',
            name='Validation Loss',
            line=dict(color='red', width=3)
        ))
        
        # Highlight best epoch
        if len(data['val_losses']) > 0:
            best_epoch = np.argmin(data['val_losses'])
            best_loss = data['val_losses'][best_epoch]
            fig.add_trace(go.Scatter(
                x=[best_epoch],
                y=[best_loss],
                mode='markers',
                name=f'Best: {best_loss:.4f}',
                marker=dict(color='green', size=15)
            ))
        
        fig.update_layout(
            title='Training and Validation Loss',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Learning rate plot
        if len(data.get('learning_rates', [])) > 0:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                y=data['learning_rates'],
                mode='lines',
                name='Learning Rate',
                line=dict(color='green', width=3)
            ))
            fig2.update_layout(
                title='Learning Rate Schedule',
                xaxis_title='Epoch',
                yaxis_title='Learning Rate',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Metrics card
        st.markdown("### Model Metrics")
        if 'val_losses' in data and len(data['val_losses']) > 0:
            final_train_loss = data['train_losses'][-1]
            final_val_loss = data['val_losses'][-1]
            best_val_loss = min(data['val_losses'])
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Final Train Loss", f"{final_train_loss:.4f}")
                st.metric("Best Val Loss", f"{best_val_loss:.4f}")
            with col_m2:
                st.metric("Final Val Loss", f"{final_val_loss:.4f}")
                st.metric("Overfitting Gap", f"{final_val_loss - final_train_loss:.4f}")

# Page 2: Predictions
elif page == "📈 Predictions":
    st.header("Model Predictions")
    
    # Generate or load test data
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(test_df)} samples from uploaded file")
    else:
        # Generate synthetic test data
        st.info("Using synthetic test data. Upload a CSV file for real predictions.")
        
        # Generate realistic synthetic data
        np.random.seed(42)
        n_samples = 200
        
        # Create synthetic features
        test_data = {}
        for feature in data['feature_columns']:
            if feature == 'temp':
                test_data[feature] = 20 + 10 * np.random.randn(n_samples)
            elif feature == 'humid':
                test_data[feature] = 50 + 20 * np.random.randn(n_samples)
            elif feature == 'co2':
                test_data[feature] = 400 + 200 * np.random.randn(n_samples)
            elif feature == 'voc':
                test_data[feature] = 0.3 + 0.2 * np.random.randn(n_samples)
            elif feature == 'pm25':
                test_data[feature] = 15 + 10 * np.random.randn(n_samples)
            elif feature == 'pm10':
                test_data[feature] = 25 + 15 * np.random.randn(n_samples)
            elif feature == 'hour':
                test_data[feature] = np.random.randint(0, 24, n_samples)
            elif feature == 'day':
                test_data[feature] = np.random.randint(1, 31, n_samples)
            elif feature == 'month':
                test_data[feature] = np.random.randint(1, 13, n_samples)
            elif feature == 'year':
                test_data[feature] = 2024
            elif feature == 'dayofweek':
                test_data[feature] = np.random.randint(0, 7, n_samples)
        
        test_df = pd.DataFrame(test_data)
    
    if show_raw_data:
        st.subheader("Raw Test Data")
        st.dataframe(test_df.head(20))
    
    # Prepare sequences for prediction
    def create_sequences_from_df(df, feature_columns, sequence_length):
        """Create sequences from DataFrame"""
        X = df[feature_columns].values
        sequences = []
        
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i + sequence_length])
        
        return np.array(sequences)
    
    if st.button("Generate Predictions", type="primary"):
        with st.spinner("Making predictions..."):
            # Create sequences
            sequences = create_sequences_from_df(
                test_df, 
                data['feature_columns'], 
                data['sequence_length']
            )
            
            if len(sequences) == 0:
                st.error(f"Need at least {data['sequence_length']} samples for prediction")
            else:
                # Scale features
                if 'scaler_X' in data:
                    sequences_scaled = data['scaler_X'].transform(
                        sequences.reshape(-1, sequences.shape[-1])
                    ).reshape(sequences.shape)
                else:
                    sequences_scaled = sequences
                
                # Make predictions
                if 'model' in data:
                    model = data['model']
                    predictions = []
                    
                    with torch.no_grad():
                        for seq in sequences_scaled:
                            seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
                            pred = model(seq_tensor)
                            predictions.append(pred.item())
                    
                    # Inverse transform if scaler available
                    if 'scaler_y' in data:
                        predictions = data['scaler_y'].inverse_transform(
                            np.array(predictions).reshape(-1, 1)
                        ).flatten()
                else:
                    # Demo predictions
                    predictions = 50 + 30 * np.sin(np.arange(len(sequences)) * 0.1)
                    predictions += np.random.randn(len(sequences)) * 5
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'Sample': range(len(predictions)),
                    'Predicted_Score': predictions
                })
                
                # Display results
                st.subheader("Prediction Results")
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    # Summary statistics
                    st.metric("Mean Prediction", f"{predictions.mean():.2f}")
                    st.metric("Std Prediction", f"{predictions.std():.2f}")
                    st.metric("Min Prediction", f"{predictions.min():.2f}")
                    st.metric("Max Prediction", f"{predictions.max():.2f}")
                
                with col_res2:
                    # Histogram
                    fig = go.Figure(data=[go.Histogram(x=predictions, nbinsx=20)])
                    fig.update_layout(
                        title='Prediction Distribution',
                        xaxis_title='Air Quality Score',
                        yaxis_title='Frequency',
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Time series plot
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=results_df['Sample'],
                    y=results_df['Predicted_Score'],
                    mode='lines+markers',
                    name='Predictions',
                    line=dict(color='blue', width=2)
                ))
                fig2.update_layout(
                    title='Predictions Over Time',
                    xaxis_title='Sample Index',
                    yaxis_title='Air Quality Score',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

# Page 3: Make Prediction
elif page == "🔮 Make Prediction":
    st.header("Single Prediction Interface")
    
    st.info("Enter feature values for a single prediction (last 24 hours)")
    
    # Create input form for sequence
    if 'feature_columns' in data:
        # We need sequence_length sets of features
        sequence_length = data.get('sequence_length', 24)
        
        st.markdown(f"**Enter data for {sequence_length} time steps:**")
        
        # Create tabs for easier input
        tabs = st.tabs([f"T-{i}" for i in range(sequence_length-1, -1, -1)])
        
        sequence_data = []
        
        for i, tab in enumerate(tabs):
            with tab:
                st.markdown(f"**Time Step T-{sequence_length-1-i}**")
                row_data = {}
                
                # Create 2 columns for features
                col1, col2 = st.columns(2)
                
                features = data['feature_columns']
                half = len(features) // 2
                
                with col1:
                    for feature in features[:half]:
                        if feature in ['temp', 'humid', 'co2', 'voc', 'pm25', 'pm10']:
                            row_data[feature] = st.number_input(
                                f"{feature}",
                                value=25.0 if feature == 'temp' else 
                                      60.0 if feature == 'humid' else
                                      500.0 if feature == 'co2' else
                                      0.3 if feature == 'voc' else
                                      15.0 if feature == 'pm25' else 25.0,
                                key=f"{feature}_{i}"
                            )
                        elif feature == 'hour':
                            row_data[feature] = st.slider(
                                "Hour", 0, 23, 12, key=f"hour_{i}"
                            )
                
                with col2:
                    for feature in features[half:]:
                        if feature == 'day':
                            row_data[feature] = st.slider(
                                "Day", 1, 31, 15, key=f"day_{i}"
                            )
                        elif feature == 'month':
                            row_data[feature] = st.slider(
                                "Month", 1, 12, 3, key=f"month_{i}"
                            )
                        elif feature == 'year':
                            row_data[feature] = st.number_input(
                                "Year", 2023, 2025, 2024, key=f"year_{i}"
                            )
                        elif feature == 'dayofweek':
                            row_data[feature] = st.selectbox(
                                "Day of Week",
                                ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                 "Friday", "Saturday", "Sunday"],
                                index=4,
                                key=f"dayofweek_{i}"
                            )
                
                sequence_data.append(row_data)
        
        if st.button("Predict Next Hour", type="primary"):
            with st.spinner("Processing..."):
                # Convert to sequence
                sequence_df = pd.DataFrame(sequence_data)
                
                # Convert dayofweek to numeric
                if 'dayofweek' in sequence_df.columns:
                    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, 
                              "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
                    sequence_df['dayofweek'] = sequence_df['dayofweek'].map(day_map)
                
                # Ensure correct column order
                sequence_df = sequence_df[data['feature_columns']]
                
                # Scale and predict
                sequence_array = sequence_df.values
                
                if 'scaler_X' in data and 'model' in data:
                    # Scale
                    sequence_scaled = data['scaler_X'].transform(sequence_array)
                    sequence_scaled = sequence_scaled.reshape(1, sequence_length, -1)
                    
                    # Predict
                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(sequence_scaled)
                        prediction = data['model'](input_tensor).item()
                    
                    # Inverse scale
                    if 'scaler_y' in data:
                        prediction = data['scaler_y'].inverse_transform(
                            np.array([[prediction]])
                        ).item()
                else:
                    # Demo prediction
                    prediction = 50 + np.mean(sequence_df['temp']) * 0.5
                    prediction += np.mean(sequence_df['pm25']) * 0.8
                
                # Display result
                st.markdown("---")
                st.markdown("### Prediction Result")
                
                # Determine category
                if prediction <= 50:
                    category = "🟢 Excellent"
                    color = "green"
                elif prediction <= 100:
                    category = "🟡 Good"
                    color = "orange"
                elif prediction <= 150:
                    category = "🟠 Moderate"
                    color = "darkorange"
                elif prediction <= 200:
                    category = "🔴 Unhealthy"
                    color = "red"
                else:
                    category = "⚫ Hazardous"
                    color = "black"
                
                col_pred1, col_pred2 = st.columns(2)
                with col_pred1:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px;'>
                        <h3>Predicted Air Quality</h3>
                        <h1 style='color: {color}; font-size: 4rem;'>{prediction:.1f}</h1>
                        <h3 style='color: {color};'>{category}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_pred2:
                    st.markdown("### Feature Importance")
                    # Simple feature importance (for demo)
                    feature_importance = {
                        'Temperature': abs(sequence_df['temp'].mean() - 25),
                        'Humidity': abs(sequence_df['humid'].mean() - 60),
                        'PM2.5': sequence_df['pm25'].mean() * 0.1,
                        'CO2': sequence_df['co2'].mean() * 0.01,
                        'Time of Day': abs(sequence_df['hour'].mean() - 12) * 0.5
                    }
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(feature_importance.keys()),
                            y=list(feature_importance.values()),
                            marker_color='lightblue'
                        )
                    ])
                    fig.update_layout(
                        title='Factors Affecting Prediction',
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

# Page 4: Model Info
else:
    st.header("Model Information")
    
    if 'model_config' in data:
        model_config = data['model_config']
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("### Model Architecture")
            st.write(f"**Input Size**: {model_config.get('input_size', 'N/A')}")
            st.write(f"**Hidden Size**: {model_config.get('hidden_size', 'N/A')}")
            st.write(f"**Number of Layers**: {model_config.get('num_layers', 'N/A')}")
            st.write(f"**Output Size**: {model_config.get('output_size', 'N/A')}")
            st.write(f"**Dropout Rate**: {model_config.get('dropout_rate', 'N/A')}")
            st.write(f"**Batch Normalization**: {model_config.get('use_batch_norm', 'N/A')}")
        
        with col_info2:
            st.markdown("### Training Parameters")
            if 'training_params' in data.get('model_config', {}):
                params = data['model_config']['training_params']
                for key, value in params.items():
                    st.write(f"**{key}**: {value}")
            else:
                st.write("**Batch Size**: 32")
                st.write("**Learning Rate**: 0.001")
                st.write("**Weight Decay**: 0.01")
                st.write("**Sequence Length**: 24")
    
    st.markdown("---")
    st.markdown("### Feature Information")
    
    if 'feature_columns' in data:
        features_df = pd.DataFrame({
            'Feature': data['feature_columns'],
            'Description': [
                'Temperature in Celsius',
                'Humidity percentage',
                'CO2 concentration in ppm',
                'Volatile Organic Compounds',
                'PM2.5 particle concentration',
                'PM10 particle concentration',
                'Hour of day (0-23)',
                'Day of month',
                'Month (1-12)',
                'Year',
                'Day of week (0-6)'
            ],
            'Type': [
                'Continuous', 'Continuous', 'Continuous', 'Continuous',
                'Continuous', 'Continuous', 'Categorical', 'Categorical',
                'Categorical', 'Categorical', 'Categorical'
            ]
        })
        st.dataframe(features_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### Usage Instructions")
    
    with st.expander("How to use this dashboard"):
        st.markdown("""
        1. **Model Performance**: View training history and metrics
        2. **Predictions**: Upload CSV data or use synthetic data for batch predictions
        3. **Make Prediction**: Enter feature values for a single prediction
        4. **Model Info**: View model architecture and feature information
        
        **Note**: The model uses sequences of 24 time steps to predict the next hour's air quality.
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Air Quality LSTM Dashboard | Powered by Streamlit | Model: Enhanced LSTM"
    "</div>",
    unsafe_allow_html=True
)
