import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.serialization
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
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

# Define the model class first (must be identical to training)
# Define the model class (EXACTLY as it was during training)
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
        
        # IMPORTANT: Name must be batch_norm_lstm (not batch_norm)
        self.batch_norm_lstm = nn.BatchNorm1d(hidden_size) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize LSTM forget gate biases
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name and 'lstm' in name:
                # Fill forget gate biases with 1.0
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
        
# Add these imports at the top of your app.py
import numpy
from sklearn.preprocessing._data import StandardScaler as SKStandardScaler

# Then update your load_saved_model function:
@st.cache_resource
def load_saved_model():
    """Load the saved LSTM model and artifacts with safe_globals"""
    try:
        # First try with weights_only=True and safe_globals
        torch.serialization.add_safe_globals([
            SKStandardScaler,
            numpy._core.multiarray.scalar,
            numpy.core.multiarray.scalar,  # Try both versions
            numpy._core.multiarray._reconstruct,
            numpy.ndarray,
            numpy.dtype
        ])
        
        # Load with safe_globals context manager
        with torch.serialization.safe_globals([
            SKStandardScaler,
            numpy._core.multiarray.scalar,
            numpy.ndarray
        ]):
            checkpoint = torch.load(
                'enhanced_lstm_air_quality_model.pth',
                map_location=torch.device('cpu'),
                weights_only=True  # Safer option
            )
        
        # Get model config
        model_config = checkpoint['model_config']
        
        # Create model with CORRECT attribute names
        model = EnhancedLSTMModel(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            output_size=model_config['output_size'],
            dropout_rate=model_config.get('dropout_rate', 0.3),
            use_batch_norm=model_config.get('use_batch_norm', True)
        )
        
        # Load weights - should match now
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
        
        st.success("✅ Model loaded successfully with safe_globals!")
        
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
        st.warning(f"First attempt failed: {str(e)}. Trying alternative method...")
        
        try:
            # Fallback: Use weights_only=False (less secure, but works)
            checkpoint = torch.load(
                'enhanced_lstm_air_quality_model.pth',
                map_location=torch.device('cpu'),
                weights_only=False  # Less secure but works
            )
            
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
            
            st.success("✅ Model loaded successfully with weights_only=False!")
            st.info("Note: Using weights_only=False for compatibility. Only use with trusted models.")
            
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
            
        except Exception as e2:
            st.error(f"❌ Both loading methods failed: {str(e2)}")
            return None

# Alternative loading method using context manager
def load_model_safely():
    """Alternative method using safe_globals context manager"""
    try:
        from sklearn.preprocessing._data import StandardScaler as SKStandardScaler
        
        # Use safe_globals context manager
        with torch.serialization.safe_globals([SKStandardScaler]):
            checkpoint = torch.load(
                'enhanced_lstm_air_quality_model.pth',
                map_location=torch.device('cpu'),
                weights_only=True  # Can keep True with safe_globals
            )
        
        # Rest of the loading code...
        model_config = checkpoint['model_config']
        
        model = EnhancedLSTMModel(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            output_size=model_config['output_size'],
            dropout_rate=model_config.get('dropout_rate', 0.3),
            use_batch_norm=model_config.get('use_batch_norm', True)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Return the loaded data
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
        st.error(f"Alternative loading failed: {str(e)}")
        return None

# Try loading with the fixed method
st.sidebar.info("Loading model...")
data = load_saved_model()

if data is None:
    # Try alternative method
    data = load_model_safely()

if data is None:
    # Create demo data as fallback
    st.warning("Using demo mode with synthetic data. Real model could not be loaded.")
    data = {
        'train_losses': 100 * np.exp(-0.05 * np.arange(100)) + 5,
        'val_losses': 110 * np.exp(-0.04 * np.arange(100)) + 8,
        'learning_rates': np.linspace(0.001, 0.0001, 100),
        'feature_columns': ['temp', 'humid', 'co2', 'voc', 'pm25', 'pm10', 
                           'hour', 'day', 'month', 'year', 'dayofweek'],
        'sequence_length': 24,
        'model_config': {
            'input_size': 11,
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 1,
            'dropout_rate': 0.3,
            'use_batch_norm': True
        }
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
        
        if 'val_losses' in data:
            fig.add_trace(go.Scatter(
                y=data['val_losses'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red', width=3)
            ))
            
            # Highlight best epoch
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
        if 'learning_rates' in data and len(data['learning_rates']) > 0:
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
        feature_columns = data.get('feature_columns', 
            ['temp', 'humid', 'co2', 'voc', 'pm25', 'pm10', 
             'hour', 'day', 'month', 'year', 'dayofweek'])
        
        for feature in feature_columns:
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
            feature_columns = data.get('feature_columns', test_df.columns.tolist())
            sequence_length = data.get('sequence_length', 24)
            
            sequences = create_sequences_from_df(
                test_df, 
                feature_columns, 
                sequence_length
            )
            
            if len(sequences) == 0:
                st.error(f"Need at least {sequence_length} samples for prediction")
            else:
                # Check if we have a real model
                if 'model' in data and 'scaler_X' in data:
                    # Scale features
                    sequences_scaled = data['scaler_X'].transform(
                        sequences.reshape(-1, sequences.shape[-1])
                    ).reshape(sequences.shape)
                    
                    # Make predictions
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
    
    # Get feature columns and sequence length
    feature_columns = data.get('feature_columns', 
        ['temp', 'humid', 'co2', 'voc', 'pm25', 'pm10', 
         'hour', 'day', 'month', 'year', 'dayofweek'])
    sequence_length = data.get('sequence_length', 24)
    
    st.markdown(f"**Enter data for {sequence_length} time steps:**")
    
    # Create a simplified input form (just last time step for demo)
    st.markdown("### Enter Current Conditions")
    
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    with col1:
        input_data['temp'] = st.number_input("Temperature (°C)", 15.0, 35.0, 25.0, 0.5)
        input_data['humid'] = st.number_input("Humidity (%)", 30.0, 90.0, 60.0, 1.0)
        input_data['co2'] = st.number_input("CO₂ (ppm)", 300.0, 1500.0, 500.0, 10.0)
        input_data['voc'] = st.number_input("VOC", 0.1, 1.0, 0.3, 0.05)
    
    with col2:
        input_data['pm25'] = st.number_input("PM2.5 (µg/m³)", 5.0, 100.0, 15.0, 1.0)
        input_data['pm10'] = st.number_input("PM10 (µg/m³)", 10.0, 200.0, 25.0, 1.0)
        input_data['hour'] = st.slider("Hour", 0, 23, 14)
        input_data['dayofweek'] = st.selectbox("Day of Week", 
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=4)
    
    input_data['day'] = st.slider("Day", 1, 31, 15)
    input_data['month'] = st.slider("Month", 1, 12, 3)
    input_data['year'] = 2024
    
    if st.button("Predict Next Hour", type="primary"):
        with st.spinner("Processing..."):
            # Convert dayofweek to numeric
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, 
                      "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            input_data['dayofweek'] = day_map[input_data['dayofweek']]
            
            # Create a sequence (replicate current values for simplicity)
            sequence = []
            for i in range(sequence_length):
                # Create slight variations for demo
                seq_point = {}
                for key, value in input_data.items():
                    if key in ['temp', 'humid', 'co2', 'voc', 'pm25', 'pm10']:
                        # Add small variations
                        variation = value * (1 + np.random.normal(0, 0.05))
                        seq_point[key] = variation
                    elif key == 'hour':
                        # Adjust hour for each time step
                        hour_val = (input_data['hour'] - (sequence_length - 1 - i)) % 24
                        seq_point[key] = hour_val
                    else:
                        seq_point[key] = value
                sequence.append(seq_point)
            
            # Convert to array
            sequence_df = pd.DataFrame(sequence)
            sequence_array = sequence_df[feature_columns].values
            
            # Make prediction
            if 'model' in data and 'scaler_X' in data:
                try:
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
                    
                    prediction_source = "Real LSTM Model"
                    
                except Exception as e:
                    st.warning(f"Model prediction failed: {str(e)}. Using demo calculation.")
                    prediction = 50 + input_data['temp'] * 0.5 - input_data['humid'] * 0.2
                    prediction += input_data['pm25'] * 0.8 + input_data['co2'] * 0.01
                    prediction_source = "Demo Calculation"
            else:
                # Demo prediction
                prediction = 50 + input_data['temp'] * 0.5 - input_data['humid'] * 0.2
                prediction += input_data['pm25'] * 0.8 + input_data['co2'] * 0.01
                prediction_source = "Demo Calculation"
            
            # Display result
            st.markdown("---")
            st.markdown("### Prediction Result")
            
            # Determine category
            if prediction <= 50:
                category = "🟢 Excellent"
                color = "green"
                advice = "Air quality is excellent. Perfect for outdoor activities."
            elif prediction <= 100:
                category = "🟡 Good"
                color = "orange"
                advice = "Air quality is good. Suitable for outdoor activities."
            elif prediction <= 150:
                category = "🟠 Moderate"
                color = "darkorange"
                advice = "Air quality is moderate. Sensitive groups should limit exposure."
            elif prediction <= 200:
                category = "🔴 Unhealthy"
                color = "red"
                advice = "Air quality is unhealthy. Limit outdoor activities."
            else:
                category = "⚫ Hazardous"
                color = "black"
                advice = "Air quality is hazardous. Avoid outdoor activities."
            
            col_pred1, col_pred2 = st.columns(2)
            with col_pred1:
                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px;'>
                    <h3>Predicted Air Quality</h3>
                    <h1 style='color: {color}; font-size: 4rem;'>{prediction:.1f}</h1>
                    <h3 style='color: {color};'>{category}</h3>
                    <p>{advice}</p>
                    <p><small>Source: {prediction_source}</small></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_pred2:
                st.markdown("### Key Factors")
                factors = {
                    'Temperature': input_data['temp'] * 0.5,
                    'Humidity': -input_data['humid'] * 0.2,
                    'PM2.5': input_data['pm25'] * 0.8,
                    'CO₂': input_data['co2'] * 0.01,
                    'Time of Day': abs(input_data['hour'] - 12) * 0.3
                }
                
                # Create bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(factors.keys()),
                        y=list(factors.values()),
                        marker_color=['blue', 'green', 'red', 'purple', 'orange']
                    )
                ])
                fig.update_layout(
                    title='Feature Contributions',
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
            st.write(f"**Sequence Length**: {data.get('sequence_length', 24)}")
            st.write(f"**Training Epochs**: {len(data.get('train_losses', []))}")
            if 'val_losses' in data and len(data['val_losses']) > 0:
                st.write(f"**Best Validation Loss**: {min(data['val_losses']):.4f}")
    
    st.markdown("---")
    st.markdown("### Model Status")
    
    if 'model' in data:
        st.success("✅ Real LSTM model loaded successfully")
        st.info("The dashboard is using your trained LSTM model for predictions.")
    else:
        st.warning("⚠️ Demo mode active")
        st.info("The dashboard is using synthetic data and calculations. Make sure your model file ('enhanced_lstm_air_quality_model.pth') is in the same directory.")
    
    st.markdown("---")
    st.markdown("### Troubleshooting")
    
    with st.expander("Having issues loading the model?"):
        st.markdown("""
        **Common issues and solutions:**
        
        1. **File not found**: Make sure `enhanced_lstm_air_quality_model.pth` is in the same folder as `app.py`
        
        2. **PyTorch version mismatch**: Try:
           ```bash
           pip install torch==2.0.0
           ```
        
        3. **Security restrictions**: The app should handle PyTorch 2.6+ security restrictions automatically
        
        4. **Model architecture mismatch**: Ensure the model class in `app.py` matches your training code
        
        **For deployment to Streamlit Cloud:**
        - Make sure all files are committed to GitHub
        - Check that `requirements.txt` includes all dependencies
        - The model file might be large (>100MB), consider compressing or using a different storage method
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Air Quality LSTM Dashboard | Powered by Streamlit"
    "</div>",
    unsafe_allow_html=True
)
