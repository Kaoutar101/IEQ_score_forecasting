elif page == "Data Analysis":
    st.markdown('<h2 class="sub-header">Data Analysis</h2>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            analysis_data = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(analysis_data)} samples for analysis")
            
            # Basic statistics
            st.markdown("#### Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", len(analysis_data))
                st.metric("Start Date", analysis_data.index[0] if 'timestamp' in analysis_data.columns else "N/A")
            with col2:
                st.metric("Columns", len(analysis_data.columns))
                st.metric("End Date", analysis_data.index[-1] if 'timestamp' in analysis_data.columns else "N/A")
            with col3:
                missing_percent = (analysis_data.isnull().sum().sum() / (len(analysis_data) * len(analysis_data.columns))) * 100
                st.metric("Missing Values", f"{missing_percent:.2f}%")
            with col4:
                duplicate_count = analysis_data.duplicated().sum()
                st.metric("Duplicates", duplicate_count)
            
            # Create tabs for different analysis views
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "📊 Distributions", "🔗 Correlations", "📋 Summary"])
            
            with tab1:
                st.markdown("#### Time Series Analysis")
                
                # Time selection
                col1, col2 = st.columns(2)
                with col1:
                    time_column = st.selectbox(
                        "Select time column",
                        options=[col for col in analysis_data.columns if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp'])] + ['Index'],
                        index=0
                    )
                
                with col2:
                    if time_column == 'Index':
                        x_data = analysis_data.index
                    else:
                        x_data = pd.to_datetime(analysis_data[time_column], errors='coerce')
                
                # Parameter selection for time series
                numeric_columns = analysis_data.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_columns) > 0:
                    selected_params = st.multiselect(
                        "Select parameters to plot",
                        options=numeric_columns,
                        default=numeric_columns[:min(3, len(numeric_columns))]
                    )
                    
                    if selected_params:
                        fig = make_subplots(
                            rows=len(selected_params), 
                            cols=1,
                            subplot_titles=[f"{param} over time" for param in selected_params],
                            vertical_spacing=0.1
                        )
                        
                        for i, param in enumerate(selected_params, 1):
                            fig.add_trace(
                                go.Scatter(
                                    x=x_data,
                                    y=analysis_data[param],
                                    mode='lines',
                                    name=param,
                                    line=dict(width=1.5),
                                    hovertemplate=f"{param}: %{{y}}<br>Time: %{{x}}<extra></extra>"
                                ),
                                row=i, col=1
                            )
                            
                            # Add rolling average
                            if len(analysis_data[param]) > 10:
                                rolling_avg = analysis_data[param].rolling(window=min(20, len(analysis_data)//10)).mean()
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_data,
                                        y=rolling_avg,
                                        mode='lines',
                                        name=f"{param} (7-day avg)",
                                        line=dict(width=2, color='red', dash='dash'),
                                        showlegend=False
                                    ),
                                    row=i, col=1
                                )
                        
                        fig.update_layout(
                            height=300 * len(selected_params),
                            showlegend=True,
                            hovermode='x unified',
                            template='plotly_white'
                        )
                        
                        for i in range(1, len(selected_params) + 1):
                            fig.update_xaxes(title_text="Time", row=i, col=1)
                            fig.update_yaxes(title_text="Value", row=i, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics for each parameter
                        st.markdown("#### Parameter Statistics")
                        stats_df = analysis_data[selected_params].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                        stats_df.columns = ['Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']
                        st.dataframe(stats_df.style.format("{:.2f}").background_gradient(cmap='Blues', axis=0))
                
                else:
                    st.warning("No numeric columns found for time series analysis.")
            
            with tab2:
                st.markdown("#### Distribution Analysis")
                
                if len(numeric_columns) > 0:
                    selected_dist_params = st.multiselect(
                        "Select parameters for distribution analysis",
                        options=numeric_columns,
                        default=numeric_columns[:min(4, len(numeric_columns))]
                    )
                    
                    if selected_dist_params:
                        # Calculate number of rows needed (2 plots per row)
                        n_params = len(selected_dist_params)
                        n_rows = (n_params + 1) // 2
                        
                        fig = make_subplots(
                            rows=n_rows, 
                            cols=2,
                            subplot_titles=selected_dist_params,
                            vertical_spacing=0.15,
                            horizontal_spacing=0.1
                        )
                        
                        colors = px.colors.qualitative.Set2
                        
                        for i, param in enumerate(selected_dist_params):
                            row = i // 2 + 1
                            col = i % 2 + 1
                            
                            # Histogram with KDE
                            fig.add_trace(
                                go.Histogram(
                                    x=analysis_data[param].dropna(),
                                    name=param,
                                    nbinsx=50,
                                    histnorm='probability density',
                                    marker_color=colors[i % len(colors)],
                                    opacity=0.7,
                                    hovertemplate=f"Value: %{{x}}<br>Density: %{{y:.3f}}<extra></extra>"
                                ),
                                row=row, col=col
                            )
                            
                            # Add box plot in a separate subplot or as overlay
                            fig.update_layout(
                                barmode='overlay',
                                showlegend=False
                            )
                        
                        fig.update_layout(
                            height=400 * n_rows,
                            title_text="Parameter Distributions",
                            template='plotly_white'
                        )
                        
                        for i in range(1, n_rows * 2 + 1):
                            fig.update_xaxes(title_text="Value", row=((i-1)//2)+1, col=((i-1)%2)+1)
                            fig.update_yaxes(title_text="Density", row=((i-1)//2)+1, col=((i-1)%2)+1)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Distribution statistics
                        st.markdown("#### Distribution Characteristics")
                        dist_stats = []
                        for param in selected_dist_params:
                            data = analysis_data[param].dropna()
                            skewness = data.skew()
                            kurtosis = data.kurtosis()
                            dist_stats.append({
                                'Parameter': param,
                                'Skewness': f"{skewness:.3f}",
                                'Kurtosis': f"{kurtosis:.3f}",
                                'Interpretation': 'Heavy-tailed' if abs(kurtosis) > 3 else 'Normal-tailed',
                                'Shape': 'Right-skewed' if skewness > 0.5 else 'Left-skewed' if skewness < -0.5 else 'Symmetric'
                            })
                        
                        dist_df = pd.DataFrame(dist_stats)
                        st.dataframe(dist_df, use_container_width=True)
                
                else:
                    st.warning("No numeric columns found for distribution analysis.")
            
            with tab3:
                st.markdown("#### Correlation Analysis")
                
                if len(numeric_columns) >= 2:
                    # Correlation matrix
                    corr_matrix = analysis_data[numeric_columns].corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu_r',
                        zmid=0,
                        text=corr_matrix.round(2).values,
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        colorbar=dict(title="Correlation")
                    ))
                    
                    fig.update_layout(
                        title="Correlation Matrix",
                        height=600,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Top correlations
                    st.markdown("#### Top Correlations")
                    
                    # Create a flattened correlation matrix
                    corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_pairs.append({
                                'Parameter 1': corr_matrix.columns[i],
                                'Parameter 2': corr_matrix.columns[j],
                                'Correlation': corr_matrix.iloc[i, j]
                            })
                    
                    corr_df = pd.DataFrame(corr_pairs)
                    corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
                    
                    # Display top positive and negative correlations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Top Positive Correlations**")
                        top_positive = corr_df.nlargest(5, 'Correlation')
                        st.dataframe(top_positive[['Parameter 1', 'Parameter 2', 'Correlation']]
                                   .style.format({'Correlation': '{:.3f}'})
                                   .background_gradient(subset=['Correlation'], cmap='Greens'), 
                                   use_container_width=True)
                    
                    with col2:
                        st.markdown("**Top Negative Correlations**")
                        top_negative = corr_df.nsmallest(5, 'Correlation')
                        st.dataframe(top_negative[['Parameter 1', 'Parameter 2', 'Correlation']]
                                   .style.format({'Correlation': '{:.3f}'})
                                   .background_gradient(subset=['Correlation'], cmap='Reds'), 
                                   use_container_width=True)
                    
                    # Scatter plot for selected correlated pair
                    st.markdown("#### Scatter Plot of Selected Variables")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        x_var = st.selectbox("X variable", numeric_columns, index=0)
                    with col2:
                        y_var = st.selectbox("Y variable", numeric_columns, index=min(1, len(numeric_columns)-1))
                    
                    if x_var != y_var:
                        fig_scatter = px.scatter(
                            analysis_data,
                            x=x_var,
                            y=y_var,
                            trendline='ols',
                            trendline_color_override='red',
                            opacity=0.6,
                            title=f"{x_var} vs {y_var}",
                            labels={x_var: x_var, y_var: y_var}
                        )
                        
                        # Calculate correlation
                        correlation = analysis_data[[x_var, y_var]].corr().iloc[0, 1]
                        
                        fig_scatter.update_layout(
                            annotations=[
                                dict(
                                    x=0.05,
                                    y=0.95,
                                    xref="paper",
                                    yref="paper",
                                    text=f"Correlation: {correlation:.3f}",
                                    showarrow=False,
                                    font=dict(size=12, color='black'),
                                    bgcolor='white',
                                    bordercolor='black',
                                    borderwidth=1,
                                    borderpad=4
                                )
                            ],
                            height=400,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
                
                else:
                    st.warning("Need at least 2 numeric columns for correlation analysis.")
            
            with tab4:
                st.markdown("#### Data Summary")
                
                # Data preview
                st.markdown("##### Data Preview")
                st.dataframe(analysis_data.head(20), use_container_width=True)
                
                # Data types
                st.markdown("##### Data Types")
                dtype_df = pd.DataFrame({
                    'Column': analysis_data.columns,
                    'Data Type': analysis_data.dtypes.astype(str),
                    'Non-Null Count': analysis_data.notnull().sum().values,
                    'Null Count': analysis_data.isnull().sum().values,
                    'Unique Values': [analysis_data[col].nunique() for col in analysis_data.columns]
                })
                
                st.dataframe(dtype_df.style.background_gradient(subset=['Null Count'], cmap='Reds'), 
                           use_container_width=True)
                
                # Export options
                st.markdown("##### Export Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Summary statistics
                    summary_stats = analysis_data.describe().T
                    csv_summary = summary_stats.to_csv()
                    st.download_button(
                        label="Download Summary Statistics",
                        data=csv_summary,
                        file_name="data_summary_statistics.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Full data
                    csv_full = analysis_data.to_csv(index=False)
                    st.download_button(
                        label="Download Full Dataset",
                        data=csv_full,
                        file_name="full_dataset.csv",
                        mime="text/csv"
                    )
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.exception(e)
    else:
        st.info(" Upload a CSV file for comprehensive data analysis")
        st.markdown("""
        **Expected data format:**
        - Time series data with timestamp column
        - Numeric parameters for analysis
        - CSV format with headers
        
        **Analysis features available:**
        1. **Time Series Analysis**: Multi-parameter trends with rolling averages
        2. **Distribution Analysis**: Histograms and statistical properties
        3. **Correlation Analysis**: Heatmaps and scatter plots
        4. **Data Summary**: Comprehensive dataset overview
        """)
