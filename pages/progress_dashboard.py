import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from model.training_history import TrainingHistory
from datetime import datetime, timedelta

def render_dashboard():
    st.title("Model Training Progress Dashboard")

    # Initialize history tracker
    history = TrainingHistory()

    # Get progress metrics
    metrics_df = history.get_progress_metrics()
    params_df = history.get_parameter_evolution()
    best_candidates = history.get_best_candidates_ever()

    # Overall Progress
    st.header("Overall Progress")
    col1, col2, col3 = st.columns(3)

    with col1:
        if not metrics_df.empty:
            latest_x_diff = metrics_df['min_x_diff'].iloc[-1]
            best_x_diff = metrics_df['min_x_diff'].min()
            st.metric("Latest X-Diff", f"{latest_x_diff:.2e}")
            st.metric("Best X-Diff Ever", f"{best_x_diff:.2e}")

    with col2:
        if not metrics_df.empty:
            total_sessions = len(metrics_df)
            improvement = ((metrics_df['min_x_diff'].iloc[0] - metrics_df['min_x_diff'].iloc[-1]) 
                         / metrics_df['min_x_diff'].iloc[0] * 100)
            st.metric("Total Training Sessions", total_sessions)
            st.metric("Overall Improvement", f"{improvement:.2f}%")

    with col3:
        if not metrics_df.empty:
            last_improvement = datetime.fromisoformat(metrics_df['timestamp'].iloc[-1])
            time_since = datetime.now() - last_improvement
            st.metric("Time Since Last Improvement", f"{time_since.days}d {time_since.seconds//3600}h")

            # Add stored candidates metric if the column exists
            if 'candidates_count' in metrics_df.columns:
                stored_candidates = metrics_df['candidates_count'].iloc[-1]
                st.metric("Stored Best Candidates", stored_candidates)
            else:
                # If column doesn't exist, get the count of best candidates
                best_candidates_count = len(best_candidates)
                st.metric("Stored Best Candidates", best_candidates_count)

    # Progress Charts
    st.header("Training Progress")

    if not metrics_df.empty:
        # X-Diff Progress
        fig = px.line(metrics_df, x='timestamp', y='min_x_diff', 
                     title="X-Difference Progress Over Time")
        fig.update_layout(yaxis_type="log")
        st.plotly_chart(fig)

        # Candidates Growth
        if 'candidates_count' in metrics_df.columns:
            fig = px.line(metrics_df, x='timestamp', y='candidates_count',
                         title="Growth of Stored Best Candidates")
            st.plotly_chart(fig)
        else:
            st.info("Candidate count history not available.")

        # Model Performance Metrics
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics_df['timestamp'], y=metrics_df['r2'],
                               name='R² Score'))
        fig.add_trace(go.Scatter(x=metrics_df['timestamp'], y=metrics_df['mse'],
                               name='MSE', yaxis="y2"))
        fig.update_layout(
            title="Model Performance Metrics",
            yaxis=dict(title="R² Score"),
            yaxis2=dict(title="MSE", overlaying="y", side="right")
        )
        st.plotly_chart(fig)

    # Parameter Evolution
    st.header("Parameter Evolution")
    if not params_df.empty:
        # Create parameter vs performance visualization
        merged_df = pd.merge(metrics_df, params_df, on=['session_id', 'timestamp'])
        
        # Parameter performance analysis
        st.subheader("Parameter Performance Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            # Polynomial degree vs X-Diff
            fig = px.scatter(merged_df, x='polynomial_degree', y='min_x_diff',
                           color='r2', size='n_estimators',
                           title="Polynomial Degree Impact on X-Diff")
            fig.update_layout(yaxis_type="log")
            st.plotly_chart(fig)
        
        with col2:
            # Tree depth vs performance
            fig = px.scatter(merged_df, x='max_depth', y='r2',
                           color='polynomial_degree', size='n_estimators',
                           title="Tree Depth Impact on R² Score")
            st.plotly_chart(fig)
        
        # Show parameter evolution over time
        for param in ['n_estimators', 'max_depth', 'min_samples_leaf', 'polynomial_degree']:
            fig = px.line(params_df, x='timestamp', y=param,
                         title=f"{param} Evolution")
            st.plotly_chart(fig)

    # Best Candidates
    st.header("Best Candidates Ever")
    if best_candidates:
        # Convert the data to a dataframe
        df_best = pd.DataFrame(best_candidates)
        
        # Preemptively convert all columns that might contain large numbers to strings
        for col in df_best.columns:
            # Check if column exists and has values
            if col in df_best and not df_best[col].empty:
                # Convert to string regardless of original type to avoid overflow
                df_best[col] = df_best[col].astype(str)
        
        # Display the dataframe with string representations
        st.dataframe(df_best)

    # Feature Importance History
    st.header("Feature Importance Evolution")
    if not metrics_df.empty and 'feature_importance' in metrics_df.columns:
        feature_names = [
            'MSB', 'Bit_77', 'Bit_80', 'Bit_120',
            'Swapped_MSB', 'Swapped_Bit_77', 'Bit_Stability',
            'Hex_0', 'Hex_19', 'Hex_20', 'Hex_26',
            'Mod_40', 'Mod_7', 'Mod_256', 'Swapped_Mod_40', 'Swapped_Mod_7',
            'Poly_2', 'Poly_3', 'Poly_4'
        ]

        # Ensure we don't try to use more feature names than we have feature importances
        latest_importance_values = metrics_df['feature_importance'].iloc[-1]
        if len(feature_names) > len(latest_importance_values):
            feature_names = feature_names[:len(latest_importance_values)]
        
        latest_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': latest_importance_values
        }).sort_values(by='Importance', ascending=False)

        fig = px.bar(latest_importance, x='Feature', y='Importance',
                    title="Current Feature Importance",
                    color='Importance')
        st.plotly_chart(fig)
        
        # Highlight top features
        st.subheader("Top 5 Most Important Features")
        st.dataframe(latest_importance.head(5))
        
    # Overfitting Analysis
    st.header("Overfitting Analysis")
    if not metrics_df.empty and 'r2' in metrics_df.columns:
        # Create overfitting gauge
        r2_values = metrics_df['r2']
        latest_r2 = r2_values.iloc[-1]
        
        # Create gauge chart for R²
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = min(max(latest_r2, -5), 1),  # Clip to viewable range
            title = {'text': "Model R² (Higher is Better)"},
            gauge = {
                'axis': {'range': [-5, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-5, -2], 'color': "red"},
                    {'range': [-2, 0], 'color': "orange"},
                    {'range': [0, 0.5], 'color': "yellow"},
                    {'range': [0.5, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        st.plotly_chart(fig)
        
        if latest_r2 < 0:
            st.warning("Negative R² indicates severe overfitting. Try reducing model complexity.")
            st.markdown("""
            **Recommendations to reduce overfitting:**
            - Decrease polynomial degree
            - Reduce max tree depth
            - Increase min samples per leaf
            - Reset model and start with simpler parameters
            """)

    # Training Parameters
    st.header("Current Training Parameters")
    if not params_df.empty:
        latest_params = params_df.iloc[-1].drop(['session_id', 'timestamp'])
        st.json(latest_params.to_dict())

if __name__ == "__main__":
    render_dashboard()