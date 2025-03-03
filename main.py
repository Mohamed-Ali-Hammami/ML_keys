import streamlit as st
import numpy as np
import plotly.express as px
from utils.data_loader import DataLoader
from utils.logger import StreamlitLogger
from model.feature_extractor import FeatureExtractor
from model.ml_trainer import MLTrainer
from pages.progress_dashboard import render_dashboard
import os
import glob
                
# Ensure model checkpoint directory exists
os.makedirs("model_checkpoints", exist_ok=True)

# Initialize components
logger = StreamlitLogger()
data_loader = DataLoader()
feature_extractor = FeatureExtractor()
trainer = MLTrainer()

# Set page config
st.set_page_config(
    page_title="ECDSA Private Key Predictor",
    layout="wide"
)

# Initialize session state
if 'trained' not in st.session_state:
    st.session_state.trained = True  # Always true since we're loading previous state
if 'show_logs' not in st.session_state:
    st.session_state.show_logs = False
if 'auto_train' not in st.session_state:
    st.session_state.auto_train = False
if 'training_count' not in st.session_state:
    st.session_state.training_count = 0

def main():
    global trainer  # Add global declaration to use the trainer instance
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select Page",
        ["Training Interface", "Progress Dashboard"]
    )

    if page == "Progress Dashboard":
        render_dashboard()
        return

    st.title("ECDSA Private Key Predictor")
    logger.info("Application started")

    # Sidebar Configuration
    st.sidebar.header("Configuration")
    target_x = 54466516474177380511287022884940819018437102136646648862302418152034269010364
    st.sidebar.text("Target X-coordinate:")
    st.sidebar.code(f"{target_x}")

    # Training Parameters in Sidebar
    st.sidebar.header("Training Parameters")
    n_estimators = st.sidebar.slider("Number of Trees", 100, 500, 200)
    max_depth = st.sidebar.slider("Max Tree Depth", 3, 10, 4, 
                                 help="Lower values reduce overfitting")
    min_samples_leaf = st.sidebar.slider("Min Samples per Leaf", 1, 15, 9,
                                        help="Higher values help prevent overfitting")
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.05, 0.01,
                                     help="Controls how quickly model adapts to the problem")
    polynomial_degree = st.sidebar.slider("Polynomial Degree", 1, 12, 6,
                                         help="Controls the complexity of polynomial features")

    # Training iterations setting
    max_iterations = st.sidebar.number_input("Number of Training Iterations", 
                                          min_value=1, 
                                          max_value=10000, 
                                          value=10,
                                          help="Number of consecutive training cycles to run")
    
    # Auto-training toggle
    st.session_state.auto_train = st.sidebar.checkbox("Enable Auto-Training", 
                                                     help="Continuously train model with updated parameters")
    
    # Polynomial optimization section
    if st.sidebar.checkbox("Optimize Polynomial Degree", help="Test different polynomial degrees to find optimal value"):
        poly_optimization = st.sidebar.expander("Polynomial Optimization")
        with poly_optimization:
            min_degree = st.number_input("Min Degree", min_value=1, max_value=10, value=1)
            max_degree = st.number_input("Max Degree", min_value=2, max_value=20, value=10)
            step = st.number_input("Step Size", min_value=1, max_value=5, value=1)
            degrees_to_test = list(range(min_degree, max_degree + 1, step))

    # Training interval
    if st.session_state.auto_train:
        auto_train_interval = st.sidebar.slider("Auto-train interval (minutes)", 0, 60, 5)

    # Development logs toggle
    st.sidebar.header("Development")
    st.session_state.show_logs = st.sidebar.checkbox("Show Logs")

    # Data Loading
    col1, col2 = st.columns(2)

    with col1:
        st.header("Data Loading")
        data_load_state = st.text('Loading data...')

        try:
            logger.info("Loading signature data and candidates...")
            sig_data_40 = DataLoader.load_signature_data("data/rsz_commonfactor40.csv")
            sig_data_7 = DataLoader.load_signature_data("data/rsz_values_commonfact7.csv")
            candidates = DataLoader.load_candidates("data/candidates_comp2.csv")

            data_load_state.text('Data loading...done!')
            logger.info("Data loading completed successfully")

            st.subheader("Dataset Statistics")
            st.write(f"Signatures with factor 40: {len(sig_data_40)}")
            st.write(f"Signatures with factor 7: {len(sig_data_7)}")
            st.write(f"Candidate keys: {len(candidates)}")

        except Exception as e:
            error_msg = f'Error loading data: {str(e)}'
            data_load_state.error(error_msg)
            logger.error(error_msg)
            return

    with col2:
        st.header("Model Configuration")
        if st.session_state.auto_train:
            st.info("Auto-training enabled. Model parameters will be automatically tuned.")
        else:
            st.info("Manual training mode. Use the sidebar to adjust parameters.")

        # Update model parameters
        trainer.tuner.current_params.update({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'learning_rate': learning_rate,
            'polynomial_degree': polynomial_degree
        })

    # Model Training
    st.header("Model Training")
    
    # Model reset option
    if st.button("Reset Model"):
        with st.spinner("Resetting model..."):
            try:

                # Remove model checkpoint files
                for checkpoint_file in glob.glob("model_checkpoints/*"):
                    os.remove(checkpoint_file)
                
                # Reinitialize trainer
                trainer = MLTrainer()
                
                st.success("Model reset successfully! Training will start from scratch.")
                logger.info("Model has been reset to initial state")
                
            except Exception as e:
                st.error(f"Error resetting model: {str(e)}")
                logger.error(f"Failed to reset model: {str(e)}")
    
    # Add polynomial optimization section
    if 'degrees_to_test' in locals():
        if st.button("Optimize Polynomial Degree"):
            with st.spinner("Testing different polynomial degrees..."):
                try:
                    # Load data for optimization
                    X, y = data_loader.prepare_data(sig_data_40, sig_data_7, candidates)
                    
                    # Run polynomial degree tests
                    poly_results = feature_extractor.test_polynomial_degrees(X, y, degrees_to_test)
                    
                    # Display results
                    st.subheader("Polynomial Degree Optimization Results")
                    st.dataframe(poly_results)
                    
                    # Plot results
                    fig = px.line(poly_results, x='polynomial_degree', y=['mse', 'r2'], 
                                 title="Performance by Polynomial Degree")
                    st.plotly_chart(fig)
                    
                    # Recommend best degree
                    best_degree = poly_results.loc[poly_results['r2'].idxmax()]['polynomial_degree']
                    st.success(f"Recommended polynomial degree: {best_degree}")
                    
                    # Update sidebar with optimal value
                    polynomial_degree = int(best_degree)
                    trainer.tuner.current_params['polynomial_degree'] = polynomial_degree
                    
                except Exception as e:
                    st.error(f"Error during optimization: {str(e)}")

    # Display current training count
    st.write(f"Training iterations completed: {st.session_state.training_count}")

    if st.button("Train Model") or st.session_state.auto_train:
        # Get the latest metrics dictionary to track improvements
        latest_metrics = None
        
        # Progress bar for multiple iterations
        progress_bar = st.progress(0)
        
        for iteration in range(max_iterations):
            st.session_state.training_count += 1
            progress_percent = (iteration / max_iterations)
            progress_bar.progress(progress_percent)
            
            iteration_header = st.empty()
            iteration_header.subheader(f"Training Iteration {iteration+1}/{max_iterations}")
            logger.info(f"Starting training iteration {st.session_state.training_count}")

            with st.spinner(f'Training model (iteration {iteration+1}/{max_iterations})...'):
                try:
                    # Prepare data (now returns private keys as well)
                    X, y, private_keys = data_loader.prepare_data(sig_data_40, sig_data_7, candidates)
                    logger.debug(f"Prepared training data with shape: {X.shape}")
                    logger.info(f"Generated {len(private_keys)} keys for training")
                    
                    # Train model
                    metrics = trainer.train(X, y, private_keys)
                    latest_metrics = metrics
                    logger.info("Model training completed")

                    # Auto-update parameters between iterations
                    if iteration < max_iterations - 1:
                        trainer.update_parameters()
                        logger.info("Model parameters updated for next iteration")

                except Exception as e:
                    error_msg = f"Error during model training: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)
                    break
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        
        # Display final metrics after all iterations
        if latest_metrics:
            st.subheader("Training Metrics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Mean Squared Error", f"{latest_metrics['mse']:.2e}")
            with col2:
                st.metric("R² Score", f"{latest_metrics['r2']:.4f}")
            with col3:
                st.metric("Minimum X-Diff", f"{latest_metrics['min_x_diff']:.2e}")

            logger.info(f"Final training metrics - MSE: {latest_metrics['mse']:.2e}, R²: {latest_metrics['r2']:.4f}")

            # Plot feature importance
            feature_importance = trainer.get_feature_importance()
            fig = px.bar(
                x=list(feature_importance.keys()),
                y=list(feature_importance.values()),
                title="Feature Importance"
            )
            st.plotly_chart(fig)

            # Schedule next training if auto-train is enabled
            if st.session_state.auto_train:
                st.warning(f"Next training will start in {auto_train_interval} minutes")

    # Predictions
    if st.session_state.trained:
        st.header("Predictions")
        logger.info("Generating predictions")

        try:
            # Get top candidates and convert to string format
            top_predictions = candidates.nsmallest(10, 'X_Diff').copy()

            # Format X_Diff and Private_Key as strings to avoid overflow
            top_predictions['X_Diff'] = top_predictions['X_Diff'].apply(
                lambda x: f"{float(x):.2e}"
            )
            top_predictions['Private_Key'] = top_predictions['Private_Key'].apply(
                lambda x: f"0x{int(x):064x}"
            )

            st.subheader("Top 10 Candidate Private Keys")
            st.dataframe(top_predictions)
            logger.info(f"Best X_Diff found: {float(candidates['X_Diff'].min()):.2e}")

            # Plot log-scale histogram of x-coordinate differences
            log_diffs = np.log10(candidates['X_Diff'].astype(float))
            fig = px.histogram(
                log_diffs,
                title="Distribution of X-Coordinate Differences (log10 scale)",
                nbins=50
            )
            fig.update_layout(
                xaxis_title="log10(X_Diff)",
                yaxis_title="Count"
            )
            st.plotly_chart(fig)
        except Exception as e:
            error_msg = f"Error displaying predictions: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)

    # Display logs if enabled
    if st.session_state.show_logs:
        st.header("Development Logs")
        st.text_area("Application Logs", logger.get_logs(), height=300)

if __name__ == "__main__":
    main()