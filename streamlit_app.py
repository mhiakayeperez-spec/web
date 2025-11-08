import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config("Cacao PK-PD Dashboard", layout="wide")
st.title(" Cacao PK–PD Study Dashboard")
st.markdown("Upload your CSV and explore visualizations, summaries, and a small predictive module.")

# Sidebar controls
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Sample template generation
def generate_template():
    df = pd.DataFrame({
        "Compound": ["Theobromine", "Catechin", "Epicatechin"],
        "Trial": [1, 1, 1],
        "AUC_N": [29.81, 45.23, 40.12],
        "AUC_C": [40.79, 60.12, 55.01],
        "Composite": [1.37, 1.80, 1.65]
    })
    return df

if st.sidebar.button("Download sample CSV template"):
    template = generate_template()
    csv = template.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download template",
        data=csv,
        file_name='cacao_template.csv',
        mime='text/csv'
    )

# Load data
@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)


if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Normalize column names AFTER df is loaded
    df.columns = [c.strip() for c in df.columns]
    expected = {"Compound", "Trial", "AUC_N", "AUC_C", "Composite"}

    if not expected.issubset(set(df.columns)):
        st.warning(f"CSV is missing expected columns. Expected at least: {sorted(list(expected))}")

# --- Main Layout ---
if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    expected = {"Compound", "Trial", "AUC_N", "AUC_C", "Composite"}
    if not expected.issubset(set(df.columns)):
        st.warning(f"CSV is missing expected columns. Expected at least: {sorted(list(expected))}")

    # --- Data Preview ---
    st.subheader("Data preview")
    st.dataframe(df)

    # --- Summary Statistics ---
    st.subheader("Summary statistics")
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        stats = numeric.describe().T
        stats['median'] = numeric.median()
        st.table(stats)
    else:
        st.warning("No numeric columns found for summary statistics.")

else:
    st.info("Please upload a CSV file to display the data and summary statistics.")

# --- Compound Ranking ---
if uploaded_file is not None and 'df' in locals():
    st.subheader("Compound ranking by Composite effectiveness")

    if 'Composite' in df.columns and 'Compound' in df.columns:
        try:
            ranking = (
                df.groupby('Compound', as_index=False)
                .agg(Mean_Composite=('Composite', 'mean'),
                     Std_Composite=('Composite', 'std'),
                     Count=('Composite', 'count'))
                .sort_values('Mean_Composite', ascending=False)
            )
            st.table(ranking)
        except Exception as e:
            st.error(f"Error computing ranking: {e}")
    else:
        st.warning("Required columns ('Compound', 'Composite') are missing from the CSV.")
else:
    st.info("Upload a CSV file to view compound rankings.")

# --- Visualizations ---
if uploaded_file is not None and 'df' in locals():
    st.subheader("Visualizations")

    if not df.empty:
        col1, col2 = st.columns(2)

        # --- Scatter Plot: AUC_N vs AUC_C ---
        with col1:
            if all(col in df.columns for col in ['AUC_C', 'AUC_N', 'Compound', 'Trial', 'Composite']):
                st.markdown("**AUC_N vs AUC_C (Scatter)**")
                fig1 = px.scatter(
                    df,
                    x='AUC_C',
                    y='AUC_N',
                    color='Compound',
                    symbol='Trial',
                    size='Composite',
                    hover_data=df.columns
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.warning("Missing columns required for scatter plot (AUC_C, AUC_N, Compound, Trial, Composite).")

        # --- Bar Chart: Mean Composite by Compound ---
        with col2:
            if 'ranking' in locals() and not ranking.empty:
                st.markdown("**Mean Composite by Compound (Bar)**")
                fig2 = px.bar(
                    ranking,
                    x='Compound',
                    y='Mean_Composite',
                    error_y='Std_Composite',
                    hover_data=['Count']
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("Ranking data not available for bar chart.")

        # --- Box Plot: Composite by Trial ---
        st.markdown("**Trial Variability (Boxplot of Composite by Trial)**")
        if all(col in df.columns for col in ['Trial', 'Composite']):
            fig3 = px.box(df, x='Trial', y='Composite', points='all')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Missing columns required for boxplot (Trial, Composite).")

    else:
        st.warning("Uploaded CSV is empty. Please check your file content.")
else:
    st.info("Upload a CSV file to view data visualizations.")

# --- Correlations ---
if uploaded_file is not None and 'df' in locals():
    st.subheader("Correlations & Simple Regression Fit")

    # Check for numeric columns
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        st.warning("No numeric columns found to compute correlations.")
    else:
        try:
            corr = numeric.corr()
            st.markdown("**Correlation Matrix**")
            st.dataframe(corr.style.background_gradient(cmap='Blues'))

            # Optional: Add regression plot if columns exist
            if all(col in df.columns for col in ['AUC_C', 'Composite']):
                import plotly.express as px
                fig_reg = px.scatter(
                    df, x='AUC_C', y='Composite',
                    trendline='ols', color='Compound',
                    title="Simple Regression: Composite vs AUC_C"
                )
                st.plotly_chart(fig_reg, use_container_width=True)
            else:
                st.info("Upload a file with 'AUC_C' and 'Composite' columns to see regression fit.")

        except Exception as e:
            st.error(f"Error computing correlations: {e}")
else:
    st.info("Upload a CSV file to analyze correlations.")

# --- Simple model training ---
st.subheader("Train a predictive model for Composite")

# Feature selection UI
features = st.multiselect(
    "Select features to use as predictors",
    options=['AUC_N', 'AUC_C'],
    default=['AUC_N', 'AUC_C']
)
model_choice = st.selectbox("Model", options=['Linear Regression', 'Random Forest'])
test_size = st.slider("Test set proportion", 0.1, 0.5, 0.25)

# Safety checks
if 'df' not in locals():
    st.warning("No data loaded. Upload a CSV to train a model.")
elif len(features) == 0:
    st.info("Choose at least one numeric feature to train a model.")
elif 'Composite' not in df.columns:
    st.warning("Column 'Composite' not found in the uploaded CSV.")
else:
    # Ensure selected features exist and are numeric
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        st.warning(f"Selected features missing from data: {missing_feats}")
    else:
        numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_features) != len(features):
            st.warning("One or more selected features are not numeric. Please choose numeric features.")
        else:
            # Enough rows?
            n_rows = len(df)
            if n_rows < 2:
                st.warning("Not enough rows to train a model. Need at least 2 rows.")
            else:
                # Ensure test set yields at least one sample
                test_count = max(1, int(np.floor(test_size * n_rows)))
                train_count = n_rows - test_count
                if train_count < 1:
                    st.warning("Test size is too large for the dataset. Reduce test size.")
                else:
                    # Prepare data
                    X = df[features].values
                    y = df['Composite'].values

                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )

                        if model_choice == 'Linear Regression':
                            model = LinearRegression()
                        else:
                            model = RandomForestRegressor(n_estimators=200, random_state=42)

                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)

                        r2 = r2_score(y_test, preds)
                        rmse = mean_squared_error(y_test, preds, squared=False)

                        st.write(f"Model: **{model_choice}** — R² = **{r2:.3f}**, RMSE = **{rmse:.3f}**")

                        # Show coefficients / importances
                        if model_choice == 'Linear Regression':
                            coef_df = pd.DataFrame({'feature': features, 'coef': model.coef_})
                            st.table(coef_df)
                        else:
                            imp_df = pd.DataFrame({
                                'feature': features,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False)
                            st.table(imp_df)

                    except Exception as e:
                        st.error(f"Model training failed: {e}")

# --- Show feature importances or coefficients ---
if 'model' in locals() and 'features' in locals():
    try:
        if model_choice == 'Linear Regression':
            if hasattr(model, 'coef_'):
                coefs = model.coef_
                coef_df = pd.DataFrame({
                    'Feature': features,
                    'Coefficient': coefs
                })
                st.markdown("**Model Coefficients**")
                st.table(coef_df)
            else:
                st.warning("Coefficients not available for this model.")
        else:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                imp_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                st.markdown("**Feature Importances**")
                st.table(imp_df)
            else:
                st.warning("Feature importances not available for this model.")
    except Exception as e:
        st.error(f"Error displaying model results: {e}")
else:
    st.info("Train a model first to view coefficients or importances.")

# --- Allow user to predict ---
st.markdown("### Predict Composite for New Input")

# Check if model and features are available
if 'model' in locals() and 'features' in locals() and len(features) > 0:
    try:
        with st.form(key='predict_form'):
            input_vals = {}
            for f in features:
                default_val = float(df[f].median()) if f in df.columns else 0.0
                input_vals[f] = st.number_input(f"Enter value for {f}", value=default_val)
            submit = st.form_submit_button('Predict')

        if submit:
            # Prepare input for prediction
            input_array = np.array([input_vals[f] for f in features]).reshape(1, -1)
            pred_val = model.predict(input_array)[0]
            st.success(f"Predicted Composite = **{pred_val:.4f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Train a model first before making predictions.")

# --- Export processed dataframe ---
if uploaded_file is not None and 'df' in locals():
    st.subheader("Export Processed Data")

    try:
        to_download = df.copy()
        csv = to_download.to_csv(index=False).encode('utf-8')

        st.download_button(
            "📥 Download Processed CSV",
            data=csv,
            file_name='cacao_processed.csv',
            mime='text/csv'
        )

        st.markdown("---")
        st.markdown(
            "**Notes:** Ensure your CSV column names match: "
            "`Compound, Trial, AUC_N, AUC_C, Composite`.\n\n"
            "If you'd like the app to compute **Composite** from a formula "
            "(instead of reading it from CSV), tell me the formula and I will add a computed column."
        )
    except Exception as e:
        st.error(f"Error exporting CSV: {e}")

else:
    st.info("Upload a CSV to get started. Use the sidebar to download a sample template.")
    st.markdown("### Expected Columns")
    st.markdown(
        "`Compound, Trial, AUC_N, AUC_C, Composite`\n\n"
        "If your column names differ, rename them or upload a modified CSV."
    )

# --- Footer ---
st.markdown("---")
st.caption("Built for: **Computational PK–PD Cacao Study** — Streamlit App Template")


