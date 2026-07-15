from zenml import Model, pipeline


from steps.clean_data_step import clean_data_step
from steps.data_ingestion_step import data_ingestion_step
from steps.model_evaluation_step import model_evaluation_step
from steps.register_model_step import register_model_step
from steps.save_cleaned_data_step import save_cleaned_data_step
from steps.data_splitting_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.feature_target_split_step import feature_target_split_step
from steps.model_building_step import model_building_step


@pipeline(model=Model(name="uae_house_price_predictor"))
def ml_pipeline():
    """
    End-to-End ML Pipeline for UAE Property Rent Prediction.

    Steps:
    1. Data Ingestion: Extract and load the raw dataset from the ZIP archive.
    2. Data Cleaning: Drop leakage/constant/irrelevant columns, remove
       invalid rows (Rent <= 0), remove outliers (99.5th percentile).
    3. Save Cleaned Data: Persist the cleaned dataset to data/processed/.
    4. Data Splitting: Stratified train/test split by City.
    5. Feature Engineering: Bucket rare locations, log-transform Rent and
       Area_in_sqft, one-hot encode categoricals — fit on train, applied
       to both splits.
    6. Feature/Target Split: Separate X (features) from y (log-transformed
       Rent) in both train and test sets.
    7. Model Training: Train a regression model (Random Forest by default,
       per configs/model_config.yaml).
    8. Model Evaluation: Assess performance in both log-space and real
       AED-space.

    Returns
    -------
    The trained model.
    """

    # Step 1: Extract and load the raw dataset
    raw_dataset = data_ingestion_step(
        file_path="data/raw/archive.zip",
        extract_to="data/raw",
    )

    # Step 2: Clean the data (drop leakage/constant columns, invalid rows, outliers)
    cleaned_data = clean_data_step(raw_dataset)

    # Step 3: Persist the cleaned dataset for inspection/debugging
    save_cleaned_data_step(cleaned_data)

    # Step 4: Stratified train/test split (by City, to preserve city proportions)
    train_df, test_df = data_splitter_step(cleaned_data, stratify_column="City")

    # Step 5: Feature engineering — fit on train, apply to both
    train_transformed, test_transformed = feature_engineering_step(train_df, test_df)

    # Step 6: Separate features from target (Rent, already log-transformed)
    X_train, X_test, y_train, y_test = feature_target_split_step(
        train_transformed, test_transformed, target_column="Rent"
    )

    # Step 7: Train the model
    model = model_building_step(X_train=X_train, y_train=y_train)

    # Step 8: Evaluate the model
    evaluation_metrics, mae_aed = model_evaluation_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    register_model_step(trained_model=model)

    return model, evaluation_metrics

if __name__ == "__main__":
    run = ml_pipeline()