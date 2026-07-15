from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from pipelines.training_pipeline import ml_pipeline
from steps.deployment_trigger_step import deployment_trigger_step
from steps.dynamic_data_importer_step import dynamic_importer
from steps.prediction_service_loader_step import prediction_service_loader
from steps.predictor_step import predictor


@pipeline
def continuous_deployment_pipeline():
    """
    Trains the model, evaluates it, and deploys it to MLflow ONLY if it
    clears the quality bar set by deployment_trigger_step, rather than
    deploying unconditionally on every run.
    """
    trained_model, evaluation_metrics = ml_pipeline()
    should_deploy = deployment_trigger_step(evaluation_metrics=evaluation_metrics)
    deployed_model = mlflow_model_deployer_step(
        model=trained_model,
        deploy_decision=should_deploy,
        workers=1,
        timeout=60,
    )
    return deployed_model


@pipeline(enable_cache=False)
def inference_pipeline():
    """Runs batch inference against the currently deployed model service."""
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
    )

    predictor(service=model_deployment_service, input_data=batch_data)