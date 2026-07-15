# steps/prediction_service_loader_step.py
import logging
from zenml import step
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str, step_name: str
) -> MLFlowDeploymentService:
    """Retrieves the running MLflow prediction service deployed by the continuous deployment pipeline."""
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # Get all services (including stopped ones)
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        running=None,  # Get all services, not just running ones
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service found for pipeline '{pipeline_name}' "
            f"step '{step_name}'."
        )

    # Get the most recent service
    service = existing_services[0]
    
    logging.info(f"Found service: {service.uuid}")
    logging.info(f"Service status: running={service.is_running}")
    logging.info(f"Service prediction URL: {service.prediction_url}")
    
    # If service is not running, try to start it
    if not service.is_running:
        logging.info("Service is not running. Attempting to start...")
        try:
            service.start(timeout=60)
            logging.info("Service started successfully!")
        except Exception as e:
            logging.error(f"Failed to start service: {e}")
            raise RuntimeError(f"Service {service.uuid} failed to start. Check logs.")
    
    return service