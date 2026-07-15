import logging
from typing import Dict

from zenml import step

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_MAE_THRESHOLD_AED = 30000.0

@step
def deployment_trigger_step(
    evaluation_metrics: Dict[str, float],
    mae_threshold_aed: float = DEFAULT_MAE_THRESHOLD_AED,
) -> bool:
    """
    Decides whether the newly trained model is good enough to deploy, based
    on its test-set MAE in real AED terms. This avoids blindly redeploying
    every training run regardless of quality a common continuous-deployment
    safeguard.

    Parameters
    ----------
    evaluation_metrics : dict
        Output of model_evaluation_step.
    mae_threshold_aed : float
        Maximum acceptable MAE (in AED) for deployment to proceed
    Returns
    -------
    bool
        True if the model should be deployed.
    """
    mae_aed = evaluation_metrics.get("MAE (AED)")

    if mae_aed is None:
        logging.warning("MAE (AED) not found in evaluation metrics — refusing to deploy.")
        return False

    should_deploy = mae_aed <= mae_threshold_aed
    logging.info(
        f"Model MAE: {mae_aed:.2f} AED | Threshold: {mae_threshold_aed:.2f} AED | "
        f"Deploy decision: {should_deploy}"
    )
    return should_deploy