import click
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri  # type: ignore

from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Whether to run the deployment pipeline, the inference pipeline, or both.",
)
def main(config: str):
    deploy = config in (DEPLOY, DEPLOY_AND_PREDICT)
    predict = config in (PREDICT, DEPLOY_AND_PREDICT)

    if deploy:
        continuous_deployment_pipeline()

    if predict:
        inference_pipeline()

    print(
        "\nRun:\n"
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "to inspect experiment runs, or check the deployed service status via "
        "the ZenML CLI (`zenml model-deployer models list`)."
    )


if __name__ == "__main__":
    main()