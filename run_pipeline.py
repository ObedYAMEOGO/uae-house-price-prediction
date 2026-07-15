import click
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri  # type: ignore

from pipelines.training_pipeline import ml_pipeline


@click.command()
def main():
    """
    Runs the UAE house rent prediction ML pipeline and prints the command
    to launch the MLflow UI for inspecting the experiment run afterward.
    """
    run = ml_pipeline()

    print(
        "Pipeline run completed.\n\n"
        "Now run:\n"
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n\n"
        "to inspect your experiment runs (model params, metrics, artifacts) "
        "in the MLflow UI."
    )


if __name__ == "__main__":
    main()