#!/bin/bash

echo "=== 1. Dynamic Importer ==="
cat steps/dynamic_importer.py 2>/dev/null || echo "File not found"

echo -e "\n=== 2. Feature Engineering ==="
cat src/feature_engineering.py 2>/dev/null || echo "File not found"

echo -e "\n=== 3. MLflow Deployment Step ==="
cat steps/mlflow_model_deployer_step.py 2>/dev/null || echo "File not found"

echo -e "\n=== 4. Model Building Step ==="
cat steps/model_building_step.py 2>/dev/null || echo "File not found"

echo -e "\n=== 5. Register Model Step ==="
cat steps/register_model_step.py 2>/dev/null || echo "File not found"

echo -e "\n=== 6. Inference Pipeline ==="
cat pipelines/inference_pipeline.py 2>/dev/null || echo "File not found"

echo -e "\n=== 7. MLflow Service Log (last 50 lines) ==="
tail -50 /home/obed/.config/zenml/local_stores/aa2ad575-e448-41cf-bfd4-420b8886ed2c/53dcc896-af42-42ed-aa14-e6ceb6c8bb79/service.log 2>/dev/null || echo "Log file not found"

echo -e "\n=== 8. MLflow Model Info ==="
mlflow models list 2>/dev/null || echo "MLflow command not available"
