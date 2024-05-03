# app init

```
conda env export > conda.yml && pip list --format=freeze > requirements.txt
```

docker run -p 5000:5000 --name mlflow-server -v /Users/minwook/code/personal/llama3-langchain-mlflow/mlflow:/mlflow ghcr.io/mlflow/mlflow mlflow server --host 0.0.0.0 --dev


docker run -p 5000:5000 --name mlflow-deployment mlflow deployments start-server --config-path ./mlflow/config.yaml --host 0.0.0.0 --port 5000 --workers 1