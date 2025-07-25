# node_ctrller_api
This repo demonstrate a simple prototype of using FastAPI and MLflow to build node(worker)/controller software architecture.

## Setup
Please install the env using `manolo_arch.yml`.

## MLflow model training and registration
Please check `train.py` for the demonstration of MLflow model registration with the wrapper in training loop.
User can launch MLflow UI by the command
```
mlflow ui --backend-store-uri sqlite:///mlruns.db
```
In the UI, you will be able to check the model registration, model filtering by the log_metric in `train.py`, and the meta data of the model.
And the inference of MLflow model is demonstrated in `inference.py`. User can load the model by ID or the registered name.

## API testing
