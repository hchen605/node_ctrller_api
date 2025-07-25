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

### Local test
User may use the command to deploy the model into an endpoint.
```
mlflow models serve -m "models:/MNIST_PyTorch_Model/2" --port 5001 --no-conda
```

### Docker 
User can run the command to build docker image
```
mlflow models build-docker -m "models:/MNIST_PyTorch_Model/2" -n mnist_model
```
Then deploy and run the model as an API within the container.
```
docker run -p 5002:8080 mnist_model
```
Please check `api_test.py` to see how to run the API endpoint.



