# Heart Disease MLOps Project

## Project Overview

This project builds an end-to-end machine learning pipeline for predicting heart disease risk using the Heart Disease UCI dataset. The project includes data acquisition, exploratory data analysis, model development, experiment tracking, model packaging, API deployment, CI/CD, Docker, Kubernetes, and monitoring.

## Project Structure

```text
heart-disease-mlops/
├── data/
├── notebooks/
├── src/
├── model/
├── reports/
├── tests/
├── api/
├── deployment/
├── requirements.txt
├── environment.yml
└── README.md
```
## Setup Instructions
###  Create and activate a Python 3.11 virtual environment:
For Windows PowerShell:

```powershell
py -3.11 -m venv venv
venv\Scripts\activate
```

For Windows Command Prompt:

```cmd
py -3.11 -m venv venv
venv\Scripts\activate
```

For macOS/Linux:

```bash
python3.11 -m venv venv
source venv/bin/activate
```
### Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
### Download and prepare the dataset:
```bash
python src/download_data.py
```
### Train the model:
```bash
python src/train.py
```
### Experiment Tracking with MLflow

MLflow was used to track model training experiments for the Heart Disease classification project.

The following items were logged for each model run:

- Model name
- Hyperparameters
- Cross-validation ROC-AUC
- Test accuracy
- Test precision
- Test recall
- Test F1-score
- Test ROC-AUC
- Confusion matrix plot
- ROC curve plot
- Trained sklearn model pipeline

To run model training with MLflow tracking:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

The MLflow experiment is named:

```text
heart_disease_classification
```

In the MLflow UI, review:

- Logistic Regression run
- Random Forest run
- Accuracy, precision, recall, F1-score, and ROC-AUC metrics
- Confusion matrix artifacts
- ROC curve artifacts
- Saved model artifacts
- Final selected model run

Screenshots from MLflow are included in the report as experiment tracking evidence.

### Run prediction using the saved model:
```bash
python src/predict.py --input model/sample_input.json
```

### Model Packaging and Reproducibility

The final model is saved as a complete sklearn pipeline:

```text
model/heart_disease_pipeline.joblib
```
### Model Package Contents
```text
model/
├── heart_disease_pipeline.joblib
├── feature_metadata.json
└── sample_input.json
```
### Reproducibility Notes

The model training workflow uses a fixed random seed for train-test splitting, cross-validation, Logistic Regression, and Random Forest training. The final model is selected based on test ROC-AUC and saved as a reusable pipeline.This pipeline includes preprocessing and the final trained classifier, which ensures the same transformation logic is used during both training and inference.The project dependencies are captured in requirements.txt, and the recommended Python version is documented in environment.yml.

### CI/CD Pipeline and Automated Testing

This project uses Pytest for automated unit testing and Flake8 for code linting.

The CI/CD workflow is implemented using GitHub Actions. The workflow is triggered on push, pull request, or manual execution.

The pipeline performs the following steps:

1. Checks out the repository
2. Sets up Python 3.11
3. Installs dependencies from `requirements.txt`
4. Runs Flake8 linting
5. Downloads and prepares the dataset
6. Trains the machine learning model
7. Validates the packaged prediction pipeline
8. Runs unit tests using Pytest
9. Uploads trained model artifacts and reports

The pipeline is designed to fail if linting or unit tests fail. This supports code quality, reproducibility, and reliable model development.

To run tests locally:

```bash
python -m pytest tests -v
```
To run linting locally:
```bash
python -m flake8 src tests
```
### Run the FastAPI Application Locally

Start the API locally:

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

Open the Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Health check endpoint:

```text
http://127.0.0.1:8000/health
```

Prediction endpoint:

```text
http://127.0.0.1:8000/predict
```

Metrics endpoint:

```text
http://127.0.0.1:8000/metrics
```

Example prediction request body:

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```
### Docker Containerization

The model-serving API is containerized using Docker. The Docker image includes the FastAPI application, trained model pipeline, source code, and all required dependencies.

```text
A separate `requirements-docker.txt` file is used for the Docker image to avoid installing Windows-specific packages such as `pywin32` inside the Linux-based container. The Docker requirements file includes only the dependencies required for model serving.
```

#### 1. Build Docker Image

```bash
docker build -t heart-disease-api:latest .
```
#### 2. Run Docker Container
```bash
docker run -d -p 8000:8000 --name heart-disease-api-container heart-disease-api:latest
```
#### 3. Check that the container is running:

```bash
docker ps
```

#### 4. Open API Documentation
```text
http://127.0.0.1:8000/docs
```
#### 5. Health Check
```text
http://127.0.0.1:8000/health
```
#### 6. Sample Prediction Request
```powershell
$body = @{
    age = 63
    sex = 1
    cp = 3
    trestbps = 145
    chol = 233
    fbs = 1
    restecg = 0
    thalach = 150
    exang = 0
    oldpeak = 2.3
    slope = 0
    ca = 0
    thal = 1
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/predict" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"
```
#### 7. Stop Container
```bash
docker stop heart-disease-api-container
docker rm heart-disease-api-container
```
### Kubernetes Deployment

The Dockerized FastAPI model-serving API was deployed to local Kubernetes using Docker Desktop Kubernetes.

The deployment files are stored in the `deployment/` folder:

```text
deployment/
├── namespace.yaml
├── deployment.yaml
└── service.yaml
```
#### 1. Enable Kubernetes
Enable Kubernetes in Docker Desktop:
```text
Docker Desktop > Settings > Kubernetes > Enable Kubernetes
```
#### 2. Build Docker Image
```bash
docker build -t heart-disease-api:latest .
```
#### 3. Deploy to Kubernetes
```bash
kubectl apply -f .\deployment\namespace.yaml
kubectl apply -f .\deployment\deployment.yaml
kubectl apply -f .\deployment\service.yaml
```
#### 4. Verify Deployment
```bash
kubectl get all -n heart-disease
kubectl get pods -n heart-disease
kubectl get svc -n heart-disease
```
Wait for the deployment to become available:

```bash
kubectl wait --for=condition=available deployment/heart-disease-api -n heart-disease --timeout=120s
```
#### 5. Access API
If the LoadBalancer is available locally, open:
```text
http://127.0.0.1:8000/docs
```
If needed, use port forwarding:
```bash
kubectl port-forward -n heart-disease service/heart-disease-api-service 8000:8000
```
Then open:
```text
http://127.0.0.1:8000/docs
```
#### 6. Test Health Endpoint:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```
#### 7. Test Prediction Endpoint
```powershell
$body = @{
    age = 63
    sex = 1
    cp = 3
    trestbps = 145
    chol = 233
    fbs = 1
    restecg = 0
    thalach = 150
    exang = 0
    oldpeak = 2.3
    slope = 0
    ca = 0
    thal = 1
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/predict" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"
```
#### 8. View Logs
```bash
kubectl logs -n heart-disease deployment/heart-disease-api --tail=50
```
#### 9. Delete Deployment
```bash
kubectl delete -f deployment/
```
### Monitoring and Logging

The FastAPI model-serving API includes request logging and Prometheus metrics.

The API logs request method, endpoint path, status code, request duration, prediction output, and heart disease probability. These logs can be viewed using Docker logs or Kubernetes logs.

A `/metrics` endpoint is exposed using `prometheus-fastapi-instrumentator`. Prometheus is configured to scrape this endpoint every 5 seconds.

#### Run Monitoring Stack

```bash
docker compose -f docker-compose.monitoring.yml up --build
```
#### API Documentation
```text
http://127.0.0.1:8000/docs
```
#### Metrics Endpoint
```text
http://127.0.0.1:8000/metrics
```
#### Prometheus
```text
http://127.0.0.1:9090
```
#### Prometheus Targets
```text
http://127.0.0.1:9090/targets
```
#### Grafana
```text
http://127.0.0.1:3000
```
Default Grafana login:
```text
Username: admin
Password: admin
```
#### View API Logs
```bash
docker logs heart-disease-api-container
```
#### Stop Monitoring Stack
```bash
docker compose -f docker-compose.monitoring.yml down
```