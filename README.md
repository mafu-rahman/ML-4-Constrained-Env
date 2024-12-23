# ML-4-Constrained-Env
This is an experimental setup for running ML models in constrained environments, as discussed in the paper here:

This repository sets up and monitors two similar experiments. 

Experiment 1 (Similar instructions for Experiment 2. Note: Due to port conflicts, please delete the conatiners and images when running the other experiment):

Three MobileNet models (original, PCA500, and PCA200) in resource-constrained environments. Each model is limited to 1 CPU and 512MB RAM. The stack includes cAdvisor for container metrics, Prometheus for metrics storage, and Grafana for visualization.

## How to Run
Clone the repository

Start the services:  docker compose up --build

Please use the model_performance.ipynb Notebook where everyhting is already implemented. Run all the cells to test all these models.
While the notebook cells are running, access Grafana for visualization of the metrics.

### Access the services:

MobileNet Original: http://localhost:8000

MobileNet PCA500: http://localhost:8001

MobileNet PCA200: http://localhost:8002

cAdvisor: http://localhost:8080

Prometheus: http://localhost:9090

Grafana: http://localhost:3000

### Importing Grafana Dashboards
Log in to Grafana at http://localhost:3000 (default: admin/admin).
Go to Dashboards > Import and upload JSON files from the grafana_dashboards folder.
Link the dashboard to the Prometheus data source (http://prometheus:9090).
