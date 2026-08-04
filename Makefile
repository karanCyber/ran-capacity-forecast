# ran-capacity-forecast
#
# `make help` lists everything. The demo path is:
#   make install && make train && make report && make api
# The Kubernetes path is:
#   make docker-build && make kind-up && make k8s-deploy && make k8s-status

IMAGE       ?= ran-capacity-forecast
TAG         ?= 0.1.0
CLUSTER     ?= ran
NAMESPACE   ?= ran-forecast
PYTHON      ?= python3
export PYTHONPATH := src

.DEFAULT_GOAL := help
.PHONY: help install data train train-fast report test lint api clean \
        docker-build kind-up kind-load k8s-deploy k8s-seed k8s-status \
        k8s-logs k8s-port-forward k8s-trigger-retrain k8s-clean kind-down verify

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- local ----
install: ## Install runtime + dev dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt

data: ## Generate synthetic KPIs and resample to hourly
	$(PYTHON) -m ran_forecast.data.generate
	$(PYTHON) -m ran_forecast.data.ingest

train: ## Full pipeline: backtest, train, forecast, write artifacts
	$(PYTHON) scripts/train.py

train-fast: ## Same, but refit at every backtest origin (slower, more rigorous)
	$(PYTHON) scripts/train.py --refit-every-days 1

report: ## Regenerate docs/RESULTS.md and the plots from artifacts
	$(PYTHON) scripts/report.py

test: ## Run the test suite
	$(PYTHON) -m pytest tests/ -q

api: ## Serve the API locally on :8000
	uvicorn ran_forecast.api.main:app --reload --host 0.0.0.0 --port 8000

verify: ## End-to-end check: tests, pipeline, API smoke test
	$(PYTHON) scripts/verify.py

clean: ## Remove generated artifacts
	rm -rf artifacts/*.parquet artifacts/*.json artifacts/*.csv artifacts/*.txt
	rm -rf .pytest_cache **/__pycache__

# --------------------------------------------------------------- docker ----
docker-build: ## Build the multi-stage image
	docker build -t $(IMAGE):$(TAG) .
	@docker images $(IMAGE):$(TAG) --format 'built {{.Repository}}:{{.Tag}} ({{.Size}})'

docker-run: ## Run the image locally against ./artifacts
	docker run --rm -p 8000:8000 \
		-v $$(pwd)/artifacts:/data/artifacts:ro \
		-e ARTIFACT_DIR=/data/artifacts \
		$(IMAGE):$(TAG)

# ------------------------------------------------------------ kubernetes ----
kind-up: ## Create the kind cluster
	kind create cluster --name $(CLUSTER)
	kubectl cluster-info --context kind-$(CLUSTER)

kind-load: docker-build ## Load the local image into kind (no registry needed)
	kind load docker-image $(IMAGE):$(TAG) --name $(CLUSTER)

k8s-deploy: kind-load ## Apply all manifests
	kubectl apply -k k8s/
	@echo "Waiting for the seed job to populate the volume (first run ~2 min)..."
	kubectl -n $(NAMESPACE) wait --for=condition=complete job/ran-forecast-seed --timeout=600s
	kubectl -n $(NAMESPACE) rollout status deployment/ran-forecast-api --timeout=300s

k8s-seed: ## Re-run the seeding job from scratch
	-kubectl -n $(NAMESPACE) delete job ran-forecast-seed --ignore-not-found
	kubectl apply -f k8s/05-seed-job.yaml
	kubectl -n $(NAMESPACE) wait --for=condition=complete job/ran-forecast-seed --timeout=600s

k8s-status: ## The screenshot shot: pods, cronjob, jobs, service
	@echo "=== pods ===";     kubectl -n $(NAMESPACE) get pods -o wide
	@echo "\n=== deployment ==="; kubectl -n $(NAMESPACE) get deployment
	@echo "\n=== service ===";    kubectl -n $(NAMESPACE) get svc
	@echo "\n=== cronjob ===";    kubectl -n $(NAMESPACE) get cronjob
	@echo "\n=== jobs ===";       kubectl -n $(NAMESPACE) get jobs
	@echo "\n=== pvc ===";        kubectl -n $(NAMESPACE) get pvc

k8s-logs: ## Tail the API logs
	kubectl -n $(NAMESPACE) logs -l app.kubernetes.io/name=ran-forecast-api --tail=100 -f

k8s-port-forward: ## Expose the API on localhost:8000
	@echo "API on http://localhost:8000/docs"
	kubectl -n $(NAMESPACE) port-forward svc/ran-forecast-api 8000:80

k8s-trigger-retrain: ## Fire the nightly CronJob immediately (proves it works)
	kubectl -n $(NAMESPACE) create job --from=cronjob/ran-forecast-retrain \
		manual-retrain-$$(date +%s)
	kubectl -n $(NAMESPACE) get jobs

k8s-clean: ## Delete all resources
	kubectl delete -k k8s/ --ignore-not-found

kind-down: ## Destroy the kind cluster
	kind delete cluster --name $(CLUSTER)
