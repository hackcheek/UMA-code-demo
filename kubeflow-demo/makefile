.PHONY: modeling
modeling:
	jupyter lab .

pipe:
	python -m pipes_code.src.local.main

pipe_kf: pipe_download_data pipe_kf_run_pipeline

pipe_kf_run_pipeline: pipe_pf_minio_pod
	python -m pipes_code.src.kubeflow.pipelines.retrain_pipe run

pipe_kf_save_pipeline: pipe_pf_minio_pod
	python -m pipes_code.src.kubeflow.pipelines.retrain_pipe save

pipe_download_data:
	bash pipes_code/download_new_dataset.sh

pipe_pf_minio_pod:
	lsof -i :9000 | grep kubectl | head -n 1 | awk '{print $$2}' | xargs kill
	$(eval minio_pod_name = $(shell kubectl get pod -n kubeflow | grep -i minio | awk '{print $$1}'))
	kubectl port-forward -n kubeflow $(minio_pod_name) 9000:9000 &>/dev/null &

pipe_kf_dashboard:
	bash -c "sleep 2; open http://localhost:8080" &
	kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

start_kubeflow_cluster:
	kind create cluster
	k3d cluster create mycluster

	export PIPELINE_VERSION=2.1.0
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
	kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-emissary?ref=$PIPELINE_VERSION"

restart_kubeflow: delete_experiment_containers
	kubectl rollout restart deployment/ml-pipeline-ui -n kubeflow

delete_experiment_containers:
	kubectl get pod -n kubeflow | grep pipeline | awk '{print $1}' | xargs kubectl delete pod -n kubeflow

.PHONY: api
api:
	lsof -i :9595 | grep -i python | head -n 1 | awk '{print $$2}' | xargs kill
	python -m api.src.main &
	sleep 5; pytest
	open http://localhost:9595/docs

start_api:
	python -m api.src.main

api_run_tests:
	pytest
