export USER="cluster"
export CONTAINER_NAME="mxnet-docker"
export CONTAINER_REGISTRY="haibinlin/bert-docker:test"
export KEY_FILE=".ssh_cluster_key.pem"
export PORT=2022
export HOSTS=hosts

launch-container:
	clush --hostfile $(HOSTS) "bash $(PWD)/start_container.sh $(CONTAINER_NAME) $(USER) $(CONTAINER_REGISTRY)"

sync-key: launch-container
	docker cp $(CONTAINER_NAME):/home/$(USER)/.ssh/ssh_host_rsa_key $(KEY_FILE)

sync-host: sync-key
	scp -P$(PORT) -i $(KEY_FILE) $(HOSTS) $(USER)@localhost:mpihosts.txt

run-nccl: sync-key sync-host
	ssh -p$(PORT) -i $(KEY_FILE) $(USER)@localhost /opt/benchmarks/run_benchmark.sh nccl
