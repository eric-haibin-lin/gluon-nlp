export USER="cluster"
export CONTAINER_NAME="mxnet-docker"
#export CONTAINER_REGISTRY="haibinlin/bert-docker:test"
#export CONTAINER_REGISTRY="haibinlin/bert-docker:2d74f7b"
#export CONTAINER_REGISTRY="haibinlin/bert-docker:3ccd7ad"
#export CONTAINER_REGISTRY="haibinlin/bert-docker:9b058b59"
#export CONTAINER_REGISTRY="haibinlin/bert-docker:b21c8e12"
#export CONTAINER_REGISTRY="haibinlin/bert-docker:1ea690c6"
#export CONTAINER_REGISTRY="haibinlin/bert-docker:ecda0001" # without atomic add
#export CONTAINER_REGISTRY="haibinlin/bert-docker:745c25cc" # without atomic add, bias_add
#export CONTAINER_REGISTRY="haibinlin/bert-docker:20190504" # 20190504
#export CONTAINER_REGISTRY="haibinlin/bert-docker:67def0e5"
#export CONTAINER_REGISTRY="haibinlin/bert-docker:016e6d56" # new lamb only withotu atomic add
export CONTAINER_REGISTRY="haibinlin/bert-docker:eb4930ce" # new deterministic lamb
export NLP_PATH="/home/ec2-user/efs/haibin/container/"
export KEY_FILE="$(NLP_PATH)/../.ssh_cluster_key.pem"
export DATA_PATH="/fsx"
export PORT=2022
export BERT_DIR="/opt/gluon-nlp/scripts/bert"
export HOSTS ?= "localhost"

launch-container:
	-$(shell sudo pkill python3)
	clush --hostfile $(HOSTS) "bash $(PWD)/start_container.sh $(CONTAINER_NAME) $(USER) $(CONTAINER_REGISTRY) $(NLP_PATH) $(DATA_PATH)"

sync-key: launch-container
	docker cp $(CONTAINER_NAME):/home/$(USER)/.ssh/ssh_host_rsa_key $(KEY_FILE)

sync-host: sync-key
	scp -o 'StrictHostKeyChecking=no' -P$(PORT) -i $(KEY_FILE) $(HOSTS) $(USER)@localhost:mpihosts.txt

run-nccl: sync-key sync-host
	ssh -o 'StrictHostKeyChecking=no' -p$(PORT) -i $(KEY_FILE) $(USER)@localhost /opt/benchmarks/run_benchmark.sh nccl

prepare-nlp: sync-key sync-host
	clush --hostfile $(HOSTS) "ssh -p$(PORT) -o 'StrictHostKeyChecking=no' -i $(KEY_FILE) $(USER)@localhost 'cd /opt/gluon-nlp && pip3 install . --user'"

run-bert-%:: prepare-nlp
	ssh -o 'StrictHostKeyChecking=no' -p$(PORT) -i $(KEY_FILE) $(USER)@localhost 'cd $(BERT_DIR) && CONFIG=configurations/$*.yml bash start_pretrain.sh $(CONTAINER_REGISTRY)'
