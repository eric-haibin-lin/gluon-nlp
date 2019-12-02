export USER="cluster"
export CONTAINER_NAME="mxnet-docker"
export CONTAINER_REGISTRY="haibinlin/bert-docker:test"
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
	scp -P$(PORT) -i $(KEY_FILE) $(HOSTS) $(USER)@localhost:mpihosts.txt

run-nccl: sync-key sync-host
	ssh -p$(PORT) -i $(KEY_FILE) $(USER)@localhost /opt/benchmarks/run_benchmark.sh nccl

prepare-nlp: sync-key sync-host
	clush --hostfile $(HOSTS) "ssh -p$(PORT) -o 'StrictHostKeyChecking=no' -i $(KEY_FILE) $(USER)@localhost 'cd /opt/gluon-nlp && pip3 install . --user'"

run-bert-test-8: prepare-nlp
	ssh -p$(PORT) -i $(KEY_FILE) $(USER)@localhost 'cd $(BERT_DIR) && CONFIG=configurations/test.yml bash start_pretrain.sh'

run-bert-test-16: prepare-nlp
	ssh -p$(PORT) -i $(KEY_FILE) $(USER)@localhost 'cd $(BERT_DIR) && CONFIG=configurations/test-2.yml bash start_pretrain.sh'

run-bert-64k-8: prepare-nlp
	ssh -p$(PORT) -i $(KEY_FILE) $(USER)@localhost 'cd $(BERT_DIR) && CONFIG=configurations/default.yml bash start_pretrain.sh'

run-bert-64k-512: prepare-nlp
	ssh -p$(PORT) -i $(KEY_FILE) $(USER)@localhost 'cd $(BERT_DIR) && CONFIG=configurations/64k-512.yml bash start_pretrain.sh'

run-bert-64k-512-multilamb: prepare-nlp
	ssh -p$(PORT) -i $(KEY_FILE) $(USER)@localhost 'cd $(BERT_DIR) && CONFIG=configurations/64k-512-multilamb.yml bash start_pretrain.sh'

run-bert-64k-512-fp32-sa-len: prepare-nlp
	ssh -p$(PORT) -i $(KEY_FILE) $(USER)@localhost 'cd $(BERT_DIR) && CONFIG=configurations/64k-512-fp32-sa-len.yml bash start_pretrain.sh'

run-bert-64k-512-fp32-sa-len-amp: prepare-nlp
	ssh -p$(PORT) -i $(KEY_FILE) $(USER)@localhost 'cd $(BERT_DIR) && CONFIG=configurations/64k-512-fp32-sa-len-amp.yml bash start_pretrain.sh'
