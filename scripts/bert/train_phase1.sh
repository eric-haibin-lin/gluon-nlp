source parse_yaml.sh
CONFIG=$(parse_yaml phase1-256.config)
set -ex
eval $CONFIG

clush --hostfile $BERT_CLUSTER_HOST "cd $CONTAINER_SHARED_FS/gluon-nlp/scripts/bert; bash start_container.sh"
docker cp $CONTAINER_NAME:/home/cluster/.ssh/ssh_host_rsa_key key
ssh -o 'StrictHostKeyChecking=no' -p2022 -i key cluster@localhost "cd /data/gluon-nlp/scripts/bert; bash horovod_train.sh"
