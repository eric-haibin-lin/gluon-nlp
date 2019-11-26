### CLUSTER VARIABLES
set -ex
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
# XXX: this cannot be hostname
export DMLC_PS_ROOT_URI=172.31.3.59
export DMLC_PS_ROOT_PORT="${DMLC_PS_ROOT_PORT:-12329}"

num_physical_server=1
server_hosts=localhost
worker_hosts=localhost

server_docker=haibinlin/byteps-server:c5fd6fc
worker_docker=haibinlin/worker_mxnet:bps-3291412-mx-cu100-87fe065

HOME=/home/ec2-user
USERNAME=chaokun

docker pull "$server_docker"
clush --hostfile $server_hosts 'sudo pkill python; sudo pkill sleep; docker kill $(docker ps -q); docker pull "$server_docker"'
clush --hostfile $worker_hosts 'sudo pkill python; sudo pkill sleep; docker kill $(docker ps -q); docker pull "$worker_docker"'

### BYTEPS ENV VARS
COMMON_ENV="export DMLC_NUM_WORKER=$DMLC_NUM_WORKER; \
            export DMLC_NUM_SERVER=$DMLC_NUM_SERVER; \
            export DMLC_PS_ROOT_URI=$DMLC_PS_ROOT_URI; \
            export DMLC_PS_ROOT_PORT=$DMLC_PS_ROOT_PORT;"

SERVER_ENV="$COMMON_ENV \
            export SERVER_PUSH_NTHREADS=1; \
            export MXNET_OMP_MAX_THREADS=8; \
            export MXNET_CPU_WORKER_NTHREADS=1;"

DOCKER="nvidia-docker run -v $HOME/.ssh:/root/.ssh -v $HOME/efs/$USERNAME:/efs/$USERNAME --network=host --shm-size=32768m"
LAUNCHER="/usr/local/byteps/launcher/launch.py"
NLP_HOME="/efs/$USERNAME/gluon-nlp"
SCHED_CMD="$SERVER_ENV export DMLC_ROLE=scheduler; python $LAUNCHER"
SERVER_CMD="$SERVER_ENV export DMLC_ROLE=server; python $LAUNCHER"

SCHED_TMUX="tmux new -d \"$DOCKER -d $server_docker bash -c '$SCHED_CMD'\""

ssh -o "StrictHostKeyChecking no" $DMLC_PS_ROOT_URI "$SCHED_TMUX"

num_server_iter=0
target_server_iter=$DMLC_NUM_SERVER/$num_physical_server
while true;
do
  if [[ $num_server_iter -ge $target_server_iter ]]
  then
    break
  fi
  SERVER_CMD_DOCKER="$DOCKER -d $server_docker bash -c '$SERVER_CMD'"
  clush --hostfile $server_hosts "$SERVER_CMD_DOCKER"
  echo "launched $num_physical_server servers"
  let "num_server_iter+=1"
done;

## TRAINING SCRIPT ARGUMENTS
CKPTDIR="/efs/$USERNAME/ckpt-test"

WORKER_ENV="$COMMON_ENV \
            export DATA=$NLP_HOME/scripts/bert/sample.txt; \
            export DATAEVAL=$NLP_HOME/scripts/bert/sample.txt; \
            export BYTEPS_TRACE_DIR=/efs/$USERNAME/bert_traces; \
            export CKPTDIR=$CKPTDIR;"

count=0
while read -u 10 host;
do
  host=${host%% slots*}
  WORKER_CMD="cd /efs/chaokun/byteps/; pip3 install . ;kpip3 install networkx --user; cd $NLP_HOME; python3 setup.py develop --user; $WORKER_ENV export DMLC_WORKER_ID=$count; cd scripts/bert; bash bps.sh; sleep infinity"
  WORKER_CMD_DOCKER="$DOCKER -d $worker_docker bash -c '$WORKER_CMD'"
  ssh -tt -o "StrictHostKeyChecking no" $host "tmux new -d \"$WORKER_CMD_DOCKER\""
  let "count+=1"
done 10<$worker_hosts;

clush --hostfile $server_hosts 'docker ps --no-trunc'
clush --hostfile $worker_hosts 'docker ps --no-trunc'
