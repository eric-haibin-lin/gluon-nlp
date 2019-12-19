### CLUSTER VARIABLES
set -ex
# FIXME: this cannot be hostname
export DMLC_PS_ROOT_URI=172.31.4.79
export DMLC_PS_ROOT_PORT="${DMLC_PS_ROOT_PORT:-12332}"
export CONFIG="${CONFIG=-configurations/test.yml}"

export DMLC_NUM_WORKER=64
export DMLC_NUM_SERVER=32
num_physical_server=32
server_hosts=/home/ec2-user/server_32
worker_hosts=hosts_64_v2.hosts
#worker_hosts=hosts_32_v2.hosts

#export DMLC_NUM_WORKER=1
#export DMLC_NUM_SERVER=1
#num_physical_server=1
#server_hosts=localhost
#worker_hosts=localhost
#LAUNCHER="/usr/local/byteps/3rdparty/ps-lite/tests/test_kv_app_benchmark "
#SCHED_CMD="set -ex; $COMMON_ENV export DMLC_ROLE=scheduler; $LAUNCHER"
#SERVER_CMD="set -ex; $SERVER_ENV export DMLC_ROLE=server; $LAUNCHER"

worker_docker=haibinlin/worker_mxnet:bps-7cde360-mx-cu100-016e6d56
server_docker=haibinlin/worker_mxnet:bps-7cde360-mx-cu100-016e6d56

HOME=/home/ec2-user
USERNAME=haibin
NLP_HOME="/efs/$USERNAME/container"

docker pull "$server_docker"
clush --hostfile $server_hosts 'sudo pkill python; sudo pkill sleep; docker kill $(docker ps -q); docker pull "$server_docker"'
clush --hostfile $worker_hosts 'sudo pkill python; sudo pkill sleep; docker kill $(docker ps -q); docker pull "$worker_docker"'

### BYTEPS ENV VAR
COMMON_ENV="export DMLC_NUM_WORKER=$DMLC_NUM_WORKER; \
            export DMLC_NUM_SERVER=$DMLC_NUM_SERVER; \
            export DMLC_PS_ROOT_URI=$DMLC_PS_ROOT_URI; \
            export DMLC_PS_ROOT_PORT=$DMLC_PS_ROOT_PORT;"

SERVER_ENV="$COMMON_ENV \
            export SERVER_PUSH_NTHREADS=1; \
            export MXNET_OMP_MAX_THREADS=8; \
            export MXNET_CPU_WORKER_NTHREADS=4;"

DOCKER="run -v $HOME/.ssh:/root/.ssh -v $HOME/efs/$USERNAME:/efs/$USERNAME -v /fsx:/data --network=host --shm-size=32768m"
LAUNCHER="/usr/local/byteps/launcher/launch.py"
SCHED_CMD="$COMMON_ENV export DMLC_ROLE=scheduler; python3 $LAUNCHER"
SERVER_CMD="$SERVER_ENV export DMLC_ROLE=server; python3 $LAUNCHER"

SCHED_TMUX="tmux new -d \"docker $DOCKER -d $server_docker bash -c '$SCHED_CMD'\""

ssh -o "StrictHostKeyChecking no" $DMLC_PS_ROOT_URI "$SCHED_TMUX"

num_server_iter=0
target_server_iter=$DMLC_NUM_SERVER/$num_physical_server
while true;
do
  if [[ $num_server_iter -ge $target_server_iter ]]
  then
    break
  fi
  SERVER_CMD_DOCKER="docker $DOCKER -d $server_docker bash -c '$SERVER_CMD'"
  clush --hostfile $server_hosts "$SERVER_CMD_DOCKER"
  let "num_server_iter+=1"
done;

WORKER_ENV="$COMMON_ENV export DMLC_ROLE=worker; export BACKEND='byteps'; "

count=0
while read -u 10 host;
do
  host=${host%% slots*}
  WORKER_CMD="cp -r $NLP_HOME ~/gluon-nlp; cd ~/gluon-nlp; python3 setup.py develop --user; $WORKER_ENV export DMLC_WORKER_ID=$count; cd scripts/bert; export CONFIG=$CONFIG; bash start_pretrain.sh $worker_docker; sleep infinity"
  echo $WORKER_CMD > .$count.sh
  WORKER_CMD_DOCKER="nvidia-docker $DOCKER --name byteps --rm $worker_docker bash $NLP_HOME/scripts/bert/.$count.sh; sleep infinity" 
  ssh -tt -o "StrictHostKeyChecking no" $host "tmux new -d \"$WORKER_CMD_DOCKER\""
  let "count+=1"
done 10<$worker_hosts;

clush --hostfile $server_hosts 'docker ps --no-trunc'
clush --hostfile $worker_hosts 'docker ps --no-trunc'
