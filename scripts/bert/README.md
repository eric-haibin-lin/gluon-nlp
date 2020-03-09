## Run the script with 2 p3dn.24xlarge nodes (8 GPUs each)
- pip install clush
- passwordless ssh between hosts
- git clone -b nvidia https://github.com/eric-haibin-lin/gluon-nlp $CONTAINER_SHARED_FS/gluon-nlp, where $CONTAINER_SHARED_FS/gluon-nlp is a shared file system mounted on all hosts
- copy phase1 checkpoint file [0014076.params](https://dist-bert.s3.amazonaws.com/demo/pretrain/phase2/0014076.params) to $CONTAINER_SHARED_FS/gluon-nlp/scripts/bert/ckpt-dir

## Change the number of nodes
- edit phase2.config. If you run with more nodes, you need to proportionally decrease the value for accumulation (BERT_PHASE2_ACC).
- update hosts, and hosts.mpi

Note
- the docker is built with github.com:eric-haibin-lin/docker/mxnet-benchmark/Dockerfile
- MXNet is build with https://github.com/eric-haibin-lin/mxnet/tree/apache-mirror, which is 1 commit ahead of MXNet 1.6.x branch
