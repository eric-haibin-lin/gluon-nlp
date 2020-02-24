Requirements
- pip install clush
- passwordless ssh between hosts
- git clone -b nvidia https://github.com/eric-haibin-lin/gluon-nlp $CONTAINER_SHARED_FS/gluon-nlp, where $CONTAINER_SHARED_FS/gluon-nlp is a shared file system mounted on all hosts
- edit phase2.config, host, and host.mpi
- copy phase1 checkpoint file 0014076.params to $CONTAINER_SHARED_FS/gluon-nlp/scripts/bert/ckpt-dir

Note
- the docker is built with github.com:eric-haibin-lin/docker/mxnet-benchmark/Dockerfile
- MXNet is build with https://github.com/eric-haibin-lin/mxnet/tree/apache-mirror, which is 1 commit ahead of MXNet 1.6.x branch
