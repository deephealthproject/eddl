#!/bin/bash 

EDDL=$HOME/git/eddl
BUILD=$EDDL/build
BIN=$BUILD/bin
SCRIPTS=$EDDL/scripts/mpi_distributed
DATASETS=~/convert_EDDL
BS=40
EPOCHS=100
LR=0.0001
AVG=2
MODEL=10

EXTRA="--mca btl_openib_want_cuda_gdr 1"

PARAMS="-w 32 -h 32 -z 3 -c 10"
DS="cifar10"


N=$1
BS=$2

NAME=distr_${DS}_n${N}_bs${BS}
OUTPUT=$NAME.out
ERR=$NAME.err
#MPI_PARAM="--report-bindings -map-by node:PE=28 --mca btl openib,self,vader --mca btl_openib_allow_ib true --mca mpi_leave_pinned 1"
#MPI_PARAM="--report-bindings -map-by node:PE=28 --mca btl openib,self,vader --mca btl_openib_allow_ib true"
#MPI_PARAM="--report-bindings -map-by node:PE=28 --mca pml ucx "
MPI_PARAM="--report-bindings -map-by node:PE=28 --mca btl ^openib --mca btl_openib_want_cuda_gdr 1"

# NO DISTR-DS
#mpirun -np 4 -hostfile $SCRIPTS/cluster.altec -map-by node $EXTRA $BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8 
# NO DISTR-DS
#Con slurm
mpirun $MPI_PARAM $BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8 > $OUTPUT 2> $ERR
