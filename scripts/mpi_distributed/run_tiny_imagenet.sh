#!/bin/bash 

EDDL=$HOME/git/eddl
BUILD=$EDDL/build
BIN=$BUILD/bin
SCRIPTS=$EDDL/scripts/mpi_distributed
DATASETS=~/convert_EDDL
BS=400
EPOCHS=100
LR=0.0001
AVG=4
#MODEL=100
MODEL=10

PARAMS="-w 64 -h 64 -z 3 -c 200"
DS="tiny-imagenet64"

N=$1
BS=$2

NAME=distr_${DS}_n${N}_bs${BS}
OUTPUT=$NAME.out
ERR=$NAME.err
#MPI_PARAM="--report-bindings -map-by node:PE=28 --mca btl openib,self,vader --mca btl_openib_allow_ib true --mca mpi_leave_pinned 1"
#MPI_PARAM="--report-bindings -map-by node:PE=28 --mca btl openib,self,vader --mca btl_openib_allow_ib true"
#MPI_PARAM="--report-bindings -map-by node:PE=28 --mca pml ucx "
MPI_PARAM="--report-bindings -map-by node:PE=28 --mca btl ^openib"
# NO DISTR-DS
#Sin slurm
#mpirun -np 4 -hostfile $SCRIPTS/cluster.altec -map-by node $BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8 
#Con slurm
mpirun $MPI_PARAM $BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8 > $OUTPUT 2> $ERR
