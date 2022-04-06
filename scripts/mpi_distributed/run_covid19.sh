#!/bin/bash 

EDDL=$HOME/git/eddl
BUILD=$EDDL/build
BIN=$BUILD/bin
SCRIPTS=$EDDL/scripts/mpi_distributed
DATASETS=~/convert_EDDL
#BS=80
EPOCHS=10
LR=0.0001
AVG=1
MODEL=10

PARAMS="-w 256 -h 256 -z 1 -c 4"
DS="COVID-19_Radiography_Dataset"

N=$1
BS=$2

NAME=distr_n${N}_bs${BS}
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
