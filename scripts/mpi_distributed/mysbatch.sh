#!/bin/bash

PROCS=$1
NODES=$2
GPN=$3
EXEC=$4
shift 4
REM=$@

#sbatch --gres=gpu:$GPN -N $NODES -n $PROCS $EXEC $PROCS $REM
echo sbatch --exclusive -N $NODES -n $PROCS $EXEC $PROCS $REM
sbatch --exclusive -N $NODES -n $PROCS $EXEC $PROCS $REM
