EDDL=$HOME/git/eddl
BUILD=$EDDL/build
BIN=$BUILD/bin
SCRIPTS=$EDDL/scripts/mpi_distributed
DATASETS=~/convert_EDDL
BS=24
EPOCHS=2
LR=0.0001
AVG=16
MODEL=10

PARAMS="-w 224 -h 224 -z 1 -c 2"
DS="chest_xray"

# NO DISTR-DS
mpirun -np 4 -hostfile $SCRIPTS/cluster.altec -map-by node $BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8 
