EDDL=$HOME/git/eddl
BUILD=$EDDL/build
BIN=$BUILD/bin
SCRIPTS=$EDDL/scripts/mpi_distributed
DATASETS=~/convert_EDDL
BS=100
EPOCHS=2
LR=0.0001
AVG=16
MODEL=3

PARAMS="-w 224 -h 224 -z 3 -c 51"
DS="pintores"

# NO DISTR-DS
mpirun -np 4 -hostfile $SCRIPTS/cluster.altec -map-by node $BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8 
