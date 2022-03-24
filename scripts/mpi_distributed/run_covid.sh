EDDL=$HOME/git/eddl
BUILD=$EDDL/build
BIN=$BUILD/bin
SCRIPTS=$EDDL/scripts/mpi_distributed
DATASETS=~/convert_EDDL
BS=40
EPOCHS=2
LR=0.0001
AVG=16
MODEL=10

PARAMS="-w 512 -h 512 -z 1 -c 3"
DS="covid19"

# NO DISTR-DS
mpirun -np 4 -hostfile $SCRIPTS/cluster.altec -map-by node $BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8 
