EDDL=$HOME/git/eddl
BUILD=$EDDL/build
BIN=$BUILD/bin
SCRIPTS=$EDDL/scripts/mpi_distributed
DATASETS=~/convert_EDDL
BS=20
EPOCHS=20
LR=0.0001
AVG=1
MODEL=10

PARAMS="-w 80 -h 60 -z 3 -c 9"
DS="marino"

# NO DISTR-DS
mpirun -np 4 -hostfile $SCRIPTS/cluster.altec -map-by node $BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8 
