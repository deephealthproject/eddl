EDDL=$HOME/git/eddl
BUILD=$EDDL/build
BIN=$BUILD/bin
SCRIPTS=$EDDL/scripts/mpi_distributed
DATASETS=~/convert_EDDL
BS=80
EPOCHS=10
LR=0.0001
AVG=1
#MODEL=1
MODEL=10

PARAMS="-w 224 -h 224 -z 3 -c 2"
DS="tcga4"

# DISTR DS
mpirun -np 4 -hostfile $SCRIPTS/cluster.altec -map-by node $BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8 -d
