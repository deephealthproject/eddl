EDDL=$HOME/git/eddl
BUILD=$EDDL/build
BIN=$BUILD/bin
SCRIPTS=$EDDL/scripts/mpi_distributed
DATASETS=~/convert_EDDL
BS=80
EPOCHS=10
LR=0.0001
AVG=1
MODEL=10

PARAMS="-w 256 -h 256 -z 1 -c 4"
DS="COVID-19_Radiography_Dataset"

# NO DISTR-DS
#mpirun -np 4 -hostfile $SCRIPTS/cluster.altec -map-by node $BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8 
mpirun $BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8 
