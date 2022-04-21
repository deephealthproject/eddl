#!/bin/bash 

if [ $# -le 1 ]; then
echo "Error: faltan parametros"
exit
fi

# Defaults
SLURM="no"
FILENAME=$0

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

# Dataset specific
PARAMS="-w 256 -h 256 -z 1 -c 4"
DS="COVID-19_Radiography_Dataset"


# process arguments
while  [ $# -ge 2 ]
do
	case $1 in 
        --slurm) SLURM="yes"  ;;
		-n) PROCS=$2 ; shift ;;
		-bs) BS=$2 ; shift ;;
		*) break ;;
	esac
	shift
done

#echo "SLURM" ${SLURM}

# Filenames
NAME=FIX1_mpi_${DS}_n${PROCS}_bs${BS}
OUTPUT=$NAME.out
ERR=$NAME.err
EDDL_EXEC="$BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8 --mpi"


#MPI_PARAM="--report-bindings -map-by node:PE=28 --mca btl openib,self,vader --mca btl_openib_allow_ib true --mca mpi_leave_pinned 1"
#MPI_PARAM="--report-bindings -map-by node:PE=28 --mca btl openib,self,vader --mca btl_openib_allow_ib true"
#MPI_PARAM="--report-bindings -map-by node:PE=28 --mca pml ucx "

# Common MPI and sbatch commands
MPI_COMMON="--report-bindings"

SBATCH="sbatch -n ${PROCS} -N ${PROCS} --out ${OUTPUT} --err ${ERR} -J ${FILENAME} --exclusive"

# node specific options
if [[ "$HOSTNAME" =~ "altec" ]]; then
MPI_MACHINE="-map-by node:PE=28 --mca btl ^openib"
fi

if [[ "$HOSTNAME" =~ "cmts" ]]; then
MPI_MACHINE="-map-by node:PE=32 --mca btl_tcp_if_include ib1"
SBATCH="${SBATCH} --gres=gpu:1"
fi

# MPI command line parameters
MPI_PARAM="$MPI_MACHINE $MPI_COMMON"


## Run if file does not exists
if [ ! -f $OUTPUT ]
then


# Patches for slurm
if [[ "$SLURM" == "yes" ]]; then

COMMAND="mpirun $MPI_PARAM ${EDDL_EXEC}"

# Generate slurm sbatch file
echo "#!/bin/bash
#
$COMMAND" > $FILENAME.sbatch
############################

echo $SBATCH $FILENAME.sbatch
$SBATCH $FILENAME.sbatch

else
# Execute without slurm to stdout

COMMAND="mpirun -np $PROCS $MPI_PARAM -hostfile $SCRIPTS/cluster.nodos ${EDDL_EXEC}"
$COMMAND

fi

else

echo "File $OUTPUT already exists. Bye"

fi
