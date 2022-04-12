#!/bin/bash 

if [ $# -le 1 ]; then
echo "Error: faltan parametros"
exit
fi

SLURM="no"
FILENAME=$0

EDDL=$HOME/git/eddl
BUILD=$EDDL/build
BIN=$BUILD/bin
SCRIPTS=$EDDL/scripts/mpi_distributed
DATASETS=~/convert_EDDL
#BS=80
EPOCHS=20
LR=0.0001
AVG=1
MODEL=12

PARAMS="-w 64 -h 64 -z 1 -c 6"
DS="medical-mnist"

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

echo "SLURM" ${SLURM}

NAME=gpu_nccl_${DS}_n${PROCS}_bs${BS}
OUTPUT=$NAME.out
ERR=$NAME.err

#MPI_PARAM="--report-bindings -map-by node:PE=28 --mca btl openib,self,vader --mca btl_openib_allow_ib true --mca mpi_leave_pinned 1"
#MPI_PARAM="--report-bindings -map-by node:PE=28 --mca btl openib,self,vader --mca btl_openib_allow_ib true"
#MPI_PARAM="--report-bindings -map-by node:PE=28 --mca pml ucx "
MPI_PARAM="--report-bindings --mca btl ^openib"

if [[ "$HOSTNAME" =~ "altec" ]]; then
MPI_PARAM="-map-by node:PE=28 ${MPI_PARAM}"
fi


# NO DISTR-DS

EDDL_EXEC="$BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8"

if [[ "$SLURM" == "yes" ]]; then

SBATCH="sbatch -n ${PROCS} -N ${PROCS} --out ${OUTPUT} --err ${ERR} -J ${FILENAME} --exclusive"
COMMAND="mpirun $MPI_PARAM ${EDDL_EXEC}"

echo "#!/bin/bash
#
$COMMAND" > $FILENAME.sbatch

echo $SBATCH $FILENAME.sbatch
$SBATCH $FILENAME.sbatch

else 

COMMAND="mpirun -np $PROCS -hostfile $SCRIPTS/cluster.altec ${EDDL_EXEC}"
$COMMAND

fi

