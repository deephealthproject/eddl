
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
EPOCHS=5
LR=0.0001
METHOD=0
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
NAME_COMMON=${DS}_m${METHOD}_n${PROCS}_bs${BS}

MPI_COMMON="-map-by node:PE=28 --mca btl_openib_verbose 1 --mca pml_ucx_verbose 1"

# Avoid message about IBA initialization
MPI_MACHINE1="${MPI_COMMON} --mca btl ^openib"
# Force UCX
MPI_MACHINE2="${MPI_COMMON} --mca pml ucx --mca btl ^openib"
# Force InfiniBand
MPI_MACHINE3="${MPI_COMMON} --mca btl_tcp_if_include ib0"
# Avoid InfiniBand
MPI_MACHINE4="${MPI_COMMON} --mca btl_tcp_if_exclude ib0"

#MPI_MACHINE="-map-by node:PE=28 --mca pml ucx -x UCX_TLS=rc,sm,cuda_copy,gdr_copy,cuda_ipc"
#MPI_MACHINE="-map-by node:PE=28 --mca pml ucx --mca coll basic,libnbc,inter,self,cuda,self"
#MPI_MACHINE="-map-by node:PE=28 --mca pml ucx --mca btl ^openib"


for i in 1 2 3 4
do

NAME_I=MPI${i}_${NAME_COMMON}

if [[ $i == 1 ]]; then
MPI_MCA=$MPI_MACHINE1
elif [[ $i == 2 ]]; then
MPI_MCA=$MPI_MACHINE2
elif [[ $i == 3 ]]; then
MPI_MCA=$MPI_MACHINE3
elif [[ $i == 4 ]]; then
MPI_MCA=$MPI_MACHINE4
elif [[ $i == 5 ]]; then
MPI_MCA=$MPI_MACHINE5
elif [[ $i == 6 ]]; then
MPI_MCA=$MPI_MACHINE6
elif [[ $i == 7 ]]; then
MPI_MCA=$MPI_MACHINE7
elif [[ $i == 8 ]]; then
MPI_MCA=$MPI_MACHINE8
elif [[ $i == 9 ]]; then
MPI_MCA=$MPI_MACHINE
fi

for OPTION in "mpi" "nca" "nccl"
#for OPTION in "mpi"
do

if [[ "$OPTION" == "mpi" ]]; then
NAME=mpi_$NAME_I
EDDL_EXEC="$BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8 --mpi"
elif [[ "$OPTION" == "nca" ]]; then
NAME=nca_$NAME_I
EDDL_EXEC="$BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8 --nca"
else 
NAME=nccl_$NAME_I
EDDL_EXEC="$BIN/generic_distr -p $DATASETS/$DS -n $MODEL $PARAMS -l $LR -a $AVG -b $BS -e $EPOCHS -8"
fi

OUTPUT=$NAME.out
ERR=$NAME.err

#MPI_PARAM="--report-bindings -map-by node:PE=28 --mca btl openib,self,vader --mca btl_openib_allow_ib true --mca mpi_leave_pinned 1"
#MPI_PARAM="--report-bindings -map-by node:PE=28 --mca btl openib,self,vader --mca btl_openib_allow_ib true"
#MPI_PARAM="--report-bindings -map-by node:PE=28 --mca pml ucx "

# Common MPI and sbatch commands
MPI_COMMON="--report-bindings"

SBATCH="sbatch -n ${PROCS} -N ${PROCS} --out ${OUTPUT} --err ${ERR} -J ${FILENAME} --exclusive"

# MPI command line parameters
MPI_PARAM="${MPI_MCA} $MPI_COMMON"

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

done

done
