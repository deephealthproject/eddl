#!/bin/bash
#======

if [ $# -le 1 ]; then
echo "Error: faltan parametros"
exit
fi

# defaults
SLURM="no"
PROCS=1
NODES=1
BS=100
PE=24
RESULT=result




# process arguments

while  [ $# -ge 2 ]
do
	case $1 in 
	    --slurm) SLURM="yes" ;;
		-n) PROCS=$2 ; NODES=$PROCS ; shift ;;
		-N) NODES=$2 ; shift ;;
		-G) GRES="#SBATCH --gres=gpu:$2" ; shift ;;
		--exec) EXEC=$2 ; shift ;;
#		--bs) BS=$2 ; shift ;;
#    	--dataset) DATASET=$2 ; shift ;;
		--PE) PE=$2 ; shift ;;
#        --workers) WORKERS=$2; shift ;;
        --out) RESULT=$2; shift ;;
		*) break ;;
	esac
	shift
done

RESULT=${RESULT}_n${PROCS}

REM=$@
echo $REM


#MPI_PARAM="--report-bindings -map-by socket:PE=${PE} --mca btl ^openib"
MPI_PARAM_SLURM="--report-bindings -map-by node:PE=${PE} --mca btl ^openib"
MPI_PARAM="--report-bindings --hostfile cluster.altec -map-by node:PE=${PE} --mca btl ^openib"


if [ -z $EXEC ]; then
echo "Error: falta exec"
exit
fi

if [[ "$SLURM" == "yes" ]]; then
echo "#!/bin/bash
#
# Project/Account (use your own) el nombre que aparece en squeue
#SBATCH -A plopez            
#
# Number of MPI tasks cuantos procesos mpi vas a utilizar. No utilizado
#SBATCH -n ${PROCS}
#SBATCH -N ${NODES}
${GRES}
#
# Number of tasks per node
##SBATCH --tasks-per-node=1 cuantos procesos de mpi quieres por nodo. No utilizado
#
# Tiempo de ejecución máximo
#SBATCH --time=10:00:00
#
# Name Nombre del trabajo
#SBATCH -J ${EXEC}
#
# Partition cola donde se ejecutará. No utilizado
##SBATCH --partition=mpi
#
# stdout 
#SBATCH --output=${RESULT}.out
#
# stderr
#SBATCH --error=${RESULT}.err
#
#SBATCH --exclusive
#
mpirun ${MPI_PARAM_SLURM} ${EXEC} ${REM}
#
" > ${RESULT}.sbatch


#bin/uc15_mlp.exe --cpus 4 --epochs 20 --batch-size 100 --data-dir dataset/5x5
#bin/uc15_mlp.exe --cpus 4 --epochs 20 --batch-size 100 --data-dir dataset/7x7

COMMAND="sbatch ${RESULT}.sbatch"
echo $COMMAND
$COMMAND

else

COMMAND="mpirun -np ${PROCS} ${MPI_PARAM} ${EXEC} ${REM}"
echo $COMMAND
$COMMAND

fi
