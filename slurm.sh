#!/bin/bash
#======
#
# Project/Account (use your own) el nombre que aparece en squeue
#SBATCH -A plopez            
#
# Number of MPI tasks cuantos procesos mpi vas a utilizar. No utilizado
##SBATCH -n 2
#
# Number of tasks per node
##SBATCH --tasks-per-node=1 cuantos procesos de mpi quieres por nodo. No utilizado
#
# Tiempo de ejecución máximo
#SBATCH --time=01:00:00
#
# Name Nombre del trabajo
#SBATCH -J "nnsim"
#
# Partition cola donde se ejecutará. No utilizado
##SBATCH --partition=mpi

#Output fichero de salida (por defecto será slurm-numerodeltrabajo.out). No utilizado
##SBATCH --output=resnet50_1_nodo_2_inter.out


#=====

MPIRUN=mpirun
HOSTFILE="-hostfile cluster.altec"
IBA=
#OUTPUT="2>&1"
OUTPUT=""
EXEC=./build/bin/mnist_mlp
CORES=28

#====================

for np in 1 2 4 8
do
COMMAND="$MPIRUN -np $np $HOSTFILE --map-by node:PE=$CORES --report-bindings $IBA $EXEC"
#echo $COMMAND
time $COMMAND >mpi.$np-proc $OUTPUT
done


