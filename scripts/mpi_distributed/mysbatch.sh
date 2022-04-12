#!/bin/bash

GPN=1


if [ $# -le 1 ]; then
echo "Error: faltan parametros"
exit
fi

while  [ $# -ge 2 ]
do
	case $1 in 
		-n) PROCS=$2 ; shift ; NODES=$PROCS ;;
		-N) NODES=$2 ; shift ;;
		-G) GRES="--gres=gpu:$2" ; shift ;;
		*) break ;;
	esac
	shift
done

EXEC=$1

if [ -z $EXEC ]; then
echo "Error: falta exec"
exit
fi

shift 
REM=$@

COMMAND="sbatch ${GRES} --exclusive -N $NODES -n $PROCS $EXEC $PROCS $REM"
echo $COMMAND
$COMMAND

