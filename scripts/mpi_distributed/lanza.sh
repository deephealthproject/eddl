
BS=32
for N in 1 2 4 8
do
./run_covid19.sh --slurm --mpi -n $N -bs $BS
done

BS=64
for N in 1 2 4 8
do
./run_covid19.sh --slurm --mpi -n $N -bs $BS
done
