# SPLIT

mpirun -np 4 -hostfile ../cluster.altec -map-by node ./bin/generic_mlp_distr -m bOCT64 -w 64 -h 64 -z 1 -c 4 -l 0.001 -a 1 -b 100  -8 
mpirun -np 4 -hostfile ../cluster.altec -map-by node ./bin/generic_mlp_distr -m bOCT64_split -w 64 -h 64 -z 1 -c 4 -l 0.001 -a 1 -b 100  -8 -s 4

# DISTRIBUTED

mpirun -np 4 -hostfile ../cluster.altec -map-by node ./bin/generic_mlp_distr -m bOCT64_split -w 64 -h 64 -z 1 -c 4 -l 0.001 -a 1 -b 100  -8 -d

