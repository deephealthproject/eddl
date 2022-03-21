# SPLIT

#mpirun -np 2 -hostfile cluster.altec -map-by node ./bin/generic_mlp_distr -m bOCT64 -w 64 -h 64 -z 1 -c 4 -l 0.001 -a 1 -b 100  -8 
#mpirun -np 2 -hostfile cluster.altec -map-by node ./bin/generic_mlp_distr -m bOCT64_split -w 64 -h 64 -z 1 -c 4 -l 0.001 -a 1 -b 100  -8 -s 4

# DISTRIBUTED

#mpirun -np 2 -hostfile cluster.altec -map-by node ./bin/generic_mlp_distr -m bOCT64_split -w 64 -h 64 -z 1 -c 4 -l 0.001 -a 1 -b 100  -8 -d

# SPLIT

#echo "===== SPLIT ====="
#mpirun -np 2 -hostfile cluster.altec -map-by node -mca pls_rsh_agent "ssh -X -n" xterm -hold -e ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/bOCT64_split -w 64 -h 64 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8 -s 4 

# DISTRIBUTED

#read
#echo "===== DISTRIBUTED ====="
#mpirun -np 2 -hostfile cluster.altec -map-by node ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/bOCT64_split -w 64 -h 64 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8 -d

# SPLIT & DISTRIBUTED

#read
echo "===== SPLIT & DISTRIBUTED ====="
#mpirun -np 4 -hostfile cluster.altec -map-by node ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/OCT4 -w 224 -h 224 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8  -d

#mpirun -np 2 -hostfile cluster.altec -map-by node ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/OCT4 -w 224 -h 224 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8  -s 2 -d


# np=2

# non distr DS
mpirun -np 2 -hostfile cluster.altec -map-by node ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/OCT -w 224 -h 224 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8  

# distr DS
mpirun -np 2 -hostfile cluster.altec -map-by node ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/OCT2 -w 224 -h 224 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8 -d

# non distr DS, split
mpirun -np 1 -hostfile cluster.altec -map-by node ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/OCT2 -w 224 -h 224 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8 -s 2


# np= 4
# non distr DS
mpirun -np 4 -hostfile cluster.altec -map-by node ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/OCT -w 224 -h 224 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8  

# distr DS
mpirun -np 4 -hostfile cluster.altec -map-by node ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/OCT4 -w 224 -h 224 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8  -d

# non distr DS, split
mpirun -np 2 -hostfile cluster.altec -map-by node ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/OCT4 -w 224 -h 224 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8 -s 2

# non distr DS, split
mpirun -np 1 -hostfile cluster.altec -map-by node ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/OCT4 -w 224 -h 224 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8 -s 4



# np= 8
# non distr DS
mpirun -np 8 -hostfile cluster.altec -map-by node ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/OCT -w 224 -h 224 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8  

# distr DS
mpirun -np 8 -hostfile cluster.altec -map-by node ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/OCT8 -w 224 -h 224 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8  -d

# non distr DS, split
mpirun -np 4 -hostfile cluster.altec -map-by node ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/OCT8 -w 224 -h 224 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8 -s 2

# non distr DS, split
mpirun -np 2 -hostfile cluster.altec -map-by node ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/OCT8 -w 224 -h 224 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8 -s 4

# non distr DS, split
mpirun -np 1 -hostfile cluster.altec -map-by node ../../build/bin/generic_vgg16_bn_split_distr -m ~/convert_EDDL/OCT8 -w 224 -h 224 -z 1 -c 4 -l 0.001 -a 1 -b 100  -e 2 -8 -s 8
