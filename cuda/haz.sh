nvcc devicequery-mpi.cu -I/opt/openmpi-4.0.3/include -L/opt/openmpi-4.0.3/lib -lmpi -o devicequery-mpi
nvcc devicequery.cu -o devicequery

