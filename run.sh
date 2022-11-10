#! /bin/bash
module load GCC/8.2.0-2.31.1
module load cuda
nvcc -O3 -arch=sm_70 -o galaxy galaxy.cu
srun -p gpu -n 1 -t 10:00 --mem=1G -e err.txt -o out.txt galaxy data_100k_arcmin.dat rand_100k_arcmin.dat omega.out
echo "Done."