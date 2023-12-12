DMRFLAGS 	= -I${DMR_PATH} -L${DMR_PATH} -ldmr
FLAGS  		= -O3 -Wall -fopenmp
MPIFLAGS    = -I/apps/dmr/mpich/include -L/apps/dmr/mpich/lib -lmpi
SLURMFLAGS  = -I${DMR_PATH}/slurm-install/include -L${DMR_PATH}/slurm-install/lib -lslurm

jacobi: jacobi.c
	mpic++ $(FLAGS) $(DMRFLAGS) $(SLURMFLAGS) jacobi.c -o jacobi

clean:
	rm -f *.out *.o core.* *.err *.so tmp hostfile.txt jacobi
