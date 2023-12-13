#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h> // include header file for openMPI
#include <omp.h> // include header file for openMP
#include "dmr.h" // include header file for dmr api

// size of plate
#define COLUMNS 10000    // number of columns without ghost cells
#define GLOBAL_ROWS 1000 // global row count
#define STEPS 5000       // maximum number of iterations
// communication tags
#define DOWN 100
#define UP 101
// largest permitted change in temp (this value takes about 3400 steps)
#define MAX_TEMP_ERROR 0.01

//  helper routines
void initialize_data(double **data, double **data_old, int world_size, int world_rank, int size);
void track_progress(int iter, int rows, double *data);
double compute(double *data, double *data_old, int world_size, int world_rank, int size);
void finalize(double *data, double *data_old);
// dmr function definitions
void send_shrink(double *data_old, int size);
void recv_shrink(double **data, double **data_old, int *size);
void send_expand(double *data_old, int size);
void recv_expand(double **data, double **data_old, int *size);

int main(int argc, char *argv[])
{
    struct timeval start_time, stop_time, elapsed_time; // timers
    int rows;                                           // rows for each piece
    double dt_global = 100;                             // delta t across all PEs

    /*
    // only for evaluation purpose
    time_t rawtime;
    struct tm * timeinfo;
    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    printf ( "Current local time and date: %s", asctime (timeinfo) );
    // end print time
    */

    // the usual MPI startup routines
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int len;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(name, &len);

    fflush(stdout);

    // define old variables world_size and rows;
    rows = (int)GLOBAL_ROWS / world_size;
    if (world_rank == (world_size - 1))
    {
        rows += GLOBAL_ROWS % world_size;
    }

    // now we can define data variables for temperature

    double *Temperature;      // temperature grid
    double *Temperature_last; // latest iteration of temperature grid

    // print off processor names
    if (world_rank == 0)
        printf("This is processor %s, rank %d out of %d processors and my rows are %d\n", name, world_rank, world_size, rows);
    // #pragma omp parallel
    //  information print off by only main thread
    // printf("Number of threads used by OpenMP: %d\n", omp_get_num_threads());

    if (world_rank==0) gettimeofday(&start_time, NULL);

    // DMR function configuration
    DMR_INIT(initialize_data(&Temperature, &Temperature_last, world_size, world_rank, rows),
             recv_expand(&Temperature, &Temperature_last, &rows),
             recv_shrink(&Temperature, &Temperature_last, &rows));
    DMR_Inhibit_iter(100);          // DMR is only called every 100 iterations
    DMR_Set_parameters(1, 8, 0, 2); // min processors, max processors, without function for random policy, powers of 2

    // DMR_it is initialized to 0 internally
    while (dt_global > MAX_TEMP_ERROR && DMR_it <= STEPS)
    {
        dt_global = compute(Temperature, Temperature_last, world_size, world_rank, rows);
        DMR_it++;
        DMR_RECONFIGURATION(send_expand(Temperature_last, rows), send_shrink(Temperature_last, rows));

        /*
        // periodically print test values - only for PE in lower corner
        if((DMR_it % 100) == 0 )
            if(world_rank == world_size -1)
                track_progress(DMR_it, rows, Temperature);
        */
    }

    // slightly more accurate timing and cleaner output
    MPI_Barrier(MPI_COMM_WORLD);

    // PE 0 finish timing and output values
    if (world_rank == 0)
    {
        gettimeofday(&stop_time, NULL);
        timersub(&stop_time, &start_time, &elapsed_time);

        printf("\nMax error at iteration %d was %f\n", DMR_it, dt_global);
        printf("Total time was %f seconds.\n", elapsed_time.tv_sec + elapsed_time.tv_usec / 1000000.0);
    }

    DMR_FINALIZE(finalize(Temperature, Temperature_last));
    MPI_Finalize();
}

// function for dt calculation
double compute(double *data, double *data_old, int world_size, int world_rank, int size)
{
    int i, j;        // grid indexes
    double incr, dt; // temporary variable for dt
    double dt_global;
    MPI_Status status;

// main calculation: average of my four neighbors
#pragma omp parallel for private(j)
    for (i = 1; i <= size; i++)
    {
        for (j = 1; j <= COLUMNS; j++)
        {
            data[i * (COLUMNS + 2) + j] = 0.25 * (data_old[(i + 1) * (COLUMNS + 2) + j] + data_old[(i - 1) * (COLUMNS + 2) + j] +
                                                  data_old[i * (COLUMNS + 2) + j + 1] + data_old[i * (COLUMNS + 2) + (j - 1)]);
        }
    }

    // COMMUNICATION PHASE: send and receive ghost rows for next iteration

    // send bottom real row down
    if (world_rank != world_size - 1) // unless we are bottom PE
    {
        MPI_Send(&data[size * (COLUMNS + 2) + 1], COLUMNS, MPI_DOUBLE, world_rank + 1, DOWN, MPI_COMM_WORLD);
    }

    // receive the bottom row from above into our top ghost row
    if (world_rank != 0) // unless we are top PE
    {
        MPI_Recv(&data[1], COLUMNS, MPI_DOUBLE, world_rank - 1, DOWN, MPI_COMM_WORLD, &status);
    }

    // send top real row up
    if (world_rank != 0) // unless we are top PE
    {
        MPI_Send(&data[COLUMNS + 2 + 1], COLUMNS, MPI_DOUBLE, world_rank - 1, UP, MPI_COMM_WORLD);
    }

    // receive the top row from below into our bottom ghost row
    if (world_rank != world_size - 1) // unless we are bottom PE
    {
        MPI_Recv(&data[(size + 1) * (COLUMNS + 2) + 1], COLUMNS, MPI_DOUBLE, world_rank + 1, UP, MPI_COMM_WORLD, &status);
    }

    dt = 0.0;
#pragma omp parallel for reduction(max : dt) private(j, incr)
    for (i = 1; i <= size; i++)
    {
        for (j = 1; j <= COLUMNS; j++)
        {
            // using ternary operators improves the execution time a little
            incr = data[i * (COLUMNS + 2) + j] - data_old[i * (COLUMNS + 2) + j];
            incr = (incr > 0) ? incr : incr * (-1);
            dt = (incr > dt) ? incr : dt;
            data_old[i * (COLUMNS + 2) + j] = data[i * (COLUMNS + 2) + j];
        }
    }

    // find global dt
    MPI_Reduce(&dt, &dt_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt_global, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return dt_global;
}

// initialize plate and boundary conditions
// Temp_last is used to start first iteration
void initialize_data(double **data, double **data_old, int world_size, int world_rank, int size)
{
    double tMin, tMax; // local boundary limits

    (*data) = (double *)malloc((size + 2) * (COLUMNS + 2) * sizeof(double));
    (*data_old) = (double *)malloc((size + 2) * (COLUMNS + 2) * sizeof(double));

    int i, j;
    for (i = 0; i <= size + 1; i++)
    {
        for (j = 0; j <= COLUMNS + 1; j++)
        {
            (*data_old)[i * (COLUMNS + 2) + j] = 0.0;
        }
    }

    // local boundary condition endpoints
    tMin = (world_rank) * 100.0 / world_size;
    tMax = (world_rank + 1) * 100.0 / world_size;

    // set left and right boundaries
    for (i = 0; i <= size + 1; i++)
    {
        (*data_old)[i * (COLUMNS + 2) + 0] = 0.0;
        (*data_old)[i * (COLUMNS + 2) + COLUMNS + 1] = tMin + ((tMax - tMin) / size) * i;
    }

    // set top boundary (PE 0 only)
    if (world_rank == 0)
    {
        for (j = 0; j <= COLUMNS + 1; j++)
        {
            (*data_old)[j] = 0.0;
        }
    }

    // set bottom boundary (last PE only)
    if (world_rank == world_size - 1)
    {
        for (j = 0; j <= COLUMNS + 1; j++)
        {
            (*data_old)[(size + 1) * (COLUMNS + 2) + j] = (100.0 / COLUMNS) * j;
        }
    }
}

// only called by last PE
// print diagonal in bottom right corner where most action is
void track_progress(int DMR_it, int size, double *data)
{
    int i;

    printf("---------- Iteration number: %d ----------\n", DMR_it);

    // output global coordinates so user does not have to understand decomposition
    for (i = 5; i >= 0; i--)
    {
        printf("[%d, %d]: %5.2f ", GLOBAL_ROWS - i, COLUMNS - i, data[(size - i) * (COLUMNS + 2) + (COLUMNS - i)]);
    }
    printf("\n");
}

// dmr function implementation
void send_shrink(double *data_old, int size)
{
    int world_rank, comm_size, intercomm_size, factor, dst;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);           // get rank number
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);            // get size of the group
    MPI_Comm_remote_size(DMR_INTERCOMM, &intercomm_size); // get size of the new group
    factor = comm_size / intercomm_size;                  // calculate factor of resize
    dst = world_rank / factor;

    // synchronous and blocking sending to guarantee the sending and receiving of data
    MPI_Ssend(data_old, (size + 2) * (COLUMNS + 2), MPI_DOUBLE, dst, 0, DMR_INTERCOMM);
}

void recv_shrink(double **data, double **data_old, int *size)
{
    int world_rank, comm_size, parent_size, src, factor, iniPart, i, j, number_amount, data_size;
    MPI_Status status;
    double *buffer_data;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_remote_size(DMR_INTERCOMM, &parent_size);
    factor = parent_size / comm_size;
    (*size) = 0;
    for (i = 0; i < factor; i++)
    {
        src = world_rank * factor + i;

        MPI_Probe(src, MPI_ANY_TAG, DMR_INTERCOMM, &status);
        MPI_Get_count(&status, MPI_DOUBLE, &number_amount);

        if (i == 0)
        {
            iniPart = 0;
            data_size = number_amount * (parent_size + 1); // estimated maximum size required
            (*data) = (double *)malloc(data_size * sizeof(double));
            (*data_old) = (double *)malloc(data_size * sizeof(double));
            buffer_data = (double *)malloc(number_amount * 2 * sizeof(double));
        }

        MPI_Recv(buffer_data, number_amount, MPI_DOUBLE, src, 0, DMR_INTERCOMM, MPI_STATUS_IGNORE);

        for (j = 0; j < number_amount; j++)
        {
            (*data_old)[j + iniPart] = buffer_data[j];
        }

        iniPart += (number_amount - 2 * (COLUMNS + 2));
        (*size) += (number_amount / (COLUMNS + 2) - 2);
    }
}

void send_expand(double *data_old, int size)
{
    int world_rank, comm_size, intercomm_size, factor, dst, iniPart;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_remote_size(DMR_INTERCOMM, &intercomm_size);

    factor = intercomm_size / comm_size;
    size = GLOBAL_ROWS / intercomm_size;

    for (int i = 0; i < factor; i++)
    {
        dst = world_rank * factor + i;
        iniPart = size * (COLUMNS + 2) * i;
        if (dst == intercomm_size - 1)
        {
            size += GLOBAL_ROWS % intercomm_size;
        }
        // synchronous and blocking sending to guarantee the sending and receiving of data
        MPI_Ssend(data_old + iniPart, (size + 2) * (COLUMNS + 2), MPI_DOUBLE, dst, 0, DMR_INTERCOMM);
    }
}

void recv_expand(double **data, double **data_old, int *size)
{
    int world_rank, comm_size, parent_size, src, factor, number_amount;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_remote_size(DMR_INTERCOMM, &parent_size);
    factor = comm_size / parent_size;
    src = world_rank / factor;

    MPI_Probe(src, MPI_ANY_TAG, DMR_INTERCOMM, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &number_amount);

    (*data) = (double *)malloc(number_amount * sizeof(double));
    (*data_old) = (double *)malloc(number_amount * sizeof(double));

    (*size) = (number_amount / (COLUMNS + 2)) - 2; // new number of rows without ghost cells

    MPI_Recv((*data_old), number_amount, MPI_DOUBLE, src, 0, DMR_INTERCOMM, MPI_STATUS_IGNORE);
}

void finalize(double *data, double *data_old)
{
    free(data);
    free(data_old);
}
