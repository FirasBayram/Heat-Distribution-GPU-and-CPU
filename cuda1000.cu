#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
/* Convert the index of 2D Matrix in 1D array*/
#define index(i, j, N)  ((i)*(N)) + (j)


/*****************************************************************/

/* Function declarations used in the program */
// CPU Implementaion of Heat Distribution -- Sequential 
void  cpu_heat(float *, unsigned int);
// GPU Implementaion of Heat Distribution -- Parallel 
void  gpu_heat(float *,  unsigned int );
// CUDA error checker
void  cuda_err(cudaError_t, const char *);
// CUDA Kernel to calculate heat distribution convergency 
__global__ void  heat_kernel(float *, float *, float *,unsigned int);
// Input Device you want to use
void device_Input(int*);
// Input Block Size for GPU
void block_Input( int*);



/*****************************************************************/
// Start Main
int main(int argc, char * argv[])
{

  unsigned int N = 1002; //Dimensions of the matrix 
  int device_used;
  device_Input(&device_used);  // Device Used --- 0 for CPU and 1 for GPU
  int i,j;

   float * mtx;// The matrix that holds the temperature of the points
	mtx = (float *)calloc(N*N, sizeof(float)); // Allocate matrix of size N*N in the memory
	
	/* Initialize The Matrix with the initial points temperature */
  
	// Set all points to 0
  for(i = 0; i < N; i++)
	  for (j=0; j<N; j++)
    mtx[index(i,j,N)] = 0;
	//Set Top Edge to 25C
  for(i = 0; i < N; i++)
    mtx[index(0,i,N)] = 25;
	//Set Right Edge to 25C
  for(i = 0; i < N; i++)
    mtx[index(i,N-1, N)] = 25;

	//define component A
  for (i = 100; i <= 200; i++ )     
        for (j = 200; j <= 400; j++) 
            mtx[index(i,j, N)] = 90;
        
    
	//define component B
    for (i = 600; i <= 700; i++ )   
        for (j = 800; j <= 900; j++) 
            mtx[index(i,j, N)] = 60;

	//define fan

    for (i = 500; i <= 800; i++ )     
        for (j = 200; j <= 300; j++) 
            mtx[index(i,j, N)] = 20;
        
   // Variables to calculate time elapsed to execute the program
  double time_elapsed;
  clock_t start, end;

  if( device_used ==0) // The CPU sequential Implementaion
  {
    start = clock();
    cpu_heat(mtx, N);
    end = clock();
  }
  else  // The GPU version
  {
     start = clock();
     gpu_heat(mtx, N);
     end = clock();
  }
	//Calculate time elapsed
  time_elapsed = ((double)(end - start))/ CLOCKS_PER_SEC;

  printf("Time elapsed for %s is %lf\n", device_used == 0? "CPU" : "GPU", time_elapsed);

  free(mtx);

  return 0;

}


/****************************************************/
//CPU Implementaion
void  cpu_heat(float * mtx, unsigned int N)
{
  printf("CPU VERSION STARTED\n");
  int i, j;
  int pts=0; //Number of points converged 
  unsigned int NBytes = N*N*sizeof(float);//Matrix size in Bytes
  float * temp;//Matrix hols the new values of the matrix
  float * diff;//Matrix holds the difference of the values
  temp = (float *)calloc(N*N, sizeof(float));
   diff = (float *)calloc(N*N, sizeof(float));
	// Initialize Temp with Matrix
  memcpy((void *)temp, (void *) mtx, NBytes);
	//While not all the points converged 
  while (pts<(N-2)*(N-2))
  {
	  pts =0;
    /* Calculate the difference of the temperature at each point and store it in diff */
    for(i = 1; i < N-1; i++)
      for(j = 1; j < N-1; j++){
	temp[index(i,j,N)] = (mtx[index(i-1,j,N)] +
	                      mtx[index(i+1,j,N)] +
			      mtx[index(i,j-1,N)] +
				mtx[index(i,j+1,N)])/4.0;
		diff[index(i,j,N)] = fabs(temp[index(i,j,N)]-mtx[index(i, j, N)]);
	  }
	// Copy the new values into the old matrix
	memcpy((void *)mtx, (void *) temp, NBytes);

    // Count how many points converged
		for (i = 1; i < N-1; i++ )
        {
            for (j = 1; j < N-1; j++)
            {
                   if(diff[index(i,j, N)] <0.001) pts++;
				   
            }
        }        
  }
  // Calculate the average temperature of the circuit after convergency
	float sum =0,avg;
  
         for (i = 0; i < N; i++ )
        {
            for (j = 0; j < N; j++)
            {
                   sum+=mtx[index(i,j,N)];
            }
        }
		
		//Calculate the average temperature
		avg = sum /(N*N);
		    printf("Temp Avg: %.4lg\n",avg);
}




/************************************************************/
// GPU Implementaion
void  gpu_heat(float * mtx, unsigned int N)
{
	/* Define Block Dimensions = BLOCK_SIZE*BLOCK_SIZE */
	int BLOCK_SIZE;
	block_Input(&BLOCK_SIZE);
	printf("GPU VERSION STARTED\n");
	float * diff;//Matrix holds the difference of the values
	diff = (float *)calloc(N*N, sizeof(float)); //
	unsigned int NBytes = N*N*sizeof(float);//Matrix size in Bytes
	int pts=0; // Number of points converged
	int i,j;

	float *d_temp = NULL, *d_mtx = NULL, *d_diff=NULL;//Device Matrices
	//CUDA memory allocation
	cudaError_t error;
	error = cudaMalloc((void**)&d_mtx, NBytes);
	error = cudaMalloc((void**)&d_temp, NBytes);
	error = cudaMalloc((void**)&d_diff, NBytes);
	cuda_err(error, "allocating memory on device.");
	//Copy Matrices to device
	error = cudaMemcpy(d_temp, mtx, NBytes, cudaMemcpyHostToDevice);
	error = cudaMemcpy(d_mtx, mtx, NBytes, cudaMemcpyHostToDevice);
	cuda_err(error, "copying array to device memory.");
	// block and grid dimensions
	dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 grid(N/BLOCK_SIZE + 1, N/BLOCK_SIZE + 1, 1);
	/*for 2D Grid and 1D Blocks
	dim3 block(BLOCK_SIZE);
	dim3 grid((N + block.x - 1) / block.x,N); */
	
	//While not all the points converged 
  while (pts<(N-2)*(N-2))
	{
		pts=0;
		//Invoke kernel
		heat_kernel<<<grid, block>>>(d_mtx, d_temp,d_diff, N);
		cudaThreadSynchronize();
		// update the orginal matrix
		error = cudaMemcpy(d_mtx, d_temp, NBytes,cudaMemcpyDeviceToDevice);
		// copy kernel result back to host side
        error = cudaMemcpy(diff, d_diff, NBytes,cudaMemcpyDeviceToHost);
		// Count how many points converged
		for (i = 1; i < N-1; i++ )
        {
            for (j = 1; j < N-1; j++)
            {
                   if(diff[index(i,j, N)] <0.001) pts++;
				   
            }
        }        
	}
		// copy the final matrix  back to host side
		error = cudaMemcpy(mtx, d_mtx, NBytes, cudaMemcpyDeviceToHost);
		cuda_err(error, "copying array back to host.");
		// free device global memory
        cudaFree(d_mtx);
        cudaFree(d_temp);
	 // Calculate the average temperature of the circuit after convergency
	float sum =0,avg;
         for (i = 0; i < N; i++ )
			{
				for (j = 0; j < N; j++)
				{
					   sum+=mtx[index(i,j,N)];
			
				}
			}
		
		//Calculate the average temperature
		avg = sum /(N*N);
		printf("Temp Avg: %.4lg\n",avg);
			
		printf("Execution configuration  <<<((%d,%d)>>>\n", block.x, block.y);
}


/*******************************************************************/
//Kernel to calculate the temperature difference at each point
__global__
void heat_kernel(float *d_mtx, float *d_temp,float *d_diff, unsigned int N)
{
	unsigned int i, j;
	i = threadIdx.x + blockIdx.x * blockDim.x;
	j = threadIdx.y + blockIdx.y * blockDim.y;
	//for 2D Grid and 1D Blocks
	//j = blockIdx.y;
	if (i > 0 && i < N-1 && j > 0 && j < N-1)
	{
		d_temp[index(i,j,N)] = (d_mtx[index(i-1,j,N)] +
				d_mtx[index(i+1,j,N)] +
				d_mtx[index(i,j-1,N)] +
				d_mtx[index(i,j+1,N)])/4.0;
			__syncthreads();
		d_diff[index(i,j,N)] = fabs(d_temp[index(i,j,N)]-d_mtx[index(i, j, N)]);
						
	}
}
//CUDA error checker
void  cuda_err(cudaError_t error, const char *msg)
{
	if (error != cudaSuccess)
	{
		fprintf(stderr, "CUDA Error: %s\n", msg);
		exit(1);
	}
}

// Input Device you want to use
void device_Input(int* device_used ) {
	printf("Enter the Device You want to use 0 for CPU 1 for GPU:\n");
	scanf("%d", device_used); 
}

// Input Block Size fro GPU
void block_Input( int* BLOCK_SIZE) {
	printf("Enter the Number of Block Size:\n");
	scanf("%d", BLOCK_SIZE);
}