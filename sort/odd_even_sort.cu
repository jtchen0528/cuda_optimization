#include <stdio.h>
#include <cuda.h>

//Device functions can only be called from other device or global functions. __device__ functions cannot be called from host code.

//Global functions are also called "kernels". It's the functions that you may call from the host side using CUDA kernel call semantics (<<<...>>>).
__global__ void testKernel(int *in, int *out, int size){

	bool oddeven = true;
	//__shared__ for shared memory
	__shared__ bool swappedodd;
	__shared__ bool swappedeven;
	
	int temp;

	while(1){
	
		if(oddeven == true){
			
/*
Using __syncthreads is sometimes necessary to ensure that all data from all threads is valid before threads read from shared memory which is written to by other threads.
			
__syncthreads() waits until all threads within the same block has reached the command and all threads within a warp (group of 32 threads) - that means all warps that belongs to a threadblock must reach the statement.
			
CUDA model is MIMD but current NVIDIA GPUs implement __syncthreads() at warp granularity instead of thread. It means, these are warps inside a thread-block who are synchronized not necessarily
*/
			__syncthreads();

			swappedeven=false;

			__syncthreads();

			//first column only which would have the array
			if (threadIdx.y == 0) {
			
				int idx = threadIdx.x;
				
				//0, 1, 2 threads will go through
				if( idx < (size/2) ){
					//COMPARISONS:
					// 0 <--> 1
					// 2 <--> 3
					// 4 <--> 5
					if ( in[2*idx] > in[2*idx+1] ){
						//BUBBLE SORT LOGIC
						temp= in[2*idx];
						in[2*idx]=in[2*idx+1];
						in[2*idx+1]=temp;
						swappedeven=true;
					
					}
				}
			}
			__syncthreads();
		}
		else{

			__syncthreads();

			swappedodd=false;

			__syncthreads();

			if (threadIdx.y == 0) {

				int idx = threadIdx.x;
				//0, 1 will go through
				if( idx < (size/2)-1 ){
					//COMPARISONS:
					// 1 <--> 2
					// 3 <--> 4
					if ( in[2*idx+1] > in[2*idx+2] ){

						temp=in[2*idx+1];
						in[2*idx+1]=in[2*idx+2];
						in[2*idx+2]=temp;
						swappedodd=true;

					}

				}


			}

			__syncthreads();

		}
	
	//if there are no swaps in odd phase as well as even phase then break (which means all sorting is done)
	// !(false) => true
	if( !( swappedodd || swappedeven ) )
		break;

	oddeven =! oddeven;	//switch phase of sorting

	}

	__syncthreads();

	//Store this phase's in[] array to out[] array
	int idx = threadIdx.x;

	if ( idx < size )
		out[idx] = in[idx];
		
}


int main(void)
{
	int i;
	int *a, *a_sorted;
	int *d_a, *d_sorted;
	int n = 10;		//make sure to keep this even
	int size = sizeof(int) *n;

/*
-----Why double pointer in void?-----
All CUDA API functions return an error code (or cudaSuccess if no error occured). All other parameters are passed by reference. However, in plain C you cannot have references, that's why you have to pass an address of the variable that you want the return information to be stored. Since you are returning a pointer, you need to pass a double-pointer.
-------------------------------------

-----cudaMalloc------
Allocates size bytes of linear memory on the device (GPU)
---------------------
*/

	cudaMalloc( (void**) &d_a, size);
	cudaMalloc( (void**) &d_sorted, size);

	a = (int*) malloc(size);
	a_sorted = (int*) malloc(size);

	printf("Enter the unsorted numbers:\n");
	
	for(i=0;i<n;i++){
		scanf("%d",&a[i]);
	}
	
	//d_a -> destination. a -> source.
	//Host to device array copy
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	
	//<<< >>> CUDA semantic
	testKernel<<<1,n>>>(d_a, d_sorted, n);

	//Device to Host array for final display (I/O)
	cudaMemcpy(a_sorted, d_sorted, size, cudaMemcpyDeviceToHost);
	
	for (i=0;i<n;i++){
		printf("%d\t",a_sorted[i]);
	}
	
	printf("\n");
	
	//free memory allocated by malloc and cudamalloc
	free(a);
	free(a_sorted);
	cudaFree(d_sorted);
	cudaFree(d_a);
}