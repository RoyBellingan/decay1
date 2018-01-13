#define PROGRAM_FILE "sum.cl"
#define KERNEL_FUNC "add_numbers"
#define ARRAY_SIZE 20480

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


void timespec_diff(struct timespec *start, struct timespec *stop,
		   struct timespec *result) {
	if ((stop->tv_nsec - start->tv_nsec) < 0) {
		result->tv_sec = stop->tv_sec - start->tv_sec - 1;
		result->tv_nsec = stop->tv_nsec - start->tv_nsec + 1000000000;
	} else {
		result->tv_sec = stop->tv_sec - start->tv_sec;
		result->tv_nsec = stop->tv_nsec - start->tv_nsec;
	}

	return;
}


/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

	cl_platform_id platform;
	cl_device_id dev;
	int err;

	/* Identify a platform */
	err = clGetPlatformIDs(1, &platform, NULL);
	if(err < 0) {
		perror("Couldn't identify a platform");
		exit(1);
	}

	/* Access a device */
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	if(err == CL_DEVICE_NOT_FOUND) {
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	}
	if(err < 0) {
		perror("Couldn't access any devices");
		exit(1);
	}

	return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

	cl_program program;
	FILE *program_handle;
	char *program_buffer, *program_log;
	size_t program_size, log_size;
	int err;

	/* Read program file and place content into buffer */
	program_handle = fopen(filename, "r");
	if(program_handle == NULL) {
		perror("Couldn't find the program file");
		exit(1);
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	/* Create program from file */
	program = clCreateProgramWithSource(ctx, 1,
					    (const char**)&program_buffer, &program_size, &err);
	if(err < 0) {
		perror("Couldn't create the program");
		exit(1);
	}
	free(program_buffer);

	/* Build program */
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
				      0, NULL, &log_size);
		program_log = (char*) malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
				      log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		exit(1);
	}

	return program;
}

int main() {
	struct timespec res1,res2,res3;
	//clock_getres(CLOCK_PROCESS_CPUTIME_ID, &res);

	/* OpenCL structures */
	cl_device_id device;
	cl_context context;
	cl_program program;
	cl_kernel kernel;
	cl_command_queue queue;
	cl_int err;
	size_t local_size, global_size;

	/* Data and buffers */
	global_size = (16 * 24) * 6;
	local_size = 16;

	//cl_int num_groups = global_size/local_size;
	cl_mem iterBuffer,nuclideBuffer, probBuffer, seedBuffer;


	/* Create device and context */
	device = create_device();
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if(err < 0) {
		perror("Couldn't create a context");
		exit(1);
	}

	//uint32_t *pbuf = (uint32_t *)aligned_alloc(4096, 64);
	//first is integer division, second is modulo
	uint32_t iteration[4]{368997473 / (global_size - 1),368997473 % global_size,0,0};
	uint32_t nuclides[2]{368997473,0};
	uint32_t seed = 0;

	float prob[2]{1.187e-7f,0.0182f};
	/* Build program */
	program = build_program(context, device, PROGRAM_FILE);

	/* Create data buffer */
	iterBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
				      CL_MEM_USE_HOST_PTR, 4 * sizeof(uint32_t), iteration, &err);
	nuclideBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
				      CL_MEM_COPY_HOST_PTR, 2 * sizeof(uint32_t), nuclides, &err);
	probBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
				    CL_MEM_COPY_HOST_PTR, 2 * sizeof(float), prob, &err);
	seedBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
					      CL_MEM_COPY_HOST_PTR, sizeof(uint32_t), &seed, &err);


	if(err < 0) {
		perror("Couldn't create a buffer");
		exit(1);
	};

	/* Create a command queue */
	queue = clCreateCommandQueue(context, device, 0, &err);
	if(err < 0) {
		perror("Couldn't create a command queue");
		exit(1);
	};

	/* Create a kernel */
	kernel = clCreateKernel(program, KERNEL_FUNC, &err);
	if(err < 0) {
		perror("Couldn't create a kernel");
		exit(1);
	};

	/* Create kernel arguments */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &iterBuffer);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &nuclideBuffer);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &probBuffer);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &seedBuffer);

	if(err < 0) {
		perror("Couldn't create a kernel argument");
		exit(1);
	}
	printf("Radium, delta, Radon, delta, elapsed \n");


	for(int zz = 0; zz < 100; zz++){
		/* Enqueue kernel */

		clock_gettime(CLOCK_MONOTONIC, &res1);

		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
					     &local_size, 0, NULL, NULL);

		if(err < 0) {
			perror("Couldn't enqueue the kernel");
			exit(1);
		}

		//CL_INVALID_COMMAND_QUEUE
		/* Read the kernel's output */

		auto old1 = nuclides[0];
		int old2 = nuclides[1];

		err = clEnqueueReadBuffer(queue, nuclideBuffer, CL_TRUE, 0,
					  2 * sizeof(uint32_t) , nuclides, 0, NULL, NULL);

		//yes I should use signed...
		if(nuclides[1] > 0xFF000000){
			nuclides[1] = 0;
		}
		if(err < 0) {
			perror("Couldn't read the buffer");
			exit(1);
		}
		clock_gettime(CLOCK_MONOTONIC, &res2);
		timespec_diff(&res1,&res2,&res3);

		double v0 = res3.tv_nsec;
		auto delta1 = old1 - nuclides[0];
		int delta2 = ((int)nuclides[1] - old2) - delta1;

		printf("%u,-%u,%u,%i,%li . %.2e\n",nuclides[0], delta1, nuclides[1],delta2,res3.tv_sec,v0);

		//uint32_t iteration[4]{368997473 / global_size,368997473 % global_size,0,0};

		iteration[0] = (nuclides[0] / (global_size - 1));
		iteration[1] = (nuclides[0] % global_size);
		iteration[2] = (nuclides[1] / (global_size - 1));
		iteration[3] = (nuclides[1] % global_size);


		err = clEnqueueWriteBuffer(queue, iterBuffer, CL_TRUE, 0, 4* sizeof(uint32_t) , iteration, 0, NULL, NULL);
		err = clEnqueueWriteBuffer(queue, seedBuffer, CL_TRUE, 0,sizeof(uint32_t) , &zz, 0, NULL, NULL);


		//iterBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 2 * sizeof(uint32_t), iteration, &err);

		//err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &iterBuffer);

	}


//	/* Check result */
//	total = 0.0f;
//	for(j=0; j<num_groups; j++) {
//		total += sum[j];
//	}
//	actual_sum = 1.0f * ARRAY_SIZE/2*(ARRAY_SIZE-1);
//	printf("Computed sum = %.1f\n", total);
//	if(fabs(total - actual_sum) > 0.01*fabs(actual_sum))
//		printf("Check failed.\n");
//	else
//		printf("Check passed.\n");

	/* Deallocate resources */
	clReleaseKernel(kernel);
	clReleaseMemObject(probBuffer);
	clReleaseMemObject(iterBuffer);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
	return 0;
}
