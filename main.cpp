#define PROGRAM_FILE "sum.cl"

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

/**
 * @brief timespec_diff is just a quick function to subtract two timespec
 */
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


/* Pick a device: the first GPU on any platform, otherwise the first device of
 * any type (e.g. pocl on the CPU). Platform order is not stable, so search.
 * Set CL_FORCE_CPU=1 to skip GPUs (useful to cross-check results on pocl). */
cl_device_id create_device() {
	cl_uint platformCount = 0;
	if (clGetPlatformIDs(0, NULL, &platformCount) != CL_SUCCESS || platformCount == 0) {
		fprintf(stderr, "No OpenCL platforms found (is an ICD installed?)\n");
		exit(1);
	}
	cl_platform_id platforms[16];
	if (platformCount > 16) platformCount = 16;
	clGetPlatformIDs(platformCount, platforms, NULL);

	const char *force = getenv("CL_FORCE_CPU");
	bool skipGpu = force && *force && strcmp(force, "0") != 0;

	const cl_device_type wanted[2] = {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL};
	for (int pass = skipGpu ? 1 : 0; pass < 2; ++pass) {
		for (cl_uint p = 0; p < platformCount; ++p) {
			cl_device_id dev;
			cl_uint n = 0;
			if (clGetDeviceIDs(platforms[p], wanted[pass], 1, &dev, &n) != CL_SUCCESS || n == 0)
				continue;
			cl_device_type type;
			clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
			if (skipGpu && (type & CL_DEVICE_TYPE_GPU))
				continue;

			char platName[256] = "", devName[256] = "";
			clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(platName), platName, NULL);
			clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(devName), devName, NULL);
			printf("Platform: %s\nDevice: %s\n", platName, devName);
			return dev;
		}
	}

	fprintf(stderr, "No usable OpenCL device found%s\n",
		skipGpu ? "" : " (for the AMD GPU, set RUSTICL_ENABLE=radeonsi)");
	exit(1);
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename, const char* options) {

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
	err = clBuildProgram(program, 0, NULL, options, NULL, NULL);
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

static void check(cl_int err, const char *what) {
	if (err != CL_SUCCESS) {
		fprintf(stderr, "%s failed: OpenCL error %d\n", what, err);
		exit(1);
	}
}

static double ms_between(const struct timespec &a, const struct timespec &b) {
	return (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

// usage: ./a.out [steps] [groups_per_compute_unit] [local_size]
int main(int argc, char **argv) {
	int steps = argc > 1 ? atoi(argv[1]) : 10000;
	size_t groupsPerCU = argc > 2 ? strtoul(argv[2], NULL, 10) : 32;
	size_t localSize = argc > 3 ? strtoul(argv[3], NULL, 10) : 0; // 0 = pick from the device

	cl_int err;
	cl_device_id device = create_device();
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	check(err, "clCreateContext");
	/* random streams per work-item, vectorised in the kernel (see sum.cl) */
	const char *lanesEnv = getenv("DECAY_LANES");
	int lanes = lanesEnv ? atoi(lanesEnv) : 8;
	/* DECAY_SEED: base seed; runs with different values use independent random numbers */
	const char *seedEnv = getenv("DECAY_SEED");
	cl_uint baseSeed = seedEnv ? (cl_uint)strtoul(seedEnv, NULL, 0) : 0;
	char buildOptions[64];
	snprintf(buildOptions, sizeof(buildOptions), "-DLANES=%d", lanes);
	cl_program program = build_program(context, device, PROGRAM_FILE, buildOptions);
	cl_kernel kRadium = clCreateKernel(program, "radium_decay", &err);
	check(err, "clCreateKernel(radium_decay)");
	cl_kernel kRadon = clCreateKernel(program, "radon_decay", &err);
	check(err, "clCreateKernel(radon_decay)");
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
	check(err, "clCreateCommandQueueWithProperties");

	/* Size the launch for this device instead of the old fixed 24 * 192 */
	cl_uint computeUnits = 0;
	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, NULL);
	size_t kernelMax = 256, maxRadon = 256, multiple = 1;
	clGetKernelWorkGroupInfo(kRadium, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernelMax), &kernelMax, NULL);
	clGetKernelWorkGroupInfo(kRadon, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maxRadon), &maxRadon, NULL);
	clGetKernelWorkGroupInfo(kRadium, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
				 sizeof(multiple), &multiple, NULL);
	if (maxRadon < kernelMax)
		kernelMax = maxRadon;
	if (localSize == 0) {
		localSize = kernelMax < 256 ? kernelMax : 256;
		if (multiple > 1 && localSize > multiple)
			localSize -= localSize % multiple; // whole wavefronts / warps
	}
	size_t globalSize = computeUnits * groupsPerCU * localSize;
	printf("%u compute units, SIMD width %zu -> global %zu, local %zu, %d lanes, %d steps\n",
	       computeUnits, multiple, globalSize, localSize, lanes, steps);

	/* nuclide[0] Radium, [1] Radon, [2] [3] snapshots (see sum.cl), rest scratch */
	uint32_t nuclides[8]{368997473, 0, 0, 0, 0, 0, 0, 0};
	cl_mem nuclideBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
					      sizeof(nuclides), nuclides, &err);
	check(err, "clCreateBuffer");

	/* Decay probability per step as a threshold on a 32-bit random integer: p * 2^32 */
	const float prob[2]{1.187e-7f * 10, 0.0182f * 10};
	cl_uint thrRadium = (cl_uint)llround((double)prob[0] * 4294967296.0);
	cl_uint thrRadon = (cl_uint)llround((double)prob[1] * 4294967296.0);

	err = clSetKernelArg(kRadium, 0, sizeof(cl_mem), &nuclideBuffer);
	err |= clSetKernelArg(kRadium, 1, sizeof(cl_uint), &thrRadium);
	err |= clSetKernelArg(kRadon, 0, sizeof(cl_mem), &nuclideBuffer);
	err |= clSetKernelArg(kRadon, 1, sizeof(cl_uint), &thrRadon);
	check(err, "clSetKernelArg");

	/* ---- Reference model ----
	 * The exact per-step probabilities are the thresholds the kernel compares against. */
	const double N0 = nuclides[0];
	const double p1 = thrRadium / 4294967296.0, p2 = thrRadon / 4294967296.0;
	/* Expected trajectory of this step model ("discrete Bateman"):
	 *   Ra' = Ra (1 - p1)          Rn' = (Rn + p1 Ra) (1 - p2)
	 * (radon created in a step can already decay in that same step, as in the kernels).
	 * Each atom evolves independently, so the count in a state is Binomial(N0, E/N0). */
	double expRa = N0, expRn = 0;
	double maxAbsZRn = 0;
	int maxZStep = 0;
	/* Per-step decay counts, given the population going in, are Binomial(n, p):
	 * these increments are independent, so their chi-square is a clean test of the
	 * RNG + kernels (should come out at 1 per degree of freedom). */
	double chi2Ra = 0, chi2Rn = 0;
	int dofRa = 0, dofRn = 0;

	printf("step, Radium, delta, Radon, delta, Rn expected, z, elapsed ms\n");
	struct timespec tStart, t1, t2;
	clock_gettime(CLOCK_MONOTONIC, &tStart);

	for (int zz = 0; zz < steps; zz++) {
		clock_gettime(CLOCK_MONOTONIC, &t1);
		uint32_t oldRa = nuclides[0], oldRn = nuclides[1];
		cl_uint seed = (cl_uint)zz + baseSeed * 0x9E3779B1u; // golden-ratio stride between bases

		/* Radium: snapshot [0] -> [2], then every work-item tests its share of it */
		check(clEnqueueCopyBuffer(queue, nuclideBuffer, nuclideBuffer, 0, 2 * sizeof(uint32_t),
					  sizeof(uint32_t), 0, NULL, NULL), "copy Ra snapshot");
		check(clSetKernelArg(kRadium, 2, sizeof(cl_uint), &seed), "clSetKernelArg(seed)");
		check(clEnqueueNDRangeKernel(queue, kRadium, 1, NULL, &globalSize, &localSize, 0, NULL, NULL),
		      "enqueue radium_decay");

		/* Radon: snapshot [1] -> [3] after radium finished (in-order queue) */
		check(clEnqueueCopyBuffer(queue, nuclideBuffer, nuclideBuffer, sizeof(uint32_t), 3 * sizeof(uint32_t),
					  sizeof(uint32_t), 0, NULL, NULL), "copy Rn snapshot");
		check(clSetKernelArg(kRadon, 2, sizeof(cl_uint), &seed), "clSetKernelArg(seed)");
		check(clEnqueueNDRangeKernel(queue, kRadon, 1, NULL, &globalSize, &localSize, 0, NULL, NULL),
		      "enqueue radon_decay");

		check(clEnqueueReadBuffer(queue, nuclideBuffer, CL_TRUE, 0, 2 * sizeof(uint32_t), nuclides,
					  0, NULL, NULL), "read result");
		clock_gettime(CLOCK_MONOTONIC, &t2);

		uint32_t decRa = oldRa - nuclides[0];
		int decRn = (int)nuclides[1] - (int)oldRn - (int)decRa; // = -(radon decays)

		/* decay-count chi-square */
		double m = oldRa * p1, v = m * (1 - p1);
		if (v > 0) { chi2Ra += (decRa - m) * (decRa - m) / v; dofRa++; }
		double nRn = (double)oldRn + decRa; // radon atoms tested this step
		m = nRn * p2; v = m * (1 - p2);
		if (v > 0) { chi2Rn += (-decRn - m) * (-decRn - m) / v; dofRn++; }

		/* trajectory vs expectation */
		expRn = (expRn + p1 * expRa) * (1 - p2);
		expRa *= 1 - p1;
		double q = expRn / N0;
		double zRn = (nuclides[1] - expRn) / sqrt(N0 * q * (1 - q));
		if (fabs(zRn) > maxAbsZRn) { maxAbsZRn = fabs(zRn); maxZStep = zz; }

		printf("%d, %u,-%u,%u,%i, %.1f, %+.2f, %.2f\n", zz, nuclides[0], decRa, nuclides[1], decRn,
		       expRn, zRn, ms_between(t1, t2));
	}

	double total = ms_between(tStart, t2);
	printf("total %.1f ms, %.2f ms/step\n", total, total / steps);

	/* ---- Comparison summary ---- */
	double qRa = expRa / N0, qRn = expRn / N0;
	double zRa = (nuclides[0] - expRa) / sqrt(N0 * qRa * (1 - qRa));
	double zRn = (nuclides[1] - expRn) / sqrt(N0 * qRn * (1 - qRn));
	printf("\n--- check against the decay model (p_Ra = %.6e, p_Rn = %.6e per step) ---\n", p1, p2);

	double tolRa = 3 * sqrt(2.0 / dofRa), tolRn = 3 * sqrt(2.0 / dofRn);
	double rRa = chi2Ra / dofRa, rRn = chi2Rn / dofRn;
	printf("decay counts per step, chi2/dof (should be 1 +- %.3f at 3 sigma):\n", tolRa);
	printf("  Radium %.3f  %s\n", rRa, fabs(rRa - 1) <= tolRa ? "ok" : "OUT OF RANGE");
	printf("  Radon  %.3f  %s\n", rRn, fabs(rRn - 1) <= tolRn ? "ok" : "OUT OF RANGE");

	printf("trajectory after %d steps (z = deviation in standard deviations):\n", steps);
	printf("  Radium sim %u  expected %.1f  z %+.2f\n", nuclides[0], expRa, zRa);
	printf("  Radon  sim %u  expected %.1f  z %+.2f\n", nuclides[1], expRn, zRn);
	printf("  largest |z| of Radon over the run: %.2f at step %d (below ~4 is normal for a long run)\n",
	       maxAbsZRn, maxZStep);

	/* Continuous Bateman, lambda = -ln(1 - p): what the physics would give with dt -> 0 */
	double l1 = -log1p(-p1), l2 = -log1p(-p2), t = steps;
	double batRa = N0 * exp(-l1 * t);
	double batRn = N0 * l1 / (l2 - l1) * (exp(-l1 * t) - exp(-l2 * t));
	printf("continuous Bateman at t = %d: Radium %.1f, Radon %.1f\n", steps, batRa, batRn);
	printf("  the step model's Radon is %.1f%% %s: that is the time-step discretisation, not noise\n",
	       fabs(expRn / batRn - 1) * 100, expRn < batRn ? "lower" : "higher");

	clReleaseMemObject(nuclideBuffer);
	clReleaseKernel(kRadium);
	clReleaseKernel(kRadon);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
	return 0;
}
