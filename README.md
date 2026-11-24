Compile with
 g++ -O2 -DCL_TARGET_OPENCL_VERSION=300 main.cpp -lOpenCL

Run from this directory (the kernel is loaded from sum.cl):
 ./a.out [steps] [groups_per_compute_unit] [local_size]
 defaults: 10000 steps, 32 groups per compute unit, local size picked from the device

Environment:
 RUSTICL_ENABLE=radeonsi  needed for Mesa Rusticl to expose an AMD GPU
 CL_FORCE_CPU=1           skip GPUs, e.g. to cross-check on pocl
 DECAY_LANES=N            random streams per work-item, 1/2/4/8/16 (default 8; lets pocl use AVX2)
 DECAY_SEED=N             base seed (default 0); different values give independent runs

At the end the run is checked against the decay model: chi2 of the per-step decay
counts (should be ~1), the trajectory against the expected (discrete Bateman) curve,
and the continuous Bateman solution for reference.

There is another parked file, to test a C++ multi threaded version
