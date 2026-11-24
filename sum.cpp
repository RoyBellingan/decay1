// Ra-226 -> Rn-222 decay, Monte Carlo: every atom is tested once per time step.
//
// nuclide[0] = Radium, nuclide[1] = Radon        live counters, changed with atomics
// nuclide[2], nuclide[3] = snapshots of [0], [1] copied by the host queue
//                          (clEnqueueCopyBuffer) right before each kernel,
//                          so every work-item splits the same population.
//
// One time step = radium_decay, then radon_decay. The queue is in-order, so the
// second kernel starts only after the first has finished.
//
// LANES (build option -DLANES=N, N = 1, 2, 4, 8, 16): every work-item runs N
// independent random streams side by side in a uintN vector. The random number
// generator is a recurrence, so one stream can't be vectorised; N streams can:
// on a CPU (pocl) each operation becomes one AVX2 instruction for 8 lanes.

#ifndef LANES
#define LANES 8
#endif

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)

#if LANES == 1
typedef uint vuint;
#define LANE_IDS ((vuint)(0))
// scalar comparisons give 1 for true
#define COUNT_IF(cnt, cond) ((cnt) += (uint)(cond))
#else
typedef CAT(uint, LANES) vuint;
#if LANES == 2
#define LANE_IDS ((vuint)(0, 1))
#elif LANES == 4
#define LANE_IDS ((vuint)(0, 1, 2, 3))
#elif LANES == 8
#define LANE_IDS ((vuint)(0, 1, 2, 3, 4, 5, 6, 7))
#elif LANES == 16
#define LANE_IDS ((vuint)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
#else
#error "LANES must be 1, 2, 4, 8 or 16"
#endif
// vector comparisons give -1 (all bits set) for true: subtracting it adds 1
#define COUNT_IF(cnt, cond) ((cnt) -= CAT(as_uint, LANES)(cond))
#endif

// Algorithm "xorwow" from p. 5 of Marsaglia, "Xorshift RNGs" (without the Weyl
// counter, as before), one step in every lane.
inline vuint xorwow(vuint *state) {
	vuint s, t = state[3];
	t ^= t >> 2;
	t ^= t << 1;
	state[3] = state[2];
	state[2] = state[1];
	state[1] = s = state[0];
	t ^= s;
	t ^= s << 4;
	state[0] = t;
	return t;
}

// lowbias32 integer hash (Chris Wellons): decorrelates neighbouring streams.
inline vuint hash32(vuint x) {
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	x ^= x >> 16;
	return x;
}

// Each lane of each work-item gets its own stream, keyed by (stream id, step, kernel).
// Hash the stream id BEFORE combining it with the step: with plain
// `stream ^ hash(step)`, two steps whose hashes share their high bits only
// permute the same set of seeds, and replay the same random numbers.
inline void seed_state(vuint *state, uint seed, uint salt) {
	vuint stream = (vuint)((uint)get_global_id(0) * LANES) + LANE_IDS;
	vuint h = hash32(hash32(stream) ^ hash32((vuint)(seed ^ salt)));
	for (int i = 0; i < 4; i++) {
		h = hash32(h + 0x9e3779b9U);
		state[i] = h;
	}
	state[0] |= 1; // xorwow state must not be all zero
}

// Exact split of `total` atoms over the work-items: the first total % n get one extra.
inline uint my_share(uint total) {
	uint n = get_global_size(0), g = get_global_id(0);
	return total / n + (g < total % n ? 1u : 0u);
}

// Count decays among `atoms`, with probability threshold / 2^32 each:
// comparing the raw integer against a precomputed threshold needs no float
// conversion and no division. The atoms are dealt over the lanes; the last
// partial round is masked to the first `atoms % LANES` lanes.
inline uint count_decays(uint atoms, uint threshold, vuint *state) {
	vuint cnt = (vuint)(0);
	uint rounds = atoms / LANES, rest = atoms % LANES;
	for (uint j = 0; j < rounds; j++)
		COUNT_IF(cnt, xorwow(state) < threshold);
	if (rest)
		COUNT_IF(cnt, (xorwow(state) < threshold) & (LANE_IDS < rest));

	// Horizontal sum by halving with swizzles. Never take the vector's address
	// (e.g. ((uint *)&cnt)[i]): on a GPU that moves it to slow scratch memory.
#if LANES == 1
	return cnt;
#else
#if LANES == 16
	uint8 c8 = cnt.lo + cnt.hi;
#elif LANES == 8
	uint8 c8 = cnt;
#endif
#if LANES >= 8
	uint4 c4 = c8.lo + c8.hi;
#elif LANES == 4
	uint4 c4 = cnt;
#endif
#if LANES >= 4
	uint2 c2 = c4.lo + c4.hi;
#else
	uint2 c2 = cnt;
#endif
	return c2.x + c2.y;
#endif
}

// Sum of `v` over the work-group, via one local atomic per work-item.
// Result is meaningful in work-item 0.
inline uint group_sum(uint v, __local uint *acc) {
	if (get_local_id(0) == 0)
		*acc = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (v)
		atomic_add(acc, v);
	barrier(CLK_LOCAL_MEM_FENCE);
	return *acc;
}

__kernel void radium_decay(__global uint *nuclide, const uint threshold, const uint seed) {
	__local uint acc;
	vuint state[4];
	seed_state(state, seed, 0x52614a5U);

	uint d = count_decays(my_share(nuclide[2]), threshold, state);

	// one pair of global atomics per work-group instead of two per decay
	uint sum = group_sum(d, &acc);
	if (get_local_id(0) == 0 && sum) {
		atomic_sub(&nuclide[0], sum);
		atomic_add(&nuclide[1], sum);
	}
}

__kernel void radon_decay(__global uint *nuclide, const uint threshold, const uint seed) {
	__local uint acc;
	vuint state[4];
	seed_state(state, seed, 0x52616eU);

	uint d = count_decays(my_share(nuclide[3]), threshold, state);

	uint sum = group_sum(d, &acc);
	if (get_local_id(0) == 0 && sum)
		atomic_sub(&nuclide[1], sum);
}
