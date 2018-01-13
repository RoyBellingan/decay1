//kernel void hello(global ulong *val) {
//		size_t i = get_global_id(0);
//		for (ulong j = 0; j < 100000000; j++) {
//				val[i] += j;
//		}
//}



//https://stackoverflow.com/a/16130111
float rand(uint *state, int g1){
/*/
	uint x = state[0];
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	state[0] = x;
	float t1 = x;
	static float max = 4294967295;
	//return t1;
	return t1 / max;
/*/
	// Algorithm "xorwow" from p. 5 of Marsaglia, "Xorshift RNGs"
	uint s, t = state[3];
	t ^= t >> 2;
	t ^= t << 1;
	state[3] = state[2];
	state[2] = state[1];
	state[1] = s = state[0];
	t ^= s;
	t ^= s << 4;
	state[0] = t;
	t + (state[4] += 362437);

	float t1 = t;
	static float max = 4294967295;
	//return t1;
	return t1 / max;
//*/
}

//The event are rare, so we just go for atomic instead of having a local buffer and performing operation at the end
//https://groups.google.com/forum/#!topic/boost-compute/xJS05dkQEJk
//https://stackoverflow.com/q/11364567
__kernel void add_numbers(__global uint* iteration,__global uint* nuclide , __global float2* prob, __global uint* seed ) {


float4 input1, input2, sum_vector;
uint global_addr, local_addr;

local_addr = get_local_id(0);
int g1 = get_global_id(0);
//global_addr = get_global_id(0) * 2;


//printf("%i : %i \n", g1,local_addr);
ulong max = iteration[0];
if(g1 == 0){//first kernel run the remnant
	max = iteration[1];
}



//Each GPU Thread will need a local state
uint state[5];
state[0] = g1 + (local_addr + 1 + ( (local_addr * 0xFFCC) << 4)) + ((*seed << 23) * *seed << 3);
state[1] = state[0] ^ ((*seed << 23) | (*seed << 3));
state[2] = g1 + g1 ^ (g1 | 5567898) + 0xFF4 + iteration[1];
state[3] =  5567898 + 0xFF4 * g1 + iteration[0];
state[4] =  5567898 + 0xFF4 * g1 + local_addr;




//printf("%u \n", state[0]);
if(max > 1000000){
	printf("Too many iteration !\n");
	return;
}
float pro = prob->x;
float sum = 0;
for(ulong j = 0;
    j < max;
    j++){
	float v = rand(&state,g1);
	//printf("%e\n", v);
	if(v  < pro){// compare r with probability density function, here p = const.: is there a decay?
		atom_dec(nuclide);
		atom_inc(nuclide + 1);
	}
	//sum += v;
}
//printf("sum %e; avg %e for %u\n", sum, sum/max,g1);

//we fence to avoid second part to start while first still in progess (so we do not decade the right amount of radon)
//barrier(CLK_LOCAL_MEM_FENCE);
ulong j = 0;
//printf("%u \n", iteration->y);
//if less than 4Bilion to avoid wrapping


max = iteration[2];
if(g1 == 0){//first kernel run the remnant
	max = iteration[3];
}

while(j < max) {
	if(rand(&state,g1)  < prob->y){ //compare r with probability density function, here p = const.: is there a decay?
		atom_dec(nuclide + 1);
	}
	j++;
}

}
