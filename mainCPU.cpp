//compile with
//g++ -O2 -march=native -ftls-model=initial-exec -g -o nancy nancy.cpp -lpthread


#include <thread>
#include <atomic>
#include <pthread.h>
#include <unistd.h>

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

//Mersenne twister as of C++1x libc++
/* Period parameters */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */

thread_local unsigned long mt[N]; /* the array for the state vector  */
thread_local int mti=N+1; /* mti==N+1 means mt[N] is not initialized */

/* initializes mt[N] with a seed */
extern "C" void init_genrand(unsigned long s)
{
	mt[0]= s & 0xffffffffUL;
	for (mti=1; mti<N; mti++) {
		mt[mti] =
				(1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
		/* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
		/* In the previous versions, MSBs of the seed affect   */
		/* only MSBs of the array mt[].						*/
		/* 2002/01/09 modified by Makoto Matsumoto			 */
		mt[mti] &= 0xffffffffUL;
		/* for >32 bit machines */
	}
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
extern "C" void init_by_array(unsigned long init_key[], int key_length)
{
	int i, j, k;
	init_genrand(19650218UL);
	i=1; j=0;
	k = (N>key_length ? N : key_length);
	for (; k; k--) {
		mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525UL))
				+ init_key[j] + j; /* non linear */
		mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
		i++; j++;
		if (i>=N) { mt[0] = mt[N-1]; i=1; }
		if (j>=key_length) j=0;
	}
	for (k=N-1; k; k--) {
		mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941UL))
				- i; /* non linear */
		mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
		i++;
		if (i>=N) { mt[0] = mt[N-1]; i=1; }
	}

	mt[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */
}

/* generates a random number on [0,0xffffffff]-interval */
extern "C" unsigned long genrand_int32(void)
{
	unsigned long y;
	static unsigned long mag01[2]={0x0UL, MATRIX_A};
	/* mag01[x] = x * MATRIX_A  for x=0,1 */

	if (mti >= N) { /* generate N words at one time */
		int kk;

		//we initialize!
		//	if (mti == N+1)   /* if init_genrand() has not been called, */
		//		init_genrand(5489UL); /* a default initial seed is used */

		for (kk=0;kk<N-M;kk++) {
			y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
			mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}
		for (;kk<N-1;kk++) {
			y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
			mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}
		y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
		mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

		mti = 0;
	}

	y = mt[mti++];

	/* Tempering */
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680UL;
	y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);

	return y;
}

/* generates a random number on [0,0x7fffffff]-interval */
extern "C" long genrand_int31(void)
{
	return (long)(genrand_int32()>>1);
}

/* generates a random number on [0,1]-real-interval */
extern "C" double genrand_real1(void)
{
	return genrand_int32()*(1.0/4294967295.0);
	/* divided by 2^32-1 */
}

/* generates a random number on [0,1)-real-interval */
extern "C" double genrand_real2(void)
{
	return genrand_int32()*(1.0/4294967296.0);
	/* divided by 2^32 */
}



thread_local uint32_t state[5];


uint32_t xorwow() {
	/* Algorithm "xorwow" from p. 5 of Marsaglia, "Xorshift RNGs" */
	uint32_t s, t = state[3];
	t ^= t >> 2;
	t ^= t << 1;
	state[3] = state[2]; state[2] = state[1]; state[1] = s = state[0];
	t ^= s;
	t ^= s << 4;
	state[0] = t;
	return t + (state[4] += 362437);
}




/* generates a random number on (0,1)-real-interval */
extern "C" double genrand_real3(void) {
	return (((double) xorwow()) / 4294967296.0);
	/* divided by 2^32 */
}


//https://en.wikipedia.org/wiki/Xoroshiro128%2B looks not the best for this task...
thread_local uint64_t s[2]{1024789654,558121992};

uint64_t next(void) {
	uint64_t s1 = s[0];
	const uint64_t s0 = s[1];
	const uint64_t result = s0 + s1;
	s[0] = s0;
	s1 ^= s1 << 23; // a
	s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5); // b, c
	return result;
}

//the amount of write activity is REALLY reduced so is ok to go for atomic

std::atomic<uint32_t>tempNuclidesCount{0};
std::atomic<uint32_t>tempRadonNuclidesCount{0};

std::atomic<uint32_t>threadReady{0};

//mutex issues memory barrier so is ok to have a NON atomic
int conditionMet = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
pthread_barrier_t brTermination;
pthread_barrier_t brMiddle;
pthread_barrier_t brReady;
bool end = false;

float p1;
float p2;

uint32_t threadCount = 1; //std::thread::hardware_concurrency();


void split(){
	state[0]= 762;
	state[1]= 556;
	state[1]= 5560000;

	timespec time;
	clock_gettime(CLOCK_MONOTONIC, &time);
	init_genrand(time.tv_nsec + time.tv_sec);
	while(true){
		//notify we are ready
		pthread_barrier_wait(&brReady);

		//avoid spurious wakeup
		while (!conditionMet) {
			//waiting RELEASE the mutex, on signaling thread are awake with the mutex lock
			//which is useless and just wastes resources... because you have to serialize in any case by yourself
			pthread_mutex_lock(&mutex);
			threadReady++;
			pthread_cond_wait(&condition,&mutex);
			//acquiring will lock the mutex
			pthread_mutex_unlock(&mutex);
			threadReady--;
		}

		if(end){
			return;
		}

		//2^64
		const double max = 18446744073709551615.0;
		//the tempNuclidesCount value is modified during the execution!
		uint32_t upToNuclides = tempNuclidesCount / threadCount;
		for(uint32_t j = 0;
		    j < upToNuclides;
		    j++){
			//In MT code RAND slows down too much because contain MUTEX internally!!!
			//double r1 = next();
			//r1 = r1 / max;
			if(genrand_real3() < p1){// compare r with probability density function, here p = const.: is there a decay?
				tempNuclidesCount--;
				tempRadonNuclidesCount++;
			}
		}

		//wait all thread to arrived to this point
		pthread_barrier_wait(&brMiddle);

		uint32_t upToRadon = tempRadonNuclidesCount / threadCount;
		for(uint32_t j = 0;
		    j < upToRadon;
		    j++){
			//			float r1 = next();
			//			r1 = r1 / max;
			if(genrand_real3() < p2){ //compare r with probability density function, here p = const.: is there a decay?
				tempRadonNuclidesCount--;
			}
		}

		pthread_barrier_wait(&brTermination);
	}
}



int main(){
	pthread_barrier_init(&brTermination, NULL, threadCount + 1);
	pthread_barrier_init(&brMiddle, NULL, threadCount);
	pthread_barrier_init(&brReady, NULL, threadCount + 1);

	timespec startTime,endTime,elapsedTime1,elapsedTime2,oldFraction;
	clock_gettime(CLOCK_MONOTONIC, &startTime);


	//Total number of nuclides
	//368997473
	uint32_t nuclidesCount = 368997473;

	//starting number of nuclides of radon
	uint32_t radonNuclidesCount = 0;

	//expressed in DAYS
	float timeStep = 0.1f;

	for(uint32_t i = 0 ; i < threadCount; i++){
		std::thread* t = new std::thread(split);
		t->detach();
	}

	//GPU really love FLOAT (and you do not need Double)
	float lambda1 = 1.187e-6f; //decay constant radium over days
	float lambda2 = 0.182f; //decay constant radon over days

	p1 = lambda1*timeStep; //probability per time step of radium decay
	p2 = lambda2*timeStep; //probability per time step of radon decay

	//easier to pass as an ARRAY
	size_t size = 10000;

	uint32_t* n_monteCarloRadium = (uint32_t*) calloc(size, sizeof (uint32_t));
	uint32_t* n_monteCarloRadon = (uint32_t*) calloc(size, sizeof (uint32_t));

	n_monteCarloRadium[0] = nuclidesCount;
	n_monteCarloRadon[0] = radonNuclidesCount;

	tempNuclidesCount = n_monteCarloRadium[0];
	tempRadonNuclidesCount = n_monteCarloRadon[0] ;



	oldFraction = startTime;
	for (volatile uint32_t i=1; i < 27; i++){
		//avoid spurious wake up
		conditionMet = 0;

		//wait all thread are ready
		pthread_barrier_wait(&brReady);

		//spinlock until the very last operation are done
		while(threadReady != threadCount){
			std::this_thread::yield();
		}
		conditionMet = 1;

		//this mutex will not be ready until the last thread has really entered the wait
		pthread_mutex_lock(&mutex);
		pthread_cond_broadcast(&condition);
		pthread_mutex_unlock(&mutex);

		pthread_barrier_wait(&brTermination);
		n_monteCarloRadium[i] = tempNuclidesCount;
		n_monteCarloRadon[i] = tempRadonNuclidesCount;

		clock_gettime(CLOCK_MONOTONIC, &endTime);
		timespec_diff(&oldFraction,&endTime,&elapsedTime1);
		timespec_diff(&startTime,&endTime,&elapsedTime2);


		double v0 = elapsedTime1.tv_nsec;
		double v1 = elapsedTime2.tv_nsec;

		printf("Cicle %i elapsedTotal : %li . %.2e, this run %li . %.2e \t",i, elapsedTime2.tv_sec,v1,elapsedTime1.tv_sec,v0);
		auto d1 = nuclidesCount - (tempNuclidesCount.load() + tempRadonNuclidesCount.load());
		printf("Nuclide status: \t%i vs \t%i \t decayedRadon: %i\n",tempNuclidesCount.load(), tempRadonNuclidesCount.load(),d1);
		oldFraction = endTime;
	}

	clock_gettime(CLOCK_MONOTONIC, &endTime);
	timespec_diff(&startTime,&endTime,&elapsedTime1);

	double v0 = elapsedTime1.tv_nsec;
	printf("elapsed : %li . %.2e\n",elapsedTime1.tv_sec,v0);

	//signal the thread to exit, avoid too much fluffering, if we crash here no one really cares anymore
	end = true;
	pthread_cond_broadcast(&condition);

	//wait a moment just for grace
	sleep(1);
}


















