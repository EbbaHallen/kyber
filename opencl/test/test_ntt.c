#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../kem.h"
#include "../params.h"
#include "../indcpa.h"
#include "../polyvec.h"
#include "../poly.h"
#include "../randombytes.h"
#include "cpucycles.h"
#include "speed_print.h"
#include "../opencl.h"
#include "../ntt.h"

#define NTESTS 1000

uint64_t t[NTESTS];
double t_time[NTESTS];
uint8_t seed[KYBER_SYMBYTES] = {0};

// Function to get current time in milliseconds
double now() {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec + time.tv_nsec * 1e-6;
}



int main(void)
{
  unsigned int i;
  uint8_t pk[CRYPTO_PUBLICKEYBYTES];
  uint8_t sk[CRYPTO_SECRETKEYBYTES];
  uint8_t ct[CRYPTO_CIPHERTEXTBYTES];
  uint8_t key[CRYPTO_BYTES];
  uint8_t coins32[KYBER_SYMBYTES];
  uint8_t coins64[2*KYBER_SYMBYTES];
  polyvec matrix[KYBER_K];
  poly ap;

  opencl_init();

  randombytes(coins32, KYBER_SYMBYTES);
  randombytes(coins64, 2*KYBER_SYMBYTES);

  gen_matrix(matrix, seed, 0);
  poly_getnoise_eta1(&ap, seed, 0);
  poly_getnoise_eta2(&ap, seed, 0);
  
  
   // NTT GPU
 
  double start = now();
  for(i=0;i<NTESTS;i++) {
    clock_t startTime = (double)clock()/CLOCKS_PER_SEC;
    poly_ntt_GPU_speed(&ap);
    // batch_ntt(&aps);
    double endTime = (double)clock()/CLOCKS_PER_SEC;
    double timeElapsed = endTime - startTime;
    t_time[i] = timeElapsed;
    //printf("Time: %f \n", timeElapsed);
  }
  double end = now();
  printf("Avg time: %f ms\n", (end - start)/NTESTS);
  print_result_time("NTT GPU: ", t_time, NTESTS);
  

  for(i=0;i<NTESTS;i++) {
    poly_ntt_GPU_speed(&ap);
    t_time[i] = g_ctx.time;
  }
  print_result_time("NTT GPU event timing: ", t_time, NTESTS);


  start = now();
  for(i=0;i<NTESTS;i++) {
    // t[i] = cpucycles();
    clock_t startTime = (double)clock()/CLOCKS_PER_SEC;
    poly_ntt(&ap);
    double endTime = (double)clock()/CLOCKS_PER_SEC;
    double timeElapsed = endTime - startTime;
    t_time[i] = timeElapsed;
  }
  end=now();
  printf("Avg time: %f ms\n", (end - start)/NTESTS);
  print_result_time("NTT: ", t_time, NTESTS);

  for(i=0;i<NTESTS;i++) {
    t[i] = cpucycles();
    poly_invntt_tomont(&ap);
  }
  print_results("INVNTT: ", t, NTESTS);


  printf("Running NTT on CPU for comparison...\n");
  start = now();
  for(i=0;i<NTESTS;i++) {
    ntt(ap.coeffs);
  }
  end = now();
  printf("Avg time: %f ms\n", (end - start)/NTESTS);


  printf("NTT GPU batch test...\n");
  poly aps[NTESTS];
  for(i=0;i<NTESTS;i++) {
    poly_getnoise_eta1(&aps[i], seed, (uint8_t)i);
  }

  opencl_cleanup();
  return 0;
}
