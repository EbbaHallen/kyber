#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "../kem.h"
#include "../params.h"
#include "../indcpa.h"
#include "../polyvec.h"
#include "../poly.h"
#include "../randombytes.h"
#include "speed_print.h"
#include "../opencl.h"
#include "../ntt.h"

#define NTESTS 1000

uint64_t t[NTESTS];
double t_time[NTESTS];
uint8_t seed[KYBER_SYMBYTES] = {0};

// Function to get current time in milliseconds
double get_time_sec() {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec + time.tv_nsec * 1e-9;
}

int compare_poly(const int16_t *a, const int16_t *b) {
  for(int i = 0; i < KYBER_N; i++) {
    if(a[i] != b[i]) {
      return 0; // Not equal
    }
  }
  return 1; // Equal
}

void print_poly(const int16_t *coeffs) {
  for(int i = 0; i < 3; i++) {
    printf("%d ", coeffs[i]);
  }
  printf("\n");
}

/* Print throughput information */
void print_throughput(const char *s, double *time, size_t tlen) {
  double total_time = 0;
  for(size_t i = 0; i < tlen; i++) {
    total_time += time[i];
  }
  double average_time = total_time / tlen;
  printf("%s Total time: %.2f ms\n", s, total_time);
  printf("%s Average time: %.2f ms\n", s, average_time);
  double throughput = ((double) BATCH_SIZE) / (total_time / 1000.0); // elements per second
  printf("%s Throughput: %.2f elements/second\n\n", s, throughput);

}


int main(void)
{
  unsigned int i;
  // uint8_t pk[CRYPTO_PUBLICKEYBYTES];
  // uint8_t sk[CRYPTO_SECRETKEYBYTES];
  // uint8_t ct[CRYPTO_CIPHERTEXTBYTES];
  // uint8_t key[CRYPTO_BYTES];
  uint8_t coins32[KYBER_SYMBYTES];
  uint8_t coins64[2*KYBER_SYMBYTES];
  polyvec matrix[KYBER_K];
  poly_batch aps_cpu;
  poly_batch aps;
  poly ap;
  poly ap_gpu;

  opencl_init();

  randombytes(coins32, KYBER_SYMBYTES);
  randombytes(coins64, 2*KYBER_SYMBYTES);

  gen_matrix(matrix, seed, 0);
 
  for (i = 0; i < BATCH_SIZE; i++) {
    poly tmp;

    poly_getnoise_eta1(&tmp, seed, i);
    poly_getnoise_eta2(&tmp, seed, i);

    memcpy(&aps.coeffs[i * KYBER_N], tmp.coeffs,
          KYBER_N * sizeof(int16_t));
    memcpy(&aps_cpu.coeffs[i * KYBER_N], tmp.coeffs,
          KYBER_N * sizeof(int16_t));
  }


  memcpy(ap.coeffs, aps.coeffs, KYBER_N * sizeof(int16_t));
  memcpy(ap_gpu.coeffs, ap.coeffs, KYBER_N * sizeof(int16_t));

  if(!compare_poly(ap.coeffs, ap_gpu.coeffs)) {
    printf("Mismatch in noise generation!\n");
    return 1;
  }
   // NTT GPU
  poly_ntt(&ap);
  poly_ntt_GPU_speed(&ap_gpu);

  if(!compare_poly(ap.coeffs, ap_gpu.coeffs)) {
    printf("CPU vs GPU NTT results mismatch!\n");
    printf("First 3 coefficients of CPU NTT: %d %d %d\n", ap.coeffs[0], ap.coeffs[1], ap.coeffs[2]);
    printf("First 3 coefficients of GPU NTT: %d %d %d\n", ap_gpu.coeffs[0], ap_gpu.coeffs[1], ap_gpu.coeffs[2]);
    exit(1);
  } 

  // Verify correctness of GPU NTT
  poly_ntt_GPU_speed_batch(&aps);
  poly_ntt_batch(&aps_cpu);

  int correct = 1;
  correct = compare_poly(ap.coeffs, aps_cpu.coeffs); // compare first polynomial in batch
  if(!correct) 
  {
    printf("Error in CPU batch\n");
    exit(1);
  }

  correct = compare_poly(aps.coeffs, aps_cpu.coeffs); // compare first polynomial in batch
  if(!correct) 
  {
    printf("Error in GPU batch\n");
    printf("First 3 coefficients of CPU batch NTT: %d %d %d\n", aps_cpu.coeffs[0], aps_cpu.coeffs[1], aps_cpu.coeffs[2]);
    printf("First 3 coefficients of GPU batch NTT: %d %d %d\n", aps.coeffs[0], aps.coeffs[1], aps.coeffs[2]);
    exit(1);
  }

  printf("Verifying GPU NTT results...\n");
  for(i=0;i<KYBER_N*BATCH_SIZE;i++) {
    if(!compare_poly(aps.coeffs, aps_cpu.coeffs)) {
      correct = 0;
      printf("Mismatch at index %d: GPU %d, CPU %d\n", i, aps.coeffs[i], aps_cpu.coeffs[i]);
      break;
    }
  }
  if(correct) {
    printf("GPU NTT matches CPU NTT!\n");
  } else {
    printf("GPU NTT does NOT match CPU NTT!\n");
  }
  
  /*-----------------------------------------------------------------*/
  /* Speed tests */
  
  /* NTT CPU batch */
  printf("NTT CPU speed batch test... %d\n", BATCH_SIZE);
  double start = get_time_sec();
  for(i=0;i<NTESTS;i++) {
    double startTime = (double)clock()/CLOCKS_PER_SEC;
    poly_ntt_batch(&aps_cpu);
    double endTime = (double)clock()/CLOCKS_PER_SEC;
    double timeElapsed = endTime - startTime;
    t_time[i] = timeElapsed;
  }
  double end = get_time_sec();
  double total_time_ms = (end - start) * 1000;
 // print_result_time("NTT CPU batch timing: ", t_time, NTESTS);
  print_throughput("NTT CPU batch timing: ", &total_time_ms, 1);


  /* NTT GPU speed batch */

  // warmup
  for(i=0;i<10;i++) {
    poly_ntt_GPU_speed_batch(&aps);
    t_time[i] = g_ctx.time;
  }


  printf("NTT GPU speed batch test... %d\n", BATCH_SIZE);
  for(i=0;i<NTESTS;i++) {
    poly_ntt_GPU_speed_batch(&aps);
    t_time[i] = g_ctx.time;
  }
  print_throughput("NTT GPU event timing: ", t_time, NTESTS);



  // double start = now();
  // for(i=0;i<NTESTS;i++) {
  //   clock_t startTime = (double)clock()/CLOCKS_PER_SEC;
  //   poly_ntt_GPU_speed_batch(&aps);
  //   // batch_ntt(&aps);
  //   double endTime = (double)clock()/CLOCKS_PER_SEC;
  //   double timeElapsed = endTime - startTime;
  //   t_time[i] = timeElapsed;
  //   //printf("Time: %f \n", timeElapsed);
  // }
  // double end = now();
  // printf("Avg time: %f ms\n", (end - start)/NTESTS);
  // print_result_time("NTT GPU: ", t_time, NTESTS);
  // print_throughput("NTT GPU: ", BATCH_SIZE, (end - start), NTESTS);
  


  // double start = now();
 // for(i=0;i<NTESTS;i++) {
    // t[i] = cpucycles();
 //   clock_t startTime = (double)clock()/CLOCKS_PER_SEC;
 //   poly_ntt(&ap);
 //   double endTime = (double)clock()/CLOCKS_PER_SEC;
 //   double timeElapsed = endTime - startTime;
  //  t_time[i] = timeElapsed;
 // }
  // double end=now();
  // printf("Avg time: %f ms\n", (end - start)/NTESTS);
  //print_result_time("NTT: ", t_time, NTESTS);



  // printf("Running NTT on CPU for comparison...\n");
  // start = now();
  // for(i=0;i<NTESTS;i++) {
  //   ntt(ap.coeffs);
  // }
  // end = now();
  // printf("Avg time: %f ms\n", (end - start)/NTESTS);


  
  
  opencl_cleanup();
  return 0;
}
