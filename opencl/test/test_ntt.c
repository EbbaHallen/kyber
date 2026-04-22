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

int compare_poly(const int16_t *a, const int16_t *b, const int len) {
  for(int i = 0; i < len; i++) {
    if(a[i] != b[i]) {
      printf("Mismatch at index %d: %d != %d\n", i, a[i], b[i]);
      return 0; // Not equal
    }
  }
  return 1; // Equal
}

void print_poly(const int16_t *coeffs) {
  for(int i = 0; i <10; i++) {
    printf("%d ", coeffs[i]);
  }
  printf("\n");
}
void verify_ntt(const char* msg, const int16_t *ntt_result, const int16_t *expected, const int len) {
  if  (! compare_poly(ntt_result, expected, len)) {
    printf("%s: verification failed!\n", msg);
    print_poly(ntt_result);
    print_poly(expected);
    exit(1);
  } 
  // printf("%s: verification passed.\n", msg);
}


/* Print throughput information */
void print_throughput(const char *s, double *time, size_t tlen) {
  double total_time = 0;
  for(size_t i = 0; i < tlen; i++) {
    total_time += time[i];
  }
  double average_time = total_time / tlen ; // Average time per batch in milliseconds
  double throughput = ((double) BATCH_SIZE) / (average_time / 1000.0); // elements per second
  printf("%s Total time: %.4f ms\n", s, total_time);
  printf("%s Average time: %.4f microseconds\n", s, average_time/BATCH_SIZE * 1000.0);
  printf("%s Throughput: %.2f elements/second\n\n", s, throughput);

}

void print_throughput_single(const char *s, double time, int n_tests) {
  double average_time = time / n_tests; // Average time per batch in milliseconds
  double throughput = ((double) BATCH_SIZE) / (average_time / 1000.0); // elements per second
  printf("%s Total time: %.4f ms\n", s, time);
  printf("%s Average time: %.4f microseconds\n", s, average_time/BATCH_SIZE * 1000.0);
  printf("%s Throughput: %.2f elements/second\n\n", s, throughput);
}

void test_intt_ntt_consistency() {
  poly ap_cpu;
  poly ap_gpu;
  poly ap_original;

  // Generate random polynomial
  // poly_getnoise_eta1(&ap_original, seed, 0);
  // poly_getnoise_eta2(&ap_original, seed, 0);
  ap_original.coeffs[0] = 1; // Random coefficient in [0, KYBER_Q-1]
  ap_cpu.coeffs[0] = 1; // Random coefficient in [0, KYBER_Q-1]
  ap_gpu.coeffs[0] = 1; // Random coefficient in [0, KYBER_Q-1]
  ap_original.coeffs[1] = 1; // Random coefficient in [0, KYBER_Q-1]
  ap_cpu.coeffs[1] = 1; // Random coefficient in [0, KYBER_Q-1]
  ap_gpu.coeffs[1] = 1; // Random coefficient in [0, KYBER_Q-1]
  for(int i = 2; i < KYBER_N; i++) {
    ap_original.coeffs[i] = 0;
    ap_cpu.coeffs[i] = ap_original.coeffs[i];
    ap_gpu.coeffs[i] = ap_original.coeffs[i];
  }
  verify_ntt("Before NTT CPU", ap_cpu.coeffs, ap_original.coeffs, KYBER_N);
  verify_ntt("Before NTT GPU", ap_gpu.coeffs, ap_original.coeffs, KYBER_N);
  // Apply NTT and then INTT on CPU
  poly_ntt(&ap_cpu);
  print_poly(ap_cpu.coeffs);
  poly_invntt_tomont(&ap_cpu);
  print_poly(ap_cpu.coeffs);

  // Apply NTT and then INTT on GPU
  poly_ntt_GPU_speed(&ap_gpu);
  poly_invntt_tomont_GPU(&ap_gpu);

  verify_ntt("INTT(NTT(a)) CPU", ap_cpu.coeffs, ap_original.coeffs, KYBER_N);
  verify_ntt("INTT(NTT(a)) GPU", ap_gpu.coeffs, ap_original.coeffs, KYBER_N);
  // Verify that the original polynomial is recovered
}

void test_multiplication() {
  poly a, b, r_cpu, r_gpu;
  // Initialize a and b with random coefficients
  for(int i = 0; i < KYBER_N; i++) {
    a.coeffs[i] = rand() % KYBER_Q;
    b.coeffs[i] = rand() % KYBER_Q;
  }

  // Perform polynomial multiplication on CPU
  poly_basemul_montgomery(&r_cpu, &a, &b);

  // Perform polynomial multiplication on GPU
  poly_basemul_montgomery_GPU(&r_gpu, &a, &b);

  // Verify that the results are the same
  verify_ntt("Polynomial multiplication single", r_gpu.coeffs, r_cpu.coeffs, KYBER_N);
}
void test_multiplication_batch() {
  poly_batch a, b, r_cpu, r_gpu;
  // Initialize a and b with random coefficients
  for(int i = 0; i < KYBER_N * BATCH_SIZE; i++) {
    a.coeffs[i] = rand() % KYBER_Q;
    b.coeffs[i] = rand() % KYBER_Q;
  }

  // Perform polynomial multiplication on CPU
  poly_basemul_montgomery_batch(&r_cpu, &a, &b);

  // Perform polynomial multiplication on GPU
  poly_basemul_montgomery_GPU_batch(&r_gpu, &a, &b);

  // Verify that the results are the same
  verify_ntt("Polynomial multiplication batch", r_gpu.coeffs, r_cpu.coeffs, KYBER_N * BATCH_SIZE);
}

void test_ntt_single() {
  poly ap_cpu;
  poly ap_gpu;

  // Generate random polynomial
  poly_getnoise_eta1(&ap_cpu, seed, 0);
  poly_getnoise_eta2(&ap_cpu, seed, 0);
  memcpy(ap_gpu.coeffs, ap_cpu.coeffs, KYBER_N * sizeof(int16_t));

  // Apply NTT on CPU and GPU
  poly_ntt(&ap_cpu);
  printf("CPU NTT\n");
  poly_ntt_GPU_speed(&ap_gpu);
  printf("GPU NTT\n");

  verify_ntt("NTT GPU single mode", ap_cpu.coeffs, ap_gpu.coeffs, KYBER_N);
}

void test_ntt_batch() {
  poly_batch ap_cpu;
  poly_batch ap_gpu;
  poly ap_cpu_single;

  // Generate random polynomials
  for (int i = 0; i < BATCH_SIZE; i++) {
    poly tmp;
    poly_getnoise_eta1(&tmp, seed, 0);
    poly_getnoise_eta2(&tmp, seed, 0);

    memcpy(&ap_cpu.coeffs[i * KYBER_N], tmp.coeffs,
          KYBER_N * sizeof(int16_t));
    memcpy(&ap_gpu.coeffs[i * KYBER_N], tmp.coeffs,
          KYBER_N * sizeof(int16_t));
  }
    memcpy(ap_cpu_single.coeffs, ap_cpu.coeffs, KYBER_N * sizeof(int16_t));


  // Apply NTT on CPU and GPU
  poly_ntt_batch(&ap_cpu);
  poly_ntt_GPU_speed_batch(&ap_gpu);
  poly_ntt(&ap_cpu_single);

  for(int i=0;i<BATCH_SIZE;i++) {
    verify_ntt("NTT CPU batch mode", ap_cpu.coeffs+(i*KYBER_N), ap_cpu_single.coeffs, KYBER_N);
  }
  verify_ntt("NTT GPU batch", ap_cpu.coeffs, ap_gpu.coeffs, KYBER_N*BATCH_SIZE);
}

void test_intt_single(){
  poly ap_cpu;
  poly ap_gpu;
  poly ap_original;

  // Generate random polynomial
  poly_getnoise_eta1(&ap_original, seed, 0);
  poly_getnoise_eta2(&ap_original, seed, 0);
  memcpy(ap_cpu.coeffs, ap_original.coeffs, KYBER_N * sizeof(int16_t));
  memcpy(ap_gpu.coeffs, ap_original.coeffs, KYBER_N * sizeof(int16_t));

  // Apply NTT on CPU and GPU
  poly_invntt_tomont(&ap_cpu);
  poly_invntt_tomont_GPU(&ap_gpu);

  verify_ntt("INTT CPU", ap_cpu.coeffs, ap_gpu.coeffs, KYBER_N);
}

void test_intt_batch() {
  poly_batch ap_cpu;
  poly_batch ap_gpu;
  poly ap_cpu_single;

  // Generate random polynomials
  for (int i = 0; i < BATCH_SIZE; i++) {
    poly tmp;
    poly_getnoise_eta1(&tmp, seed, 0);
    poly_getnoise_eta2(&tmp, seed, 0);

    memcpy(&ap_cpu.coeffs[i * KYBER_N], tmp.coeffs,
          KYBER_N * sizeof(int16_t));
    memcpy(&ap_gpu.coeffs[i * KYBER_N], tmp.coeffs,
          KYBER_N * sizeof(int16_t));
  }
  memcpy(ap_cpu_single.coeffs, ap_cpu.coeffs, KYBER_N * sizeof(int16_t));

  // Apply INTT on CPU and GPU
  poly_invntt_tomont_batch(&ap_cpu);
  poly_invntt_tomont_GPU_batch(&ap_gpu);
  poly_invntt_tomont(&ap_cpu_single);

  for(int i=0;i<BATCH_SIZE;i++) {
    verify_ntt("INTT CPU batch mode", ap_cpu.coeffs+(i*KYBER_N), ap_cpu_single.coeffs, KYBER_N);
  }
  verify_ntt("INTT GPU batch mode", ap_gpu.coeffs, ap_cpu.coeffs, KYBER_N*BATCH_SIZE);
}


int main(void)
{
  unsigned int i;
  uint8_t coins32[KYBER_SYMBYTES];
  uint8_t coins64[2*KYBER_SYMBYTES];
  polyvec matrix[KYBER_K];
  poly_batch aps_gpu;
  poly_batch aps_cpu;
  poly ap_cpu;
  poly ap_gpu;
  poly ap_original;
  double start, end, total_time_ms;

  opencl_init();

  // test_intt_ntt_consistency();
  test_multiplication();
  test_ntt_single();
  test_ntt_batch();
  test_intt_single();
  test_intt_batch();

  /*-----------------------------------------------------------------*/
  /* Speed tests */
  
  /* NTT CPU batch */
  printf("NTT CPU speed batch test... %d\n", BATCH_SIZE);
  start = get_time_sec();
  for(i=0;i<NTESTS;i++) {
    double startTime = (double)clock()/CLOCKS_PER_SEC;
    poly_ntt_batch(&aps_cpu);
    double endTime = (double)clock()/CLOCKS_PER_SEC;
    double timeElapsed = endTime - startTime;
    t_time[i] = timeElapsed;
  }
  end = get_time_sec();
  total_time_ms = (end - start) * 1000;
 // print_result_time("NTT CPU batch timing: ", t_time, NTESTS);
  print_throughput_single("NTT CPU batch timing: ", total_time_ms, NTESTS);


  /* NTT GPU speed batch */
  // warmup
  for(i=0;i<10;i++) {
    poly_ntt_GPU_speed_batch(&aps_gpu);
    t_time[i] = g_ctx.time;
  }


  printf("NTT GPU speed batch test... %d\n", BATCH_SIZE);
  for(i=0;i<NTESTS;i++) {
    poly_ntt_GPU_speed_batch(&aps_gpu);
    t_time[i] = g_ctx.time;
  }
  print_throughput("NTT GPU event timing: ", t_time, NTESTS);


  start = get_time_sec();
  for(i=0;i<NTESTS;i++) {
    poly_ntt_GPU_speed_batch(&aps_gpu);
  }
  end = get_time_sec();
  total_time_ms = (end - start) * 1000;
  print_throughput_single("NTT GPU total timing: ", total_time_ms, NTESTS);
  

  /* INV NTT */

  printf("INV NTT \n");
  /* NTT CPU batch */
  printf("NTT CPU speed batch test... %d\n", BATCH_SIZE);
  start = get_time_sec();
  for(i=0;i<NTESTS;i++) {
    double startTime = (double)clock()/CLOCKS_PER_SEC;
    poly_ntt_batch(&aps_cpu);
    double endTime = (double)clock()/CLOCKS_PER_SEC;
    double timeElapsed = endTime - startTime;
    t_time[i] = timeElapsed;
  }
  end = get_time_sec();
  total_time_ms = (end - start) * 1000;
 // print_result_time("NTT CPU batch timing: ", t_time, NTESTS);
  print_throughput_single("NTT CPU batch timing: ", total_time_ms, NTESTS);

  /* INV NTT GPU speed batch */
  // warmup
  for(i=0;i<10;i++) {
    poly_invntt_tomont_GPU_batch(&aps_gpu);
    t_time[i] = g_ctx.time;
  }


  printf("NTT GPU speed batch test... %d\n", BATCH_SIZE);
  for(i=0;i<NTESTS;i++) {
    poly_invntt_tomont_GPU_batch(&aps_gpu);
    t_time[i] = g_ctx.time;
  }
  print_throughput("NTT GPU event timing: ", t_time, NTESTS);


  opencl_cleanup();
  return 0;
}
