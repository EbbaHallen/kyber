#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "cpucycles.h"
#include "speed_print.h"

static int cmp_uint64(const void *a, const void *b) {
  if(*(uint64_t *)a < *(uint64_t *)b) return -1;
  if(*(uint64_t *)a > *(uint64_t *)b) return 1;
  return 0;
}

static uint64_t median(uint64_t *l, size_t llen) {
  qsort(l,llen,sizeof(uint64_t),cmp_uint64);

  if(llen%2) return l[llen/2];
  else return (l[llen/2-1]+l[llen/2])/2;
}

static uint64_t average(uint64_t *t, size_t tlen) {
  size_t i;
  uint64_t acc=0;

  for(i=0;i<tlen;i++)
    acc += t[i];

  return acc/tlen;
}

void print_results(const char *s, uint64_t *t, size_t tlen) {
  size_t i;
  static uint64_t overhead = -1;

  if(tlen < 2) {
    fprintf(stderr, "ERROR: Need a least two cycle counts!\n");
    return;
  }

  if(overhead  == (uint64_t)-1)
    overhead = cpucycles_overhead();

  tlen--;
  for(i=0;i<tlen;++i)
    t[i] = t[i+1] - t[i] - overhead;

  printf("%s\n", s);
  printf("median: %llu cycles/ticks\n", (unsigned long long)median(t, tlen));
  printf("average: %llu cycles/ticks\n", (unsigned long long)average(t, tlen));
  printf("\n");
}

static int cmp_double(const void *a, const void *b) {
  if(*(double *)a < *(double *)b) return -1;
  if(*(double *)a > *(double *)b) return 1;
  return 0;
}

static double median_time(double *l, size_t llen) {
  qsort(l,llen,sizeof(double),cmp_double);

  if(llen%2) return l[llen/2];
  else return (l[llen/2-1]+l[llen/2])/2;
}

static double average_time(double *t, size_t tlen) {
  size_t i;
  double acc=0;

  for(i=0;i<tlen;i++)
    acc += t[i];

  return acc/tlen;
}

void print_result_time(const char *s, double *t, size_t tlen) {
  // size_t i;
  printf("%s\n", s);
  printf("median: %f ms\n", median_time(t, tlen));
  printf("average: %f ms\n", average_time(t, tlen));
  printf("\n");
}
