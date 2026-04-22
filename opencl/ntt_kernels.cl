#define KYBER_Q 3329
#define QINV -3327 // q^-1 mod 2^16


short montgomery_reduce(int a)
{
  short t;

  t = (short)a*QINV;
  t = (a - (int)t*KYBER_Q) >> 16;
  return t;
}

short barrett_reduce(short a) {
  short t;
  short v = ((1<<26) + KYBER_Q/2)/KYBER_Q;

  t  = ((int)v*a + (1<<25)) >> 26;
  t *= KYBER_Q;
  return a - t;
}

__constant short zetas[128] = {
  -1044,  -758,  -359, -1517,  1493,  1422,   287,   202,
   -171,   622,  1577,   182,   962, -1202, -1474,  1468,
    573, -1325,   264,   383,  -829,  1458, -1602,  -130,
   -681,  1017,   732,   608, -1542,   411,  -205, -1571,
   1223,   652,  -552,  1015, -1293,  1491,  -282, -1544,
    516,    -8,  -320,  -666, -1618, -1162,   126,  1469,
   -853,   -90,  -271,   830,   107, -1421,  -247,  -951,
   -398,   961, -1508,  -725,   448, -1065,   677, -1275,
  -1103,   430,   555,   843, -1251,   871,  1550,   105,
    422,   587,   177,  -235,  -291,  -460,  1574,  1653,
   -246,   778,  1159,  -147,  -777,  1483,  -602,  1119,
  -1590,   644,  -872,   349,   418,   329,  -156,   -75,
    817,  1097,   603,   610,  1322, -1285, -1465,   384,
  -1215,  -136,  1218, -1335,  -874,   220, -1187, -1659,
  -1185, -1530, -1278,   794, -1510,  -854,  -870,   478,
   -108,  -308,   996,   991,   958, -1460,  1522,  1628
};

static short fqmul(short a, short b) {
  return montgomery_reduce((short)a*b);
}
kernel void ntt(__global short *r){
  __private unsigned int len, start, j, k, group;
  __private short t, zeta;
  const int tid = get_global_id(0);
  const int block = get_global_id(1);
  int base = block * 256; // base index for this polynomial in batch
  k = 1;

  __local short local_r[256];
  local_r[tid] = r[tid + base];
  local_r[tid + 128] = r[tid + 128 + base];
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int len = 128; len >=2; len >>=1) {
    zeta = zetas[k + (tid/len)];
    j = (tid/len) * len + tid;
    t = fqmul(zeta, local_r[j + len]);
    local_r[j + len] = local_r[j] - t;
    local_r[j] = local_r[j] + t;
    k = k << 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  r[tid + base] = barrett_reduce(local_r[tid]);
  r[tid + 128 + base] = barrett_reduce(local_r[tid +128]);
}

kernel void invntt(__global short *r) {
  __private unsigned int start, len, j, k;
  __private short t, zeta;
  const short f = 1441; // mont^2/128
  const int tid = get_global_id(0);
  const int block = get_global_id(1);
  int base = block * 256;

  __local short local_r[256];
  local_r[tid] = r[tid + base];
  local_r[tid + 128] = r[tid + 128 + base];
  barrier(CLK_LOCAL_MEM_FENCE);

  k = 127;
  for(len = 2; len <= 128; len <<= 1) {
    zeta = zetas[k - (tid/len)];
    j = (tid/len) * len + tid;
    t = local_r[j];
    local_r[j] = barrett_reduce(t + local_r[j + len]);
    local_r[j + len] = local_r[j + len] - t;
    local_r[j + len] = fqmul(zeta, local_r[j + len]);
    k = k >> 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  r[tid + base]       = fqmul(local_r[tid], f);
  r[tid + 128 + base] = fqmul(local_r[tid + 128], f);
}

void basemul(__global short *r, __local short *a, __local short *b, short zeta)
{
  r[0]  = fqmul(a[1], b[1]);
  r[0]  = fqmul(r[0], zeta);
  r[0] += fqmul(a[0], b[0]);
  r[1]  = fqmul(a[0], b[1]);
  r[1] += fqmul(a[1], b[0]);
}

/* Assumes 64 threads per block */
kernel void poly_basemul(__global short *r, __global short *a, __global short *b){
  const int tid = get_global_id(0);
  const int block = get_global_id(1);
  int base = block * 256;
  short zeta = zetas[64 + tid];
  // printf("tid: %d, block: %d\n", tid, block);

  __local short local_a[256];
  __local short local_b[256];
  local_a[tid] = a[tid + base];
  local_a[tid + 64] = a[tid + 64 + base];
  local_a[tid + 128] = a[tid + 128 + base];
  local_a[tid + 192] = a[tid + 192 + base];
  local_b[tid] = b[tid + base];
  local_b[tid + 64] = b[tid + 64 + base];
  local_b[tid + 128] = b[tid + 128 + base];
  local_b[tid + 192] = b[tid + 192 + base];
  barrier(CLK_LOCAL_MEM_FENCE);

  basemul(&r[4*tid + base], &local_a[4*tid], &local_b[4*tid], zeta);
  basemul(&r[4*tid+2 + base], &local_a[4*tid+2], &local_b[4*tid+2], -zeta);
}
