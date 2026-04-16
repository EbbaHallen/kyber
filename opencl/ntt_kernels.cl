#define KYBER_Q 3329
#define QINV -3327 // q^-1 mod 2^16


short montgomery_reduce(int a)
{
  short t;

  t = (short)a*QINV;
  t = (a - (int)t*KYBER_Q) >> 16;
  return t;
}

short4 montgomery_reduce_vec(int4 a)
{
  short4 a_low = convert_short4(a);
  short4 t;

  t = a_low* (short)QINV;
  int4 t_int = convert_int4(t);
  int4 result = (a - t_int*KYBER_Q) >> 16;
  return convert_short4(result);
}

short barrett_reduce(short a) {
  short t;
  short v = ((1<<26) + KYBER_Q/2)/KYBER_Q;

  t  = ((int)v*a + (1<<25)) >> 26;
  t *= KYBER_Q;
  return a - t;
}

short4 barrett_reduce_vec(short4 a)
{
    const int v = ((1 << 26) + KYBER_Q/2) / KYBER_Q;
    int4 a_int = convert_int4(a);
    int4 t = (a_int * v + (1 << 25)) >> 26;
    t = t * KYBER_Q;
    return convert_short4(a_int - t);
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

static short4 fqmul(short a, short4 b) {
  int4 prod = (int)a * convert_int4(b);
  return montgomery_reduce_vec(prod);
}
 // Batched + shared memory NTT kernel
 // Each work-group processes one polynomial, and each thread processes 4 coefficients
 // 32 threads
kernel void ntt(__global short *r){
  __private unsigned int len, start, j, k, group;
  __private short zeta;
  __private short4 t;
  const int tid = get_global_id(0); // each thread processes 4 coefficients
  const int block = get_global_id(1);
  int base = block * 256; // base index for this polynomial in batch
  k = 1;
  printf("Thread %d processing polynomial %d\n", tid, block);

  // __local short4 local_r[256];
  // local_r[tid] = vload4(0, r + tid + base);
  __local short4 local_r[64];
  local_r[tid] = vload4(0, r + tid*4 + base);
  local_r[tid + 32] = vload4(0, r + tid*4 + base + 128);
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int len = 128; len >=4; len >>=1) {
    zeta = zetas[k + (tid*4/len)]; // same zeta
    j = (tid*4/len) * len + tid*4;
    int vj     = j >> 2;
    int vj_len = (j + len) >> 2;
    t = fqmul(zeta, local_r[vj_len]);
    local_r[vj_len] = local_r[vj] - t;
    local_r[vj] = local_r[vj] + t;
    k = k << 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // last iteration with len = 2
  len = 2;
  __local short scalar = (__local short*)local_r;

  zeta = zetas[k + (tid*4/len)];
  j = (tid*4/len) * len + tid;

  t = fqmul(zeta, scalar[j + len]);
  scalar[j + len] = scalar[j] - t;
  scalar[j] = scalar[j] + t;


  t = fqmul(zeta, scalar[j + 1 + len]);
  scalar[j + 1 + len] = scalar[j + 1] - t;
  scalar[j + 1] = scalar[j + 1] + t;
  zeta = zetas[k + (tid*4/len) + 1];
  j = (tid*4+2/len) * len + tid*4 + 2;
  t = fqmul(zeta, scalar[j + len]);
  scalar[j + len] = scalar[j] - t;
  scalar[j] = scalar[j] + t;
  t = fqmul(zeta, scalar[j + 1 + len]);
  scalar[j + 1 + len] = scalar[j + 1] - t;
  scalar[j + 1] = scalar[j + 1] + t;

  short4 reduced = barrett_reduce_vec(vload4(0, local_r + tid));
  short4 reduced2 = barrett_reduce_vec(vload4(0, local_r + tid + 32));
  vstore4(reduced, 0, r + tid*4 + base);
  vstore4(reduced2, 0, r + tid*4 + base + 128);
}