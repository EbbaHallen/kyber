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

static short4 fqmul(short4 a, short4 b) {
  int4 prod = convert_int4(a)*convert_int4(b);
  return montgomery_reduce_vec(prod);
}


 // Batched + shared memory NTT kernel
kernel void ntt(__global short *r){
  __private unsigned int len, start, j, k, group;
  __private short zeta;
  __private short4 t;
  const int tid = get_global_id(0) * 4; // each thread processes 4 coefficients
  const int block = get_global_id(1);
  int base = block * 256; // base index for this polynomial in batch
  // TODO Fix indexing and so that each kernel accesses correct poly
  k = 1;
  printf("block: %d, tid: %d\n", block, tid);

  // __local short4 local_r[256];
  // local_r[tid] = vload4(0, r + tid + base);
  __local short local_r[256];
  local_r[tid] = r[tid + base];
  local_r[tid + 128] = r[tid + 128 + base];
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int len = 128; len >=4; len >>=1) {
    zeta = zetas[k + (tid/len)]; // same zeta
    j = (tid/len) * len + tid;
    short4 r_j = vload4(0, local_r + j);
    short4 r_j_len = vload4(0, local_r + j + len);

    t = fqmul(zeta, r_j_len);
    vstore4(r_j - t, 0, local_r + j + len);
    vstore4(r_j + t, 0, local_r + j);
    k = k << 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // last iteration with len = 2

  r[tid + base] = barrett_reduce(local_r[tid]);
  r[tid + 128 + base] = barrett_reduce(local_r[tid +128]);
  // r[tid + base] = local_r[tid];
  // r[tid + 128 + base] = local_r[tid +128];
}

