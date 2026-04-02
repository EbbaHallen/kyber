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


 // Batched + combining levels
kernel void ntt(__global short *r){
  __private unsigned int len, start, j, k, group;
  __private short t, zeta, j1, j2, g1, g2, g3, g4;
  const int tid = get_global_id(0);
  const int block = get_global_id(1);
  int base = block * 256; // base index for this polynomial in batch
  // TODO Fix indexing and so that each kernel accesses correct poly
  k = 1;
  len = 128;

  

  zeta = zetas[k + (tid/len)];
  j1 = (tid/len) * len + tid;
  j2 = (tid/len) * len + tid + 64;
  t = fqmul(zeta, r[j1+len]);
  g1 = r[j1] - t;
  g2 = r[j1] + t;
  t=fqmul(zeta, r[j2+len]);
  g3 = r[j2] - t;
  g4 = r[j2] + t;
  k = k << 1;

  len = 64;
  zeta = zetas[k + (tid/len)];
  j1 = (tid/len) * len + tid;
  j2 = (tid/len) * len + tid + 128;
  // 64-128
  t = fqmul(zeta, g4);
  r[j1+len]= g2 - t;
  r[j1]= g2 + t;
  // barrier(CLK_GLOBAL_MEM_FENCE);

  zeta = zetas[k + ((tid+64) /len) ];
  t=fqmul(zeta, g3);
  r[j2 + len] = g1 - t;
  r[j2] = g1 + t;
  k = k << 1;
  // len = 32;

  // barrier(CLK_GLOBAL_MEM_FENCE);
  for(len = 32; len >=2; len >>=1) {
    zeta = zetas[k + (tid/len)];
    j = (tid/len) * len + tid;
    t = fqmul(zeta, r[j + len]);
    r[j + len] = r[j] - t;
    r[j] = r[j] + t;

    zeta = zetas[k + ((tid+64) /len) ];
    j = (tid/len) * len + tid + 128;
    t = fqmul(zeta, r[j + len]);
    r[j + len] = r[j] - t;
    r[j] = r[j] + t;
    
    k = k << 1;
    barrier(CLK_GLOBAL_MEM_FENCE);
  }

  // r[tid + base] = barrett_reduce(local_r[tid]);
  // r[tid + 128 + base] = barrett_reduce(local_r[tid +128]);
  // r[tid + base] = local_r[tid];
  // r[tid + 128 + base] = local_r[tid +128];
}
