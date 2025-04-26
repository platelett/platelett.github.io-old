---
title: 任意模数 NTT
date: 2021-04-20 09:08:39
updated: 2021-04-20 09:08:39
tags: [知识总结,快速傅里叶变换,模板]
categories: 算法
---
> 给定一个 $n$ 次多项式 $A$ 和一个 $m$ 次多项式 $B$，计算 $A \times B$，系数对 $p$ 取模。
>
> $n,m \le 10^5$

常见的有两种做法：

- 先做三模数 ```NTT``` 再用中国剩余定理合并。

<!-- more -->

- 取 $M = \lceil \sqrt p \rceil$，把每个系数 $x$ 拆成 $M \cdot \lfloor \frac Mx \rfloor$ 和 $M \bmod x$ 两部分。

这里讲第二种做法及其优化。

### 做法

设
$$
A0_i=\lfloor \frac {A_i}M \rfloor,A1_i=A_i \bmod M\\\\B0_i=\lfloor \frac {B_i}M \rfloor,B1_i=B_i \bmod M
$$
于是
$$
A = M \cdot A0 + A1\\\\B = M \cdot B0  + B1
$$
进一步
$$
A \times B = M^2 \cdot A0 \times B0 + M(A0 \times B1 + A1 \times B0) + A1 \times B1
$$
先对 $A0,A1,B0,B1$ 做一遍 ```DFT```，求出 $A0 \times B0, A0 \times B1 + A1 \times B0,A1 \times B1$ 的点值表示后再分别 ```IDFT```，共 $7$ 次。

注意系数可能达到 $10^{14}$，需要用 $w_n^k=\cos \frac {2k\pi}n+i \cdot \sin \frac {2k\pi}n$ 对 $w_n^k$ 进行预处理保证精度。

### 优化

下面考虑这样两件事：

- 现在要对**实系数**多项式 $A,B$ 进行 ```DFT```。

  我们定义
  $$
  P = A + iB\\\\
  Q = A - iB
  $$
  推导可以得到一个很优美的结论：
  $$
  \text {conj}(Q(w_n^k))=P(\text{conj}({w_n^{k}}))
  $$
  于是只需一次 ```DFT``` 就可以求出 $A,B$ 的点值表示。

- 现在已知**实系数**多项式 $A,B$ 的点值表示 $A',B'$，要求 $A,B$。

  我们定义
  $$
  P = A' + iB'
  $$
  对 $P$ 进行 ```IDFT``` 得到 $P'$，于是
  $$
  P' = A + iB
  $$
  因此 $P'$ 的实数部分就是 $A$, 虚数部分就是 $B$。

  于是只需一次 ```IDFT``` 就可以达到对 $A',B'$ 分别做 ```IDFT``` 的效果.

将 $7$ 次 ```DFT``` 两两配对可以合并成 $4$ 次 ```DFT```。

定义

```cpp
typedef complex <lf> cmp;
const lf PI = acos(-1);
const cmp I(0, 1);

int n, m, P;
int M, lim = 1, bit = -1, rev[N];
cmp w[N], a0[N], a1[N], b0[N], b1[N];
```

函数

```cpp
void FFT(cmp a[], int t) {
    if(t) reverse(a + 1, a + lim);
    For(i, 0, lim) if(i < rev[i]) swap(a[i], a[rev[i]]);
    for(int i = 1; i < lim; i *= 2) for(int j = 0; j < lim; j += i * 2) For(k, j, j + i) {
        cmp x = a[k], y = a[i + k] * w[lim / i * (k - j)];
        a[k] = x + y, a[i + k] = x - y;
    }
    if(t) For(i, 0, lim) a[i] *= inv;
}
void FFT2(cmp a[], cmp b[]) {
    For(i, 0, lim) a[i] += b[i] * I;
    FFT(a, 0);
    For(i, 0, lim) b[i] = conj(a[i ? lim - i : 0]);
    For(i, 0, lim) {
        cmp x = a[i], y = b[i];
        a[i] = (y + x) * 0.5, b[i] = (y - x) * 0.5 * I;
    }
}
ll num(cmp x) { return M * ll(real(x) + 0.5) % P + ll(imag(x) + 0.5); }
```

预处理

```cpp
M = sqrt(P);
rep(i, 0, n) a0[i] = A[i] / M, a1[i] = A[i] % M;
rep(i, 0, m) b0[i] = B[i] / M, b1[i] = B[i] % M;
while(lim <= n + m) lim *= 2, bit++;
inv = 1. / lim;
For(i, 0, lim) {
    rev[i] = rev[i / 2] / 2 | (i & 1) << bit;
    w[i] = cmp(cos(PI / lim * i), sin(PI / lim * i));
}
```

使用

```cpp
FFT2(a0, a1), FFT2(b0, b1);
For(i, 0, lim) {
    cmp t = a0[i] + I * a1[i];
    b0[i] *= t, b1[i] *= t;
}
FFT(b0, 1), FFT(b1, 1);
rep(i, 0, n + m) C[i] = (M * num(b0[i]) + num(b1[i])) % P;
```