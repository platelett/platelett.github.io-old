---
title: Trinity | AGC021F
tags: [动态规划，生成函数]
categories: AGC
---

> [题目链接](https://atcoder.jp/contests/agc021/tasks/agc021_f)
> 
> 对于 $n\times m$ 的 $01$ 矩阵 $mat$，定义序列 $A,B,C$：
> 
> - $A_i(1\le i \le n)$ 表示最小的 $j$ 满足 $mat_{i,j}=1$（如果没有则为 $m+1$​）。
> 
> - $B_i(1\le i \le m)$ 表示最小的 $j$ 满足 $mat_{j,i}=1$（如果没有则为 $n+1$​）。
> ​
> - $C_i(1\le i \le n)$ 表示最大的 $j$ 满足 $mat_{j,i}=1$（如果没有则为 $0$​）。
> 
> 有多少种不同的三元组 $(A,B,C)$？模 $998244353$。
> 
> $n \le 8000,m\le 200$

设 $dp_{m,n}$ 表示 **强制** 每行至少有一个黑格时 $n\times m$ 矩阵的答案。

那么答案就是 $\sum_{i=0}^N\binom Nidp_{M,i}$。

转移考虑哪些行的第一个黑格在最后一列，分两种情况：

- 没有任何一行的第一个黑格在最后一列，那么最后一列可以没有黑格，可以恰好有一个，也可以多于一个，三种的方案对应的情况数分别是 $1,n,\binom n2$，转移系数为 $1+n+\binom n2$。

- 有 $k(k\ge 1)$ 行的第一个黑格在最后一列，相当于最后一列已经事先填好了 $k$ 个黑格，设这些黑格中自上到下第一个在第 $a$ 行，最后一个在 **倒数** 第 $b$ 行，这种方案对应的情况数是 $ab$，转移系数为

$$
\sum_{a,b}\binom a1\binom{n-a-b}{k-2}\binom b1=\binom{n+2}{k+2}=\binom{n+2}{n-k}
$$

综上，转移式为

$$
dp_{m,n}=dp_{m-1,n}(1+n+\binom n2)+\sum_{k=0}^{n-1}dp_{m-1,k}\binom{n+2}k
$$

转移可以使用 NTT 优化，复杂度 $O(NM\log N)$，下面有复杂度更低的做法。

转移时乘的是组合数，考虑写成 EGF：记 $F_m(x)=\sum dp_{m,n}\frac{x^n}{n!}$。

$$
dp_{m,n}=\sum_{k=0}^ndp_{m-1,k}\binom{n+2}k-ndp_{m-1,n}
$$

设

$$
\begin{aligned}
G(x)&=\sum_n\frac{x^n}{n!}\sum_{k=0}^{n-2}dp_{m-1,k}\binom nk\\
&=F_{m-1}(x)(e^x-x-1)
\end{aligned}
$$

那么把 $G(x)$ 中 $\frac{x^n}{n!}$ 的系数左移 $2$ 就可以得到转移式的第一部分，可以用求导实现左移 $\frac{x^n}{n!}$ 的系数。

易得第二部分为 $-xF_{m-1}'(x)$。

得到 $F_m(x)$ 的转移式：

$$
F_m(x)=(F_{m-1}(x)(e^x-1-x))''-xF_{m-1}'(x)
$$

由于这个转移式中 $F_{m-1}(x)$ 只乘了 $e^x,x$ 两个和 $x$ 有关的东西，考虑把 $F_m(x)$ 写成 $\sum_{a,b}f_{a,b}e^{ax}x^b$，显然 $F_m'(x)$ 也可以写成这个形式，并且 $e^x$ 和 $x$ 和最高次幂都不会比 $F_m(x)$ 大，所以 $F_m(x)$ 中 $e^x$ 和 $x$ 的最高次幂最多比 $F_{m-1}(x)$ 中 $e^x$ 和 $x$ 的最高次幂大 $1$，即不超过 $m$。

在 $\sum_{a,b}f_{a,b}e^{ax}x^b$ 的形式下，乘 $e^x,x$ 以及求导都是简单的。至此，已经可以 $O(m^2)$ 地从 $F_{m-1}(x)$ 推出 $F_m(x)$ 的 $\sum_{a,b}f_{a,b}e^{ax}x^b$ 表示。

最后考虑算答案，设 $F_M(x)=\sum_{a,b}f_{a,b}e^{ax}x^b$。

$$
\begin{aligned}
&\sum_{i=0}^N\binom Nidp_{M,i}\\
=&\sum_{i=0}^N\binom Ni\left[\frac{x^i}{i!}\right]F_M(x)\\
=&\left[\frac{x^N}{N!}\right]e^xF_M(x)\\
=&N!\sum_{a=0}^M\sum_{b=0}^{\min(N,M)}f_{a,b}\frac{(a+1)^{N-b}}{(N-b)!}\\
=&\sum_{a=0}^M\sum_{b=0}^{\min(N,M)}f_{a,b}(a+1)^{N-b}N^{\underline b}
\end{aligned}
$$

计算答案复杂度 $O(M^2\log N)$ 或 $O(M^2+M\log N)$，总复杂度 $O(M^3)$。