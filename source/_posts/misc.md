---
title: 杂项
date: 2021-08-23 23:35:20
updated: 2021-08-23 23:35:20
tags: []
categories: 算法
---
### Barrett reduction 取模优化

> 计算 $ab\bmod m(0\le a,b<m<2^{31})$。

如果 $m$ 在编译时已知，那么编译器就会帮你完成这个优化（准确地说是下面的变种），否则你可以自己实现。

#### 原生算法

首先，设 $m^{-1}$ 表示浮点数形式的 $\frac 1m$。
$$
ab\bmod m=ab-\lfloor ab\cdot m^{-1}\rfloor m
$$
即使预处理了 $m^{-1}$，由于浮点数乘法比整数除法快不了多少，考虑进一步优化，取 $\frac {m'}{2^k}\approx m^{-1}$，然后
$$
\lfloor ab\cdot m^{-1}\rfloor \approx \lfloor \frac{abm'}{2^k}\rfloor
$$
计算右边就快多了，因为只有整数乘法和右移，如何合理选取 $m'$ 和 $k$？

令 $k=2\lceil \log_2 m\rceil,m'=\lceil \frac{2^k}m\rceil$，下面分析误差：

- 设 $m'm=2^k+r(0\le r<m)$
- 设 $ab=cm+d(0\le c,d<m)$
- $abm'=(cm+d)m'=cm'm+dm'=c2^k+cr+dm'$
- $cr+dm'<(m-1)^2+m'm<(m-1)^2+2^k+(m-1)=2^k+m(m-1)<2^k\cdot 2$
- $\lfloor \frac{abm'}{2^k}\rfloor=c\ \text{or}\ c+1$

设真实答案为 $ans$，那么 $ab-\lfloor \frac{abm'}{2^k}\rfloor m=ans\ \text{or}\ ans-m$。

更一般的结论是只要 $c \ge \max(a, b^2)$，$\lfloor \frac{a\lfloor \frac cb\rfloor}c\rfloor=\lfloor \frac ab\rfloor\ \text{or}\ \lfloor \frac ab\rfloor+1$ 成立。

实践中可以令 $k=64$，可以用 `~0ull / m + 1` 来计算 $m'$。


<details><summary><span style="font-size: large; font-weight: bold; color: rgb(33,150,243);">查看代码</span></summary>

```cpp
using uint = unsigned int;
using ull = unsigned long long;
struct barrett {
    uint m; ull im;
    barrett(uint m) :m(m), im(~0ull / m + 1) {}
    uint mul(uint a, uint b) const {
        ull z = (ull)a * b;
        uint v = z - uint((__uint128_t)z * im >> 64) * m;
        return m <= v ? v + m : v;
    }
};
```
</details>

速度比较：先在 $[0, m)$ 中随机生成 $10^5$ 个数，然后计算两两乘积模 $m$ 的异或和。

每个格子前者为直接取模的时间，后者为 `Barrett reduction` 取模的时间。

| 模数 |  998244353  | 1000000007  | 1000000009  |  19260817   |
| :-----------: | :---------: | :---------: | :---------: | :---------: |
|      -O2      | 35.8s/6.4s  | 35.2s/6.1s  | 35.0s/6.3s  | 35.4s/6.3s  |

#### 无误差除法的变种

$k$ 取得越大，$\frac {m'}{2^k}$ 和 $m^{-1}$ 就越接近，事实上，当 $k$ 足够大时，在一定范围内的 $a/m$ 都可以准确地计算。

$m'$ 有两种选择，$\lceil \frac{2^k}m\rceil$ 和 $\lfloor \frac{2^k}m\rfloor$，其中后者是不起作用的，因为无论 $k$ 取多大，$6/3$ 总是会算错。

假设 $a \in [0, A]$，分析 $k$ 需要多大 $a/m$ 才能准确计算。

$$
\begin{aligned}
2^k/m \le m' < 2^k/m+1\\\\
a/m-1<\lfloor\frac{am'}{2^k}\rfloor<a/m+a/2^k\\\\
a/m-1<\lfloor\frac{am'}{2^k}\rfloor<a/m+A/2^k
\end{aligned}
$$

如果在 $(a/m-1,a/m+A/2^k)$ 中只有一个整数，那么结果就是准确的。最坏情况是什么？该如何挑选 $a$，使区间中包含两个整数？

首先可以发现 $a$ 取 $m$ 的倍数是不起作用的，因为这个区间是开区间，因此 $a/m$ 应当小于一个整数并尽可能接近这个整数，即 $a\bmod m=m-1$，这时它已经包含了 $\lceil a/m-1\rceil$，因此它不能包含 $\lceil a/m\rceil$，即

$$a/m+A/2^k\le \lceil a/m\rceil$$

因为我们有 $a\bmod m=m-1$，所以 $\lceil a/m\rceil=a/m+1/m$，然后

$$A/2^k \le 1/m\Rightarrow 2^k\ge Am$$

在乘法取模的时候，$A=(m-1)^2$，所以 $2^k$ 会达到 $m^3$ 级别，比如 $m=998244353$ 时，应当取 $k=90$。

#### 速度更快的 Lemire reduction

~~其实已经咕了。~~

另外，笔者想到几个可以应用的地方。

- 高精度计算 $ab\bmod m$，高精度乘法是可以用 `FFT` 优化到 $O(\log m\log\log m)$ 的（粗略分析，不考虑位数相当大时用的 `Schönhage-Strassen` 算法)，而取模只能 $O(\log^2 m)$。当 $m$ 固定时，可以事先 $O(\log^2 m)$ 计算出 $m'$，然后就可以 $O(\log m\log\log m)$ 地取模了。将此应用于 `Miller Rabin` 素性检验可以将复杂度优化到 $O(k\log^2n\log\log n)$。事实上 `Miller Rabin` 在实践中一般用另一种取模算法 montgomery reducing 速度更快。

- 加速计算 $\gcd(a,b)$。比如 $a,b$ 都比较小，但求 $\gcd$ 的次数很多。事实上基于值域预处理的 $O(1)$ GCD 速度更快。

- 优化一些瓶颈为除法的算法，比如 min25 筛和 Powerful Number 筛。

### 卡特兰数的两个扩展

本节所有的路径都是指每一步向上或向右走一个单位长度的路径。

> 给定一条直线 $l:y=kx+b$，求从 $(0,0)$ 走到 $(a, l(a))$ 且不越过 $l$ 的路径条数。
>
> $k,b,a$ 都是正整数。

记 $G(k,b,a)$ 表示这个路径条数。

考虑从 $(0,0)$ 走到 $(a, l(a))$ 的路径条数，它可能不会越过 $l$，这部分路径条数为 $G(k,b,a)$，否则可以枚举路径第一次越过 $l$ 时的横坐标 $i$，得到 $(1)$ 式：
$$
\binom {l(a)+a}a=G(k,b,a)+\sum_{i+j=a}G(k,b,i)\binom{kj+j-1}j
$$

考虑从 $(0,0)$ 走到 $(a-1,l(a))$ 的路径条数，它一定会越过 $l$，枚举路径第一次越过 $l$ 时的横坐标 $i$，得到 $(2)$ 式：
$$
\binom {l(a)+a}{a-1}=\sum_{i+j=a}G(k,b,i)\binom {kj+j-1}{j-1}
$$

注意到
$$
\frac{\binom{kj+j-1}j}{\binom{kj+j-1}{j-1}}=\frac{kj}j=k
$$
$(1)-(2)\times k$ 得：
$$
G(k,b,a)=\binom{l(a)+a}a-k\binom{l(a)+a}{a-1}
$$

>给定一条直线 $l:y=kx+b$，求从 $(0,0)$ 走到 $(x,y)$ 的所有路径与 $l$ 的交点总数。
>
>$k,b,a$ 都是正整数，$l(x) \le y$。

记 $F(k,b,x,y)$ 表示这个交点总数。

考虑每个交点的贡献，枚举交点的横坐标 $i$，得到：
$$
F(k,b,x,y)=\sum_{i+j=x}\binom{l(i)+i}i\binom{y-l(i)+j}j
$$
进一步
$$
\begin{aligned}
&F(k,b,x,y)-kF(k,b,x-1,y+1)\\\\
&=\sum_{i+j=x}\binom{l(i)+i}i\binom{y-l(i)+j}j-k\binom{l(i)+i}i\binom{y-l(i)+j}{j-1}\\\\
&=\sum_{i+j=x}\binom{l(i)+i}i\left(\binom{y-l(i)+j}j-k\binom{y-l(i)+j}{j-1}\right)\\\\
&=\sum_{i+j=x}\binom{l(i)+i}iG(k,y-l(x),j)
\end{aligned}
$$
注意到 $\binom{l(i)+i}i$ 是从 $(0,0)$ 走到 $(i,l(i))$ 的路径条数，而 $G(k,y-l(x),j)$ 是从 $(x,y+1)$ **倒着**走到 $(i,l(i)+1)$ 且不越过 $l$ 的路径条数，整体求和就是从 $(0,0)$ 走到 $(x,y+1)$ 的路径条数，因为 $i$  就是在枚举最后一次越过 $l$ 时的横坐标。

于是
$$
F(k,b,x,y)=kF(k,b,x-1,y+1)+\binom{x+y+1}x
$$
因为 $F(k,b,x+y,0)=1$

归纳得到
$$
F(k,b,x,y)=\sum_{i+j=x}\binom{x+y+1}ik^j
$$
它甚至和 $b$ 无关。



### Lucas–Lehmer 素性检验

> 原理：$M_p=2^p-1(p\in\text{Odd prime})$ 是质数当且仅当 $M_p|(2+\sqrt 3)^{2^{p-2}}+(2-\sqrt 3)^{2^{p-2}}$。

后面有证明。

算法：令 $s_n=(2+\sqrt 3)^{2^n}+(2-\sqrt 3)^{2^n}\bmod M_p$，那么
$$
s_n=\begin{cases}
4&(i=0)\\\\
s_{n-1}^2-2\bmod M_p&(i>1)
\end{cases}
$$
计算平方可以用 `FFT` 做到 $O(p\log p)$，由于模数很特殊，取模可以做到线性：

$a \bmod M_p=(a_0\cdot 2^p+a_1)\bmod M_p=(a_0+a_1)\bmod M_p$

$s_{n-1}^2-2$ 最多展开两遍就行了，复杂度 $O(p^2\log p)$。

#### 充分性

原文：J. W. Bruce. A Really Trivial Proof of the Lucas-Lehmer Test. The American Mathematical Monthly, Vol.100, No.4, pp.370–371. April 1993.

反证法：假设 $M_p$ 是合数，它的最小质因子为 $q$，定义一个 $q^2$ 个元素的集合 $X=\{a+b\sqrt 3|a,b\in\mathbb Z_q\}$，其中 $\mathbb Z_q$ 表示 $0,1,2\cdots,q-2,q-1$，定义 $X$ 中的乘法运算为
$$
(a+b\sqrt 3)(c+d\sqrt 3)=(ac+3bd)\bmod q+(bc+ad)\bmod q\sqrt3
$$
$X$ 中任何两个元素的乘积一定也在 $X$ 内，但它不是群，因为不是所有元素都可逆，只保留有逆的元素，就得到了群 $X^*$。因为 $(2+\sqrt 3)(2+(q-1)\sqrt 3)\equiv 1\pmod q$，所以 $2+\sqrt 3\in X^{*}$。

考虑
$$
(2+\sqrt 3)^{2^{p-2}}+(2-\sqrt 3)^{2^{p-2}} \equiv 0\pmod {M_p}\\\\
(2+\sqrt 3)^{2^{p-1}}+[(2+\sqrt 3)(2-\sqrt 3)]^{2^{p-2}} \equiv 0\pmod {M_p}\\\\
(2+\sqrt 3)^{2^{p-1}}\equiv -1\pmod {M_p}
$$
这说明了 $2 + \sqrt 3$ 的阶为 $2^p$，即 $(2 + \sqrt 3)^0,(2 + \sqrt 3)^1,(2 + \sqrt 3)^2,\cdots,(2+\sqrt 3)^{2^p-1}$ 两两不同。

于是有 $2^p \le |X*| \le q^2 \le M_p$，得出矛盾 $2^p \le M_p$，故 $M_p$ 为质数。

#### 必要性

原文：Öystein J. R. Ödseth. A note on primality tests for N = h · 2n − 1. Department of Mathematics, University of Bergen.

现在已知 $M_p$ 是质数，要说明 $M_p|(2+\sqrt 3)^{2^{p-2}}+(2-\sqrt 3)^{2^{p-2}}$。

可以归纳证明，对于奇数 $p\ge 3$，有 $2^p-1\equiv 7\pmod 8$ 和 $2^p-1\equiv 7\pmod {12}$。

对于奇质数 $p$：

- $2^p-1\equiv 7\pmod 8\Rightarrow \left(\dfrac 2p\right)=1\Rightarrow 2^{\frac{M_p-1}2}\equiv 1\pmod {M_p}$
- $2^p-1\equiv 7\pmod {12}\Rightarrow \left(\dfrac 3p\right)=-1\Rightarrow 3^{\frac{M_p-1}2}\equiv -1\pmod {M_p}$

第一步的原理都是二次互反律及其补充，第二步的原理都是欧拉准则。

像前面一样，定义 $X=\{a+b\sqrt 3|a,b\in\mathbb Z_{M_p}\}$，保留有逆的元素得到群 $X^*$。

> 引理：$\forall x,y\in X^*,(x+y)^{M_p}\equiv x^{M_p}+y^{M_p}\pmod{M_p}$。

根据 $M_p|\binom{M_p}i$。

那么，我们有：
$$
\begin{aligned}
(6+2\sqrt 3)^{M_p}&=6^{M_p}+2^{M_p}(\sqrt 3)^{M_p}\\\\
&=6+2\sqrt 3\cdot 3^{\frac{M_p-1}2}\\\\
&=6-2\sqrt 3
\end{aligned}
$$
由于 $2+\sqrt 3=\frac{(6+2\sqrt 3)^2}{24}$，进一步
$$
\begin{aligned}
(2+\sqrt 3)^{\frac{M_p+1}2}&=\frac{(6+2\sqrt 3)^{M_p+1}}{24^{\frac{M_p+1}2}}\\\\
&=\frac{(6+2\sqrt 3)(6-2\sqrt 3)}{24\cdot (2^{\frac{M_p-1}2})^3\cdot 3^{\frac{M_p-1}2}}\\\\
&=-1
\end{aligned}
$$
最后，在两边同乘 $(2-\sqrt 3)^{\frac{M_p+1}4}$，并利用 $(2+\sqrt 3)(2-\sqrt 3)=1$
$$
\begin{aligned}
(2+\sqrt 3)^{\frac{M_p+1}2}(2-\sqrt 3)^{\frac{M_p+1}4}&=-(2-\sqrt 3)^{\frac{M_p+1}4}\\\\
(2+\sqrt 3)^{\frac{M_p+1}4}+(2-\sqrt 3)^{\frac{M_p+1}4}&=0\\\\
(2+\sqrt 3)^{2^{p-2}}+(2-\sqrt 3)^{2^{p-2}}&=0
\end{aligned}
$$

### 关于前 $k$ 个正奇数乘积模 $2^m$

> 求 $2^n$ 内所有奇数的乘积 $\bmod 2^n$。
>
> $n \le 10^4$

设 $f_n=\prod_{i=1}^{2^{n-1}}2i-1\bmod 2^n$，对于 $n\ge 2$，有
$$
\begin{aligned}
f_{n+1}&=(\prod_{i=1}^{2^{n-1}}2i-1\bmod 2^{n+1})(\prod_{i=1}^{2^{n-1}}2i-1+2^n\bmod 2^{n+1})\bmod 2^{n+1}\\\\
&=(\prod_{i=1}^{2^{n-1}}2i-1\bmod 2^{n+1})^2\bmod 2^{n+1}\\\\
\end{aligned}
$$
设 $\prod_{i=1}^{2^{n-1}}2i-1\bmod 2^{n+1}=f_n+k2^n(k\in\{0,1\})$，继续推。
$$
\begin{aligned}
f_{n+1}&=(f_n+k2^n)^2\bmod 2^{n+1}\\\\
&=f_n^2+k2^{n+1}+k^22^{2n}\bmod 2^{n+1}\\\\
&=f_n^2\bmod 2^{n+1}
\end{aligned}
$$
计算平方可以用 `FFT` 做到 $O(n\log n)$，总复杂度 $O(n^2\log n)$。

### 一个非常困难的问题

> 求 $\frac {p-1}2!\bmod p$，$p$ 是质数且 $p\equiv 3\pmod 4$。
>
> $p \le ???$

快速阶乘算法 $O(\sqrt p\log p)$。

其他做法：根据 Wilson 定理，$(p-1)!\equiv -1\pmod p$，那么 $\frac {p-1}2!\equiv \pm 1\pmod p$。准确地，$\frac {p-1}2!\equiv 1\pmod p$ 当且仅当以下之一成立：

- $p\equiv 3\pmod 8 \land h(-p)\equiv 1\pmod 4$

- $p\equiv 7\pmod 8\land h(-p)\equiv 3\pmod 4$

$h(-p) \bmod 4$ 怎么快速算？不会。
