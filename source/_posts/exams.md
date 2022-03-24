---
title: 考试题
date: 2021-09-17 16:27:28
---
### AUOJ1760

> 有 $n$ 个物品，其中可能有一个次品，它的质量与其他物品有差异。你需要多次使用天平后回答谜题：是否存在次品？次品是偏轻还是偏重？称量时，在天平两边放相同数量的物品，以得知那边更重。
>
> 构造一个能够解决谜题且称量次数最少的**固定**称量方案。
>
> $n \le 10^6$

考虑什么样的称量方案能够解决谜题。

假设称量次数为 $m$，定义矩阵 $A$：
$$
A_{i,j}=
\begin{cases}
-1 &(第 j 次称量物品 i 在天平左边)\\
0 &(第 j 次称量物品 i 不在天平上)\\
1 &(第 j 次称量物品 i 在天平右边)
\end{cases}
$$
首先怎么判断有没有次品：如果有物品没上过天平，哪无论如何都不能判断，否则可以判断，不存在次品当且仅当每次天平都平衡。

得到条件一：$A_i\ne \{0,0,\cdots, 0\}$。

确定了有次品，怎么确定是哪一个：先考虑已知次品偏重时怎么确定，根据每次天平的倾斜情况，可以得到每次称量时次品在天平的哪一边或不在天平上，定义序列 $B$：
$$
B_i=
\begin{cases}
-1 &(第 i 次称量次品在天平左边)\\
0 &(第 i 次称量次品不在天平上)\\
1 &(第 i 次称量次品在天平右边)
\end{cases}
$$
然后看 $B$ 和 $A_?$ 相等就可以确定次品是哪一个。

但是并不知道次品偏重还是偏轻，上面说了假设次品偏重可以得到一个序列 $B$，类似地假设次品偏轻可以得到一个序列 $C$，并且满足 $C=-B$（元素对于互为相反数），可以解决谜题的条件是 $B$ 和 $C$ 不能同时和某个 $A_i$ 相等。

得到条件二：$\forall i \ne j, A_i\ne A_j \land A_i \ne -A_j$。

由于每次称量时两边放相同数量的物品。

得到条件三：$\forall j\in[1,m],\sum_{i=1}^nA_{i,j}=0$。

满足以上三个条件就已经合法了，考虑对于一个 $m$，哪些 $n$ 可以构造出 $A$ 矩阵。

首先在前两个条件的限制下，$n$ 最大能取得 $\frac{3^m-1}2$，再加上第三个限制，$n$ 还能不能取到 $\frac{3^m-1}2$？

答案是否定的，因为对于所有的方案，$\forall j\in[1,m],\sum_{i=1}^n|A_{i,j}|=3^{m-1}$。

> 引理，$m$ 次称量可以解决谜题当且仅当 $n\le \frac{3^m-3}2$。

这样就可以求出最小的 $m$ 了，怎么求方案？

当 $3|n$ 时有一个简单的构造：定义 $\text{next}(A)$ 表示把序列 $A$ 每一项循环移位（$-1\rightarrow 0,0\rightarrow 1,1\rightarrow -1$）后得到的序列，不难发现一个事实，$A,\text{next}(A),\text{next}(\text{next}(A))$ 这三个序列对应位置之和等于 $0$。

如：

```plain
1 0 -1 1
-1 1 0 -1
0 -1 1 0
```

除去 $\{-1,-1,\cdots,-1\}$ 和 $\{1,1,\cdots,1\}$，把每三个这样的序列分为一组，恰好可以分成 $\frac {3^m-3}3$ 组，其中两两组互相为相反序列，于是删去一 半这样的组，恰好可以得到一个合法的矩阵 $A$。

### 思路一

当 $3\not|n$ 时，感觉问题非常困难，考虑模拟退火：

- 用 $\sum_{j=1}^m|\sum_{i=1}^nA_{i,j}|$ 作为一个解的权值，问题就是求权值最小的解。
- 还是先分组，取若干个组和一个不完整的组作为初始解。
- 每次以以下方式之一生成一个新解，如果新解更优则接受新解，否则以很低的概率接受新解。
  - 选择一个 $i$，将 $A_i$ 替换成 $-A_i$。
  - 选择一个 $i$，选择一个 $B$ 满足 $B$ 和 $-B$ 都没有在 $A$ 中出现过，令 $A_i=B$。

效率还行，但不稳定，对于少数 $n$ 速度极慢，$n\le 10^4$ 是完全没有问题的。

<details><summary><span style="font-size: large; font-weight: bold; color: rgb(33,150,243);">�鿴����</span></summary>
```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)
#define pb push_back

using namespace std;
using ll = long long;
using lf = double;

ll gen(ll x) {
    const ll k = 0x9ddfea08eb382d69ull;
    rep(i, 1, 3) x *= k, x ^= x >> 47;
    return x * k;
}
int rnd() {
    static ll s = time(0);
    return (s += gen(s)) & INT_MAX;
}
void wrt(int x, int ed) {
    static streambuf* out = cout.rdbuf();
    #define pc out -> sputc
    static char c[11]; int sz = 0;
    do c[++sz] = x % 10, x /= 10; while(x);
    while(sz) pc(c[sz--] + 48);
    pc(ed);
}
int n, m = 2, t = 9, nw[15], idx, A[2500000][15], c[15];
void dfs(int i) {
    if(i > m) {
        rep(j, 2, 4) {
            idx++;
            rep(k, 1, m) A[idx][k] = (nw[k] + j) % 3 - 1;
            if(idx == t) break;
        }
        return;
    }
    rep(j, -1, 1) {
		nw[i] = j, dfs(i + 1);
		if(idx == t) break;
	}
}
int main() {
    ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
    cin >> n;
    while(t - 3 >> 1 < n) m++, t *= 3;
    wrt(m, 10);
    t = t - 1 >> 1, dfs(2);
    rep(i, 1, n) rep(j, 1, m) c[j] += A[i][j];
    int su = 0;
    rep(i, 1, m) su += abs(c[i]);
    while(su) {
        int i = rnd() % n + 1;
        if(rnd() % t < n) {
            int d = 0;
            rep(j, 1, m) d += abs(c[j] - 2 * A[i][j]);
            if(d < su || rnd() % n < 10) {
                su = d;
                rep(j, 1, m) c[j] -= 2 * A[i][j], A[i][j] *= -1;
            }
        } else {
            int k = rnd() % (t - n) + n + 1, d = 0;
            rep(j, 1, m) d += abs(c[j] - A[i][j] + A[k][j]);
            if(d < su || rnd() % n < 10) {
				su = d;
				rep(j, 1, m) c[j] -= A[i][j] - A[k][j], swap(A[i][j], A[k][j]);
			}
        }
    }
    rep(i, 1, n) {
        int v = 0;
        per(j, m, 1) v = v * 3 + A[i][j] + 1;
        wrt(v, 32);
    }
}
```
</details>

### 思路二

另外，还有一种优秀的乱搞做法。

- 先分组，取若干个组和一个不完整的组作为初始解，那么 $\forall j\in[1,m],|\sum_{i=1}^nA_{i,j}|\le 1$。
- 对于每个 $j$，如果 $\sum_{i=1}^nA_{i,j}=-1$，找到一个 $i$ 满足 $A_{i,j}<1$ 并且将 $A_{i,j}$ 增大一后仍然合法，然后将 $A_{i,j}$ 加一。

会有极个别 $n$ 求出的解不合法，$10^6$ 以内应该不会超过 $10$ 个，而且都比较小，取决于初始解（所以可以 AC）。

结合思路一可以构造出 $10^6$ 内的所有 $n$。

<details><summary><span style="font-size: large; font-weight: bold; color: rgb(33,150,243);">�鿴����</span></summary>
```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)
#define pb push_back

using namespace std;
using ll = long long;

void wrt(int x, int ed) {
    static streambuf* out = cout.rdbuf();
    #define pc out -> sputc
    static char c[11]; int sz = 0;
    do c[++sz] = x % 10, x /= 10; while(x);
    while(sz) pc(c[sz--] + 48);
    pc(ed);
}
int n, m = 2, t = 9, nw[15], idx, A[1000005], c[15], as;
bool vs[1600000];
void dfs(int i) {
    if(i) rep(j, -1, 1) { nw[i] = j, dfs(i - 1); if(idx == n) break; }
    else rep(j, 1, 3) {
        int x = 0;
        per(k, m, 1) x = x * 3 + (j + nw[k]) % 3;
        if(!x || vs[x] || idx == n) break;
        A[++idx] = x, vs[x] = vs[t - x] = 1;
        rep(k, 1, m) c[k] += (j + nw[k]) % 3 - 1;
    }
}
int main() {
    ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
    cin >> n;
    while(t - 3 >> 1 < n) m++, t *= 3;
    t--, vs[t / 2] = 1, wrt(m, 10), nw[m] = -1, dfs(m - 1);
    if(n % 3) {
        rep(i, 1, m) as += abs(c[i]);
        int th = 1;
        rep(j, 1, m) {
            rep(i, 1, n) {
                if(!c[j]) break;
                int v = A[i] / th % 3;
                auto Try = [&](int x) {
                    int nw = A[i] + x * th;
                    if(vs[nw]) return;
                    vs[A[i]] = vs[t - A[i]] = 0, vs[nw] = vs[t - nw] = 1;
                    as -= abs(c[j]), c[j] += x, v += x, as += abs(c[j]), A[i] = nw;
                };
                if(v < 2 && c[j] < 0) Try(1);
                if(v > 0 && c[j] > 0) Try(-1);
            }
            th *= 3;
        }
    }
    rep(i, 1, n) wrt(A[i], 32);
}
```
</details>

### 思路三

这是官方解法，说是爬山算法，但感觉比较微妙，因为稍微扰动一下初始解就会有极个别 $n$ 跑不出来，流程是这样的：

- 用 $\sum_{j=1}^m|\sum_{i=1}^nA_{i,j}|$ 作为一个解的权值，问题就是求权值最小的解。

- $A$ 矩阵初始为空，然按字典序**从大到小**枚举长度为 $m$，值域为 $\{-1,0,1\}$ 的序列 $B$，如果 $B$ 和 $-B$ 没有在 $A$ 出现过，就把 $B,\text{next}(B),\text{next}(\text{next}(B))$ 依次加入 $A$ 末尾，加入 $n$ 行时终止。那么 $\forall j\in[1,m],|\sum_{i=1}^nA_{i,j}|\le 1$。

- 对当前解重复进行如下修改，直到权值为 $0$：

  从小到大依次枚举 $j$ 和 $i$，然后进行以下操作：

  - 如果 $\sum_{i=1}^nA_{i,j}=-1\land A_{i,j}<1$，尝试让 $A_{i,j}$ 加一。
  - 如果 $\sum_{i=1}^nA_{i,j}=1$，尝试让 $A_{i,j}=0$。

实测能构造出 $10^6$ 内的所有 $n$。

<details><summary><span style="font-size: large; font-weight: bold; color: rgb(33,150,243);">�鿴����</span></summary>
```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)
#define pb push_back

using namespace std;
using ll = long long;

void wrt(int x, int ed) {
    static streambuf* out = cout.rdbuf();
    #define pc out -> sputc
    static char c[11]; int sz = 0;
    do c[++sz] = x % 10, x /= 10; while(x);
    while(sz) pc(c[sz--] + 48);
    pc(ed);
}
int n, m = 2, t = 9, nw[15], idx, A[797170], c[15], as;
bool vs[1600000];
void dfs(int i) {
    if(i) per(j, 1, -1) { nw[i] = j, dfs(i - 1); if(idx == n) break; }
    else rep(j, 1, 3) {
        int x = 0;
        per(k, m, 1) x = x * 3 + (j + nw[k]) % 3;
        if(x == t || vs[x] || idx == n) break;
        A[++idx] = x, vs[x] = vs[t - x] = 1;
        rep(k, 1, m) c[k] += (j + nw[k]) % 3 - 1;
    }
}
int main() {
    ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
    cin >> n;
    while(t - 3 >> 1 < n) m++, t *= 3;
    t--, vs[t / 2] = 1, wrt(m, 10), nw[m] = 1, dfs(m - 1);
    rep(i, 1, m) as += abs(c[i]);
    while(as) {
        int th = 1;
        rep(j, 1, m) {
            rep(i, 1, n) {
                if(!c[j]) break;
                int v = A[i] / th % 3;
                auto Try = [&](int x) {
                    int nw = A[i] + x * th;
                    if(vs[nw]) return;
                    vs[A[i]] = vs[t - A[i]] = 0, vs[nw] = vs[t - nw] = 1;
                    as -= abs(c[j]), c[j] += x, v += x, as += abs(c[j]), A[i] = nw;
                };
                if(v < 2 && c[j] < 0) Try(1);
                if(v > 0 && c[j] > 0) Try(-v);
            }
            th *= 3;
        }
    }
    rep(i, 1, n) wrt(A[i], 32);
}
```
</details>

### AUOJ1761

> 有一个长度为 $n$ 的序列 $A$ （下标从 $1$ 开始）和一个长度为 $m$ 的序列 $B$（下标从 $0$ 开始）。
>
> $A$ 初始全为 $0$，每一天 $A_i$ 会增加 $i$，在第 $j$ 天，如果 $A_i>B_{j\bmod m}$，则进行一次操作，令 $A_i=B_{j\bmod m}$。
>
> $q$ 次询问，每次询问前 $d_i$ 天总共会进行多少次操作。
>
> $n,m,q\le 3\cdot 10^5,d_i\le 3\cdot 10^{12},t_i\le 10^{18}$

由于 $d_i$ 很大，应该会用到操作的周期性，事实上对于每个 $A_i$，从第一次操作它开始周期就为 $m$。

> 引理：假设在第 $p$ 天操作了 $A_i$，那么它在第 $p+m$ 天又会被操作。

证明：第 $p-m$ 天 $A_i$ 小于等于第 $p$ 天的 $A_i=B_{p \bmod m}$，感性理解 $A_i$ 越小在 $m$ 天后越容易操作，第 $p$ 天操作了，第 $p+m$ 天肯定要操作。

设 $A_i$ 第一次操作在第 $p_i$ 天，那么在第 $p_i,p_i+m,p_i+2m,pi+3m,\cdots$  都会操作，于是周期就是 $m$。

不难发现 $p_i$ 是递减的，因为任意时刻 $A$ 序列都是递增的，每次操作的都是一段后缀。那么对于一次询问，进行过操作的 $A_i$ 是一段后缀，只要求出了 $p_i$ 就可以二分出这个后缀，下面考虑怎么求 $p_i$。

设 $pos$ 表示 $B$ 中最小元素的位置，对于每个 $i$ 可以直接算出 $A_i$ 第一次被 $B_{pos}$ 操作的时间 $T_i$，那么 $p_i\in[T_i-m+1,T_i]$，设 $p_i=T_i-m+k_i$，$k_i$ 就是最小的 $x$ 满足 $ix+(T_i-m)i>B_{pos+x}$，考虑在线段树上每个结点 $[l,r]$ 维护 $\min_{j=l}^rB_{pos+j}-ij$，每次只需要二分出最小的 $x$ 满足 $\min_{j=1}^x\{B_{pos+j}-ij\}<(T_i-m)i$，注意到式子是可以斜率优化的，随着 $i$ 的增大，对每个结点用一个单调队列维护下凸包。复杂度为 $O(n\log n)$。这样就求出了 $p_i$。

对于询问 $d$，每个 $A_i$ 的贡献可以分为若干个完整周期和一个不完整周期。对于完整周期，由于周期长度为 $m$，可以用线段树直接模拟周期中每一天的修改，然后就知道每个 $A_i$ 一个周期被清理多少次。对于不完整周期，可以把所有询问离线下来，然后用一个线段树模拟这个不完整周期，大致思路是在 $p_i\bmod m$ 天插入 $A_i$，在 $d\bmod m$ 天进行查询，但有可能 $d\bmod m<p_i\bmod m$，所以实现上需要分类讨论一下。

### CatOJ1009

> 定义一个正整数是好的当且仅当它的十进制表示是它的二进制表示的后缀。
>
> 求第 $k$ 小的好数。
>
> $k\le 10^4$

考虑怎么判断一个 $n$ 位数 $x$ 是否是好的，首先它的十进制表示只能有 $01$，其次 $x\bmod 2^n$ 的二进制表示要和 $x$ 的十进制表示相同。

第 $10^4$ 个好数是 $163$ 位数，可以 `DFS` 出所有位数不超过 $163$ 的好数，从低位往高位枚举每一个 $1$ 在哪一位，维护 $x\bmod 2^{163}$，假设当前最高位为 $n$，如果 $x\bmod 2^n$ 的二进制表示和 $x$ 的十进制表示不同就结束。由于位数不超过 $163$ 的好数不多，复杂度可以接受。

### CatOJ1011

> 一种代码由 `+,-,[,]` 四种字符组成，还有一个初始为 $0$ 的存储单元。
>
> 规则为从左到右执行代码的每一个字符：
>
> - `+`：存储单元加一。
> - `-`：存储单元减一。
> - `[`：如果存储单元为 $0$，跳转到与之匹配的 `]`。
> - `]`：如果存储单元不为 $0$，跳转到与之匹配的 `[` 。
>
> 求出长度为 $n$ 的合法代码（括号序列合法）中，有多少个会在有限步内结束。
>
> $n \le 100$

先分析一下合法代码的性质：走出一个括号后，储存单元的值必然是 $0$，对于任意一段代码和一个输入 $x$ （输入为开始时储存单元的值，输出为结束后存储单元的值），如果有输出（即能跑完），那么其输出 $y$ 只有两种形式：

- 在代码中只有 `+-` 时 $y=x+d$。
- 在代码中有括号时 $y=d$。

首先预处理 $g_{i,j}$ 表示长度为 $i$ 的代码中，形式为 $y=x+j$ 的代码数，$h_i$ 表示长度为 $i$ 的合法代码数。

根据这个性质，设计 `DP` 状态 $f_{i,j}$ 表示长度为 $i$ 的代码中，输入为 $j$，输出为 $0$ 的代码数，转移为：

- 第一位为 `+-`，从 $f_{i-1,j\pm 1}$ 转移来。
- 第一位为 `[`，枚举第一对括号中代码的长度 $k$，分两种情况：

  - $j=0$，此时会直接跳过第一对括号，括号内的方案数位 $h_k$。
  - $j\ne 0$。

    - 括号内代码形式为 $y=0$ 或 $y=x-j$，方案数为 $f_{k,j}$。
    - 括号内代码形式为 $y=x+d$，其中 $d$ 为 $j$ 的真因数，且与 $j$ 的符号相反，方案数为 $g_{k,d}$。

  - 括号外方案数为 $f_{i-k-2,0}$。

统计答案时需要计算 $F_i$ 表示长度为 $i$ 的代码中，输入为 $0$，输出为 $0$，且以 `]` 结尾的代码数。

$$
F_i=f_{i,0}-\sum_{j=0}^{i-1}F_jg_{i-j,0}
$$

答案为

$$
\sum_{i=0}^nF_i2^{n-i}
$$

由于 `+-` 的对称性，$f_{i,j}=f_{i,-j},g_{i,j}=g_{i,-j}$，实现上并不需要存下标为负数的状态。

复杂度 $O(n^3\log n)$。

### CatOJ1062

> 在一个 $n\times m$ 的矩形中，一个点初始位于 $(r,c)$，每一次等概率地朝上下左右四个方向移动一 步，如果移出了矩形它将消失。
>
> 求 $S$ 步后点没有消失的概率，模 $998244353$。
>
> $n,m,S\le 5\cdot 10^5$

对始终不走出矩阵的方案数计数，最后再除以 $4^S$。

设这 $S$ 步中有 $s$ 步是左右移动，另外 $S-s$ 步是上下移动，需要分别求出 $s$ 步左右移动不出界的方案数和 $S-s$ 步上下移动不出界的方案数。
