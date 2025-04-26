---
title: 部分题解合集
date: 2021-09-17 16:27:28
---
### URAL2118

> 给定前 $k$ 个字母的 $01$ 前缀编码（不存在一个编码是另一个编码的前缀）。
>
> 给定字符串 $s$，设其解码后的 $01$ 串为 $S$，求最多能将 $S$ 划分为多少段使得每一段都无法解码，无解输出 $-1$。
>
> $k \le 52, n\le 10^6$

首先考虑两种特殊情况：

- 编码中既有 $0$，也有 $1$，那 $S$ 无论怎么划分都可以解码。
- 编码中既没 $0$，也没 $1$，那么答案为 $|S|$。

剩下的情况为：有 $0$ 无 $1$ 和有 $1$ 无 $0$，由于对称性，只考虑有 $0$ 无 $1$。

首先答案的上界为 $1$ 的个数，因为全 $0$ 的一段是可以被解码的。

如果最后一位为 $1$，那么在每个 $1$ 后面断开，就可以取到这个上界。

如果最后一位为 $0$，假设最后一段 $T$，可以说明存在最优解满足 $T$ 前面是 $1$：假设 $T$ 前面有 **连续** 的 $x$ 个 $0$，前面总共有 $y$ 个 $1$。那么答案的上界为 $y+1$，把这 $x$ 个 $0$ 加入 $T$，然后在 $T$ 前面的每个 $1$ 后面断开就可以到达这个上界。

如果 $T$ 前面有 $y$ 个 $1$，最大段数就是 $y+1$，问题就是求最大的 $y$，假设 $T'$ 是最短的无法被解码的后缀，可以说明最大的 $y$ 等于 $T'$ 前面 $1$ 的个数：由于 $|T|\ge |T'|$，所以 $y$ 不会超过 $T'$ 前面 $1$ 的个数，另外，令 $T=T'前面极长的一段0+T'$，$y$ 就可以取到这个上界。

问题转化成了求 $T'$，那么 $T'$ 的任何前缀都无法解码，否则 $T'$ 不是最短的，记 $suffix(i)$ 表示 $S$ 长度为 $i$ 的后缀，问题就是依次判断 $suffix(1),suffix(2),suffix(3),\cdots$ 是否有前缀可以被解码，`AC` 自动机即可。

复杂度 $O(nk)$。

### Graph Subpaths

> 没有提交地址。
>
> 给定一张 $n$ 个点 $m$ 条边的有向无环图，再给定 $k$ 条路径，每条路径长度为 $l_i$，一条合法路径不包含这 $k$ 条路径。
>
> 对于 $i \in [2,n]$，求 $1\rightarrow i$ 的合法路径条数。
>
> $n,m,\sum_{i=1}^kl_i \le 10^5$

### Sol 1

首先把所有边反向，就转化成了求 $i\rightarrow 1$ 的合法路径条数。

对于边 $(u,v)$，标记它的权值为 $v$，设 $T_i$ 表示 $i\rightarrow 1$ 所有合法路径组成的 `trie`，考虑怎么按拓扑序求出每一个 $T_u$，对于边 $(u,v)$，$T_v$ 是已经求过了，把 $T_v$ 复制到根的儿子，但这样会有一些以 $u$ 为起点的不合法路径，需要删除这些路径：从 $T_u$ 的根出发沿着不合法路径走，把以终点为根的子树删除即可。$u$ 的答案就是 $T_u$ 的叶子个数。

由于 $T_u$ 非常大，当然不能直接存下来，可行的方法是用可持久化的 `trie`，对于边 $(u,v)$，只需要从 $T_u$ 的根向 $T_v$ 连一条边就行了，而不用复制整棵 $T_v$，删除子树也只需要对路径上的结点建新版本。

一个问题是儿子列表的维护，如果用链表维护的话，删除子树时新建的结点需要从原版本复制整个链表，复杂度可能达到 $O(n^2)$。

用主席树维护儿子列表就可以 $O(n\log n)$ 地新建结点。

复杂度 $O(n\log n)$。

### Sol 2

对所有路径建 `AC` 自动机，不难想到一个 $O(n^2)$ 状态的 `DP`，设 $f_{i,u}$ 表示在原图中走到 $i$，`AC` 自动机上走到 `u`，且没有经过 `AC` 自动机上终止结点的路径条数。

由于 `AC` 自动机上的点对应唯一原图中的点的，所以 `DP` 状态定义成 $f_u$ 就可以转移了。

由于建 `AC` 自动机需要主席树，复杂度 $O(n\log n)$。

### ZOJ3390

> 对于两棵树 $T_1,T_2$，定义 $T_1+T_2$ 表示把根合并，$T_1\cdot T_2$ 表示把 $T_1$ 中每个结点换成 $T_2$，$T_1=T_2$ 表示树同构。
>
> 给定树 $A,B,C$，求 $X,Y$ 满足 $AX+BY=C$。
>
> $|A|,|B|,|C| \le 10^5$

设 $height(T)$ 表示 $T$ 的树高（最深叶子到根的距离）。

可以发现 $height(T_1+T_2)=\max(height(T_1),height(T_2)),height(T_1\cdot T_2)=height(T_1)+height(T_2)$。

那么 $\max(height(A)+height(X),height(B)+height(Y))=height(C)$，假设 $height(A)+height(X)=height(C)$（另一种情况是一样的）。

这样就知道了 $height(X)$，考虑怎么求 $X$，很简单，假设 $u$ 为 $C$ 中最深叶子的 $height(X)$ 级祖先，以 $u$ 为根的子树就是 $X$，这样就知道了 $AX$，然后就可以求出 $BY$，再用类似的方法就可以求出 $Y$ 了。

虽然思路很简单，但是有一定实现难度，求 $BY$ 时树的“减法“，以及判定答案是否合法都需要用到树哈希。

树哈希公式：
$$
f_u=1+\sum_{v\in son(u)}f_v\cdot \text{prime}(size_v)
$$
其中 $\text{prime}(i)$ 表示第 $i$ 个质数。

**注：**对于两棵大小不同的树 $T_1,T_2$，$f_{T_1}=f_{T_2}$ 是可能的，因此哈希应当和子树大小捆绑在一起。

`Generator` ：参数 $T,A,B,C,D$ 可调，分别表示数据组数和树 $A,X,B,Y$ 的大小。

<details><summary><span style="font-size: large; font-weight: bold; color: rgb(33,150,243);">查看代码</span></summary>

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)
#define pb push_back
#define eb emplace_back
#define mp make_pair
#define fi first 
#define se second
#define all(x) (x).begin(), (x).end()
#define SZ(x) int((x).size())
#define mid ((l + r) / 2)
#define lc o * 2
#define rc o * 2 + 1
#define lch l, mid, lc
#define rch mid + 1, r, rc
#define cmi(a, b) (a = min(a, b))
#define cma(a, b) (a = max(a, b))
#define lb lower_bound
#define ub upper_bound
#define bs binary_search
#define pop __builtin_popcount
#define llpop __builtin_popcountll
#define ctz __builtin_ctz
#define llctz __builtin_ctzll
#define clz __builtin_clz
#define llclz __builtin_clzll
#define par __builtin_parity
#define llpar __builtin_parityll

using namespace std;
using ll = long long;
using lf = double;
// using P = pair<int, int>;
using V = vector<int>;
// using cmp = complex<lf>;

ll gen(ll x) {
    const ll k = 0x9ddfea08eb382d69ull;
    rep(i, 1, 3) x *= k, x ^= x >> 47;
    return x * k;
}
int rnd() {
    static ll s = time(0) + (ll)new char;
    return (s += gen(s)) & INT_MAX;
}
V tmul(V a, V b) {
    int n = a.size() - 1, m = b.size() - 1, su = n;
    rep(i, 1, n) {
        rep(j, 2, m) a.pb(b[j] == 1 ? i : b[j] + su - 1);
        su += m - 1;
    }
    return a;
}
V tplus(V a, V b) {
    int n = a.size() - 1, m = b.size() - 1;
    rep(j, 2, m) a.pb(b[j] == 1 ? 1 : b[j] + n - 1);
    return a;
}
const int N = 5, A = 5, B = 5, C = 5, D = 5;
int main() {
    const int T = 100000;
    printf("%d\n", T);
    rep(kase, 1, T) {
        int a = rnd() % A + 1, b = rnd() % B + 1, c = rnd() % C + 1, d = rnd() % D + 1;
        V t1{0}, t2{0}, t3{0}, t4{0};
        auto get = [](V& t, int n) { rep(i, 1, n) t.pb(i > 1 ? rnd() % (i - 1) + 1 : 0); };
        get(t1, a), get(t2, b), get(t3, c), get(t4, d);
        V t5 = tplus(tmul(t1, t2), tmul(t3, t4));
        printf("%d %d %llu\n", a, c, t5.size() - 1);
        auto prt = [](V& v) { For(i, 1, v.size()) printf("%d ", v[i]); puts(""); };
        prt(t1), prt(t3), prt(t5);
    }
}
```
</details>

`Special judge`：假设保存为 `checker.cpp`，编译后在命令行中使用：`checker <input-file> <output-file>`，答案正确返回值为 $0$，否则返回值为 $1$，输出为第一组出错的数据。

<details><summary><span style="font-size: large; font-weight: bold; color: rgb(33,150,243);">查看代码</span></summary>

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)
#define pb push_back
#define eb emplace_back
#define mp make_pair
#define fi first 
#define se second
#define all(x) (x).begin(), (x).end()
#define SZ(x) int((x).size())
#define mid ((l + r) / 2)
#define lc o * 2
#define rc o * 2 + 1
#define lch l, mid, lc
#define rch mid + 1, r, rc
#define cmi(a, b) (a = min(a, b))
#define cma(a, b) (a = max(a, b))
#define lb lower_bound
#define ub upper_bound
#define bs binary_search
#define pop __builtin_popcount
#define llpop __builtin_popcountll
#define ctz __builtin_ctz
#define llctz __builtin_ctzll
#define clz __builtin_clz
#define llclz __builtin_clzll
#define par __builtin_parity
#define llpar __builtin_parityll

using namespace std;
using ll = long long;
using lf = double;
using P = pair<int, int>;
using V = vector<int>;
// using cmp = complex<lf>;

const int N = 1e5 + 5, M = 1299709;
int f[M + 5], pid, prm[N];

V tmul(V a, V b) {
    int n = a.size() - 1, m = b.size() - 1, su = n;
    rep(i, 1, n) {
        rep(j, 2, m) a.pb(b[j] == 1 ? i : b[j] + su - 1);
        su += m - 1;
    }
    return a;
}
V tplus(V a, V b) {
    int n = a.size() - 1, m = b.size() - 1;
    rep(j, 2, m) a.pb(b[j] == 1 ? 1 : b[j] + n - 1);
    return a;
}

P dfs(int u, const V G[]) {
    P re(1, 1);
    for(int v : G[u]) {
        auto [h, sz] = dfs(v, G);
        re.fi = (re.fi + (ll)h * prm[sz]) % 999999937;
        re.se += sz;
    }
    return re;
}
int Hash(const V& t) {
    vector<int> G[N];
    For(i, 1, t.size()) G[t[i]].pb(i);
    return dfs(1, G).fi;
}
int read(FILE* f) { int x; return fscanf(f, "%d", &x) == 1 ? x : -1; }

int main(int argc, char* argv[]) {
    rep(i, 2, M) {
        if(!f[i]) prm[++pid] = i;
        for(int j = 1; i * prm[j] <= M; j++) {
            f[i * prm[j]] = 1;
            if(i % prm[j] == 0) break;
        }
    }
    FILE *in = fopen(argv[1], "r"), *out = fopen(argv[2], "r");
    for(int T = read(in); T--;) {
        V t1{0}, t2{0}, t3{0}, t4{0}, t5{0};
        auto get = [](FILE* f, V& t, int n) { rep(i, 1, n) t.pb(read(f)); };
        int a = read(in), b = read(in), c = read(in);
        get(in, t1, a), get(in, t3, b), get(in, t5, c);
        int d = read(out), e = read(out);
        auto err = [&]() {
            printf("1\n%d %d %d\n", a, b, c);
            rep(i, 1, a) printf("%d ", t1[i]); puts("");
            rep(i, 1, b) printf("%d ", t3[i]); puts("");
            rep(i, 1, c) printf("%d ", t5[i]);
            exit(1);
        };
        if(!~d) err();
        get(out, t2, d), get(out, t4, e);
        if((ll)a * d + (ll)b * e != c && Hash(tplus(tmul(t1, t2), tmul(t3, t4))) != Hash(t5)) err();
    }
}
```
</details>

### IZhO 2020 D1T3

> [题目链接](codeforces.com/group/Uo1lq8ZyWf/contest/265564)
>
> 给定长度为 $n$ 的序列 $A$，问有多少三元组 $(i,j,k)$ 满足 $i\le j<k$ 且 $[i,j]$ 和 $[j+1,k]$ 两个区间中数的集合相同。
>
> $n \le 2\cdot 10^5$

记 $prev_i$ 表示 $A_i$ 上一次出现的位置，$next_i$ 表示 $A_i$ 下一次出现的位置。

三元组 $(i,j,k)$ 合法的充要条件为：

- 区间 $[j+1,k]$ 中的数都在 $[i,j]$ 中出现，即 $i \le \min prev_{j+1\cdots i}$，相当于 $i$ 有个上界 $R_{j,k}=\min prev_{j+1\cdots k}$。
- 区间 $[i,j]$ 中的数都在 $[j+1,k]$ 中出现，即 $\max next_{i\cdots j}\le k$，相当于 $i$ 有个下界 $L_{j,k}$，其中 $L_{j,k}$ 是最小的 $i$ 满足 $\max next_{i\cdots j}\le k$。

那么 $(j,k)$ 的贡献就是 $\max(R_{j,k}-L_{j,k}+1,0)$，考虑只枚举 $k$，用数据结构动态维护 $\sum_{j=1}^{k-1}\max(R_{j,k}-L_{j,k}+1,0)$。

先考虑 $L_{j,k}$ 和 $R_{j,k}$ 分别怎么维护。

对于 $R_{j,k}=\min prev_{j+1\cdots k}$，这是 $prev$ 数组上的后缀 $\min$，当 $k\rightarrow k+1$ 时，$R_{j,k}$ 发生的改变是一个后缀变成了 $prev_{k+1}$，具体可以用单调栈求出这个后缀，然后区间赋值。

对于 $L_{j,k}$，直接分析 $k\rightarrow k+1$ 不太行，换一个角度考虑对于一个 $j$，$L_{j,k}$ 和 $k$ 的关系，$\max next_{i\cdots j}$ 是 $next$ 数组上的后缀 $\max$，将 $next_{1\cdots j}$ 依次插入单调栈，设单调栈中元素分别为 $next_{i_1},next_{i_2},next_{i_3},\cdots,next_{i_k}$：

- 当 $k\in[j+1,next_{i_k}-1]$ 时，$L_{j,k}=\infty$。
- 当 $k\in [next_{i_k},next_{i_{k-1}-1}]$，$L_{j,k}=i_{k-1}+1$。
- $\cdots$
- 当 $k\in [next_{i_1},n]$，$L_{j,k}=1$。

综上，单调栈元素 $i_x$ 意味着当 $k\in [next_{i_x},next_{i_{x-1}}-1]$，$L_{j,k}=i_{x-1}+1$，假设插入 $next_y$ 后 $next_{i_x}$ 被弹掉了，那么当 $j\in [i_x,y-1],k\in [next_{i_x},next_{i_{x-1}}-1]$ 时，$L_{j,k}=i_{x-1}+1$，相当于 $k$ 从 $next_{i_x}-1\rightarrow next_{i_x}$ 时，对 $L_{i_x\cdots y-1,k}$ 进行区间赋值为 
$i_x+1$。每个单调栈元素意味着一次区间赋值，所以只需要 $n$ 次区间赋值就可以维护 $L_{j,k}$。

但维护的是 $\sum_{j=1}^{k-1}\max(R_{j,k}-L_{j,k}+1,0)$，相当于夹在两条递增折线之间的面积。由于 $L_{j,k}$ 和 $R_{j,k}$ 都是随着 $k$ 增大而减小的，所以每次赋值都是减小。比如将 $R_{l\cdots r,k}$ 改为 $v$，如果 $v \le \min L_{l\cdots r,k}$，这一段的面积就是 $0$，如果 $v \ge \max L_{l\cdots r,k}$，这一段的面积就是 $\sum_{j=l}^rR_{j,k}-\sum_{j=l}^rL_{j,k}+(r-l+1)$，否则可以二分一个分界点 $x$，使得 $\forall j \in[l,x], v\ge L_{j,k},\forall j\in[x+1,r],v\le L_{j,k}$，分界点左右两段分别对应上述两种情况。所以只要维护了 $L_{j,k},R_{j,k}$ 的区间 $\min,\max$ 和 $sum$ 就可以维护 $\sum_{j=1}^{k-1}\max(R_{j,k}-L_{j,k}+1,0)$ 了。

在实现上并不需要求 $x$，假设当前要将 $L_{ql\cdots qr,k}$ 赋值为 $v$，当前线段树结点区间为 $[l,r]$，一般区间赋值是在 $ql\le l\land r \le qr$ 时停止递归，这里把条件改成 $ql\le l\land r \le qr \land (v \le \min L_{l\cdots r,k}\lor v \ge \max L_{l\cdots r,k})$ 才可以方便地维护 $\sum_{j=1}^{k-1}\max(R_{j,k}-L_{j,k}+1,0)$，由于只有一个分界点，所以复杂度不变。

复杂度 $O(n\log n)$，实现难度较大，附上代码：

<details><summary><span style="font-size: large; font-weight: bold; color: rgb(33,150,243);">查看代码</span></summary>

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)
#define pb push_back
#define eb emplace_back
#define mid ((l + r) / 2)
#define lc o * 2
#define rc o * 2 + 1
#define lch l, mid, lc
#define rch mid + 1, r, rc

using namespace std;
using ll = long long;

const int N = 2e5 + 5;

int n, a[N], pre[N], nxt[N], L[N];
vector<tuple<int, int, int>> vl[N], vr[N];
int tagL[N * 4], minL[N * 4], maxL[N * 4];
int tagR[N * 4], minR[N * 4], maxR[N * 4];
ll sum[N * 4], sumL[N * 4], sumR[N * 4], as;

void pushUp(int o) {
    sum[o] = sum[lc] + sum[rc];
    sumL[o] = sumL[lc] + sumL[rc];
    sumR[o] = sumR[lc] + sumR[rc];
    minL[o] = min(minL[lc], minL[rc]);
    minR[o] = min(minR[lc], minR[rc]);
    maxL[o] = max(maxL[lc], maxL[rc]);
    maxR[o] = max(maxR[lc], maxR[rc]);
}
void pushL(int v, int l, int r, int o) {
    tagL[o] = minL[o] = maxL[o] = v, sumL[o] = v * (r - l + 1ll);
    sum[o] = max(sumR[o] - sumL[o], 0ll);
}
void pushR(int v, int l, int r, int o) {
    tagR[o] = minR[o] = maxR[o] = v, sumR[o] = v * (r - l + 1ll);
    sum[o] = max(sumR[o] - sumL[o], 0ll);
}
void pushDown(int l, int r, int o) {
    if(~tagL[o]) pushL(tagL[o], lch), pushL(tagL[o], rch), tagL[o] = -1;
    if(~tagR[o]) pushR(tagR[o], lch), pushR(tagR[o], rch), tagR[o] = -1;
}
void updL(int L, int R, int v, int l, int r, int o) {
    if(L <= l && r <= R && (v <= minR[o] || v >= maxR[o])) return pushL(v, l, r, o);
    pushDown(l, r, o);
    if(L <= mid) updL(L, R, v, lch);
    if(R > mid) updL(L, R, v, rch);
    pushUp(o);
}
void updR(int L, int R, int v, int l, int r, int o) {
    if(L <= l && r <= R && (v >= maxL[o] || v <= minL[o])) return pushR(v, l, r, o);
    pushDown(l, r, o);
    if(L <= mid) updR(L, R, v, lch);
    if(R > mid) updR(L, R, v, rch);
    pushUp(o);
}
int main() {
    cin >> n;
    rep(i, 1, n) scanf("%d", &a[i]);
    rep(i, 1, n) L[i] = 0;
    rep(i, 1, n) pre[i] = L[a[i]], L[a[i]] = i;
    rep(i, 1, n) L[i] = n + 1;
    per(i, n, 1) nxt[i] = L[a[i]], L[a[i]] = i;
    rep(i, 2, n) {
        int& j = L[i] = i - 1;
        for(; j && pre[j] >= pre[i]; j = L[j]);
        vr[i].eb(max(j, 1), i - 1, pre[i]);
    }
    rep(i, 2, n) {
        int& j = L[i] = i - 1;
        for(; j && nxt[j] <= nxt[i]; j = L[j])
            vl[nxt[j]].eb(max(j, 1), i - 1, L[j]);
    }
    mem(tagL, 63), mem(tagR, -1);
    rep(i, 1, n) {
        for(auto [l, r, v] : vl[i]) updL(l, r, v, 1, n, 1);
        for(auto [l, r, v] : vr[i]) updR(l, r, v, 1, n, 1);
        as += sum[1];
    }
    cout << as;
}
```
</details>

### BaekjoonOJ18733

> 给定 $n$ 个数 $a_i$，要把它们放进 $k$ 个有标号的盒子中，要求每个盒子非空。
>
> 每个盒子的价值为盒内所有数的按位与，要最大化所有数的价值，同时求达到最大值的方案数。
>
> $k \le n \le 10^5,0\le a_i \le 10^9$

如果所有 $a_i=0$，那么方案数为 $k!{n\brace k}$。

假设所有 $a_i$ 在二进制下最高位为 $b$，可以证明最优的策略一定会最大化价值在第 $b$ 位为 $1$ 的盒子数量，分以下几种情况考虑：

- 所有 $a_i$ 在第 $b$ 位为 $1$，那么所有方案都会有 $k\cdot 2^k$ 的贡献，把所有 $a_i$ 减去 $2^b$ 继续做。
- 有 $x$ 个 $a_i$ 在第 $b$ 位为 $1$。如果 $x<k$，那么这 $x$ 个 $a_i$ 会一对一地放进 $x$ 个盒子中，其他的 $a_i$ 放进剩余的 $k-x$ 个盒子中，所有方案都会有 $x\cdot 2^k$ 的贡献，方案数乘上 $\binom kx$。
- 如果 $x\ge k$，那么其他的 $a_i$ 一定放在同一个盒子中，所有方案都会有 $(k-1)\cdot 2^k$ 的贡献，把第 $b$ 位为 $1$ 的 $a_i$ 减去 $2^b$，把所有第 $b$ 位为 $0$ 的 $a_i$ 替换成它们的按位与。

这样递归做下去，即可 $O(n\log V)$ 地求出答案。

### OpenCup

> [题目链接](https://official.contest.yandex.ru/opencupXIX/contest/11931/problems/D/?lang=en) [提交记录](https://official.contest.yandex.ru/opencupXIX/contest/11931/run-report/52963174/)
>
> 维护一个长度为 $2^p$ 的数组 $A$（下标从 $0$ 开始），支持五种操作：
>
> - $A_v$ 增加 $\delta$。
>
> - 查询 $\sum_{i=l}^rA_i$。
>
> - 给定 $k$，$\forall i \in [0,2^p)$，满足 $i\text{ and } k\ne i$，将 $A_{i\text{ and }k}$ 加上 $A_i$，然后令 $A_i=0$。
>
> - 给定 $k$，$\forall i \in [0,2^p)$，满足 $i\text{ or } k\ne i$，将 $A_{i\text{ or }k}$ 加上 $A_i$，然后令 $A_i=0$。
>
> - 给定 $k$，$\forall i \in [0,2^p)$，满足 $i\text{ xor } k\ne i$，将 $A_{i\text{ xor }k}$ 加上 $A_i$，然后令 $A_i=0$。
>
> $p\le 19,v,k< 2^p,A_i,\delta \le 10^9$

先分析反复进行后三种操作后对 $A$ 的影响是怎样的，对于一个 $A_i$，假如它最终加到了 $A_j$ 上（$A_i$ 不变认为 $i=j$），那么 $j$ 是把 $i$ 的某些位赋为 $0$，某些位赋为 $1$，某些位 $01$ 翻转得到的，不难用 $tag_0,tag_1,tag_f$ 三个状压标记来表示后三种操作的影响。

考虑用 `0-1 trie` 维护这个序列，每个结点维护 $tag_0(u),tag_1(u),tag_f(u)$ 三个懒标记，问题是怎么下放标记，假设当前要将结点 $u$ 上的标记下放。

首先需要将标记传给左右儿子，对于儿子 $v$，它按如下方式接受 $tag_0(u),tag_1(u),tag_f(u)$ 对它的影响：

- $tag_0(v)=tag_0(v)\cup tag_0(u),tag_1(v)=tag_1(v)\setminus tag_0(u),tag_f(v)=tag_f(v)\setminus tag_0(u)$。

- $tag_1(u)$ 类似。

- 设 $x=tag_0(v)\cap tag_f(u),y=tag_1(v)\cap tag_f(u)$，那么 $tag_0(v)=(tag_0(v)\setminus x)\cap y,tag_1(v)=(tag_1(v)\setminus y)\cap x,tag_f(v)=(tag_f(v)\cap tag_f(u))\setminus x\setminus y$。

用位运算可以做到 $O(1)$。

另外还要考虑标记对点 $u$ 自己的影响，设点 $u$ 对应第 $d$ 位，分三种情况：

- 如果 $tag_0(u)$ 包含第 $d$ 位，需要将 $u$ 的右子树合并到左子树去。
- 如果 $tag_1(u)$ 包含第 $d$ 位，需要将 $u$ 的左子树合并到右子树去。
- 如果 $tag_f(u)$ 包含第 $d$ 位，需要将 $u$ 的左右子树交换。

交换左右子树是容易的，将以 $v$ 为根的子树合并到以 $u$ 为根的子树可以像线段树合并一样做：

- 如果 $u,v$ 中有一个是空的，就返回。

- 否则下放 $u,v$ 的标记，然后分别合并 $u,v$ 的左右儿子，并删除 $v$。

由于第二种情况每发生一次就会删一个结点，所以复杂度均摊是 $O(p)$ 的。

对于单点加和区间查询，像普通线段树一样做就行了，注意由于子树合并会删除结点，所以单点加的时候如果遇到了不存在的结点时需要动态开点。

复杂度 $O(p2^p)$。

### BaekjoonOJ19394

> 给定一张 $n$ 个点 $m$ 条边的无向图，定义一个边的子集是好的当且仅当仅保留这些边时，每个点的度数为偶数。
>
> 问所有好的边集的边数平方和。
>
> $n,m\le 2\cdot 10^5$

下面称合法边集为欧拉子图。

首先欧拉子图的数量是多少？答案是 $2^{m-n+c}$，其中 $c$ 是连通块数。考虑求出图的一个生成森林，那么有 $m-n+c$ 条非树边，每条非树边和它两个端点的树上路径构成一个基环，从这些基环中选若干个异或起来就能得到一张欧拉子图，同时任意一张欧拉子图一定是若干基环异或起来的。

假设一张欧拉子图有 $x$ 条边，它的贡献 $x^2=2\binom x2+x$，$x$ 可以考虑每条边的贡献，$\binom x2$ 可以考虑每对边的贡献。

- 一条边 $e$ 的贡献。如果 $e$ 是割边，那么它没有贡献，否则贡献等于删掉它后的欧拉子图数量 $2^{m-1-n+c}$。
- 一对边 $e_1,e_2$ 的贡献。如果 $e_1,e_2$ 中有割边，那么没有贡献，否则贡献等于删掉它们后的欧拉子图数量，如果 $e_1,e_2$ 是割集，贡献为 $2^{m-1-n+c}$，否则为 $2^{m-2-n+c}$。

因此问题是数有多少对非割边是割集，用数据结构当然是可以做的，但有一个优秀做法。

判断 $k$ 条边是否组成割集有如下算法：

- 求出生成树，给每非树边随机一个 $[0,2^{64})$ 内的权值。
- 每条树边的权值等于所有覆盖它的非树边的权值的异或和，割边权值为 $0$。
- $k$ 条边组成割集当前仅当其中一些边的异或和为 $0$。

这个条件是割集的充分条件，有 $2^{k-64}$ 的概率误判成割集，但用来数两条边的割集是很难出错的，可以认为两条非割边组成割集当且仅当两条边权值相等。

直接统计有多少对边权值相等就行了，复杂度 $O(n\log n)$ 或 $O(n)$（哈希表）。

### OpenCup

> [题目链接](https://official.contest.yandex.ru/opencupXVIII/contest/5917/problems/E/?success=53051713&lang=en) [提交记录](https://official.contest.yandex.ru/opencupXVIII/contest/5917/run-report/53051713/)
>
> 给定字符串 $s,t$，每次可以删除 $s$ 中一个字符 $c$ 的第一次出现或最后一次出现，删除代价取决于删除字符的初始位置。
>
> $|s|,|t| \le 2\cdot 10^5$

先考虑一个 $O(n^2)$ 的 `DP`，设 $f_{i,j}$ 表示使 $s_{1\cdot i}$ 和 $t_{1\cdots j}$ 相等的最小代价，转移为：

- $f_{i,j}\leftarrow f_{i-1,j-1}(s_i=t_j)$。
- $f_{i,j}\leftarrow f_{i-1,j}(t_{1\cdots j} 或 t_{j+1\cdots |t|} 中没有字符s_i)$。

设 $first(c)$ 表示字符 $c$ 在 $t$ 中第一次出现的位置，$last(c)$ 表示最后一次出现的位置，那么转移二的条件可以改写为 $j<first(s_i)\lor j\ge last(s_i)$。

当 $j\ne first(s_i)\land j\ne last(s_i)$ 时两种转移只能选择一种，考虑从这里优化。

把所有的 $first(c),last(c)$ 标记为关键值，那么最多有 $52$ 个关键值，考虑只计算 $j$ 为关键值的 $f_{i,j}$，设 $j$ 之后的一个关键值为 $j'$，先从 $f_{*,j}$ 转移到 $f_{*,j'-1}$，在这一过程，$s$ 中一些字符必须进行转移一，其他字符必须进行转移二，因此一个 $f_{i,j}$ 最多能转移到一个的 $f_{i',j'-1}$，具体转移到哪一个 $i'$ 可以通过 KMP 算出，然后再从 $f_{*,j'-1}$ 转移到 $f_{*,j'}$

复杂度 $O(26n)$。

### OpenCup

> [题目链接](https://official.contest.yandex.ru/opencupXX/contest/17756/problems/J/) [提交记录](https://official.contest.yandex.ru/opencupXX/contest/17756/run-report/53639932/)
>
> 给定两个长度为 $n$ 的 01 串 $s,t$，可以对 $s$ 进行以下三种操作任意次：
>
> - 令 $s_i=0$，代价为 $t_0$。
> - 令 $s_i=1$，代价为 $t_1$。
> - 交换 $s_i,s_{i+1}$，代价为 $t_s$。
>
> $n \le 4\cdot 10^6$

可以把所有交换操作调整到最前面，即先用交换操作把 $s$ 变成 $s'$，再用修改操作把 $s'$ 变成 $t$。

考虑 $s$ 变成 $s'$ 的最小代价，假设 $s_i$ 交换后的位置为 $s'_{p_i}$，可以证明最优的 $p$ 满足：$p_{p_i}=i$。可以认为交换位置 $i,j$ 的代价是 $t_s|i-j|$，任何位置最多参与一次交换。

对于满足 $s_i=t_i$ 的 $i$，它们没有必要参与交换，删除这些位置后满足两个性质：

- 任何交换 $(i_1,j_1),(i_2,j_2)$ 都是不交叉的，即区间 $[i_1,j_1]$ 和区间 $[i_2,j_2]$ 不交叉。
- 对于任何交换 $(i,j)$，区间 $(i,j)$ 中所有位置都参与交换。

综上，对于所有的极大交换 $(i,j)$，不妨假设 $s_i=0,s_j=1$，把 $0$ 变成 '('，$1$ 变成 ')'，区间 $[i,j]$ 是一个合法括号串，并且 $i$ 和 $j$ 是匹配的。这意味着如果 $s_i$ 要和 $s_j(j<i)$ 交换，那么 $j$ 是唯一的（或者没有，当所有的 $[j,i]$ 都不是合法括号串时）。

然后就可以 `DP` 了，设 $f_i$ 表示把 $s[1\cdots i] \rightarrow t[1\cdots i]$ 的最小代价，转移为：

$$
f_i=\min(f_{i-1}+t_{1-s_i},f_{j-1}+\text{cost}(j,i))
$$

其中 $j$ 是和 $i$ 匹配的位置，$\text{cost}(j,i)$ 是区间 $[j,i]$ 中所有匹配的位置 $x,y$ 的 $t_s|x-y|$ 之和，可以用栈维护。

另一个正确的转移为：

$$
f_i=\min(f_{i-1}+t_{1-s_i},f_{j-1}+f_{i-1}-f_j)
$$

正确的原因是 $f_{j-1}+f_{i-1}-f_j \le f_{j-1}+\text{cost}(j,i)$，并且 $f_{j-1}+f_{i-1}-f_j$ 是一个合法方案的代价（那个方案的描述有点复杂，略）。

复杂度 $O(n)$。
