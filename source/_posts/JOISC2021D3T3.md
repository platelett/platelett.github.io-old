---
title: Meetings 2 | JOISC2021 D3T3
date: 2021-03-22 20:16:23
updated: 2021-03-22 20:16:23
tags: [数据结构,树的重心,树的直径]
categories: JOISC
---
> 给定 $n$ 个点的树。
>
> 当一些结点上的人开会时，会将满足以下条件的结点作为会议的候选结点：所有参会者到此结点的距离和最小。
>
> 对于每一个的 $k \in [1,n]$，求出所有可能的 $k$ 个人的会议最多有多少个候选结点。
>
> $n \le 2 \cdot 10^5$

#### 这题告诉我们：涉及到 ```size``` 时可以考虑把重心作为树的重心有没有什么性质。

### 结论 $1$：当 $k$ 为奇数时，答案为 $1$。

证明：

设 $f(u)$ 表示删去 $u$ 后，每个连通块内关键结点（有参会者的结点）数量的最大值。

假设 $u$ 是一个候选结点，那么 $f(u) \le \lfloor \frac k2 \rfloor$ ，否则可以找到一个与 $u$ 相邻的结点 $v$ 满足所在连通块大小大于 $\lfloor \frac k2 \rfloor$ ，不难发现 $v$ 比 $u$ 严格更优。

因此任意一个不同于 $u$ 的结点 $v$ 必定满足 $f(v) \ge k -f(u) > \lfloor \frac k2 \rfloor$，所以 $v$ 一定不是候选结点。

### 结论 $2$：当 $k$ 为偶数时，所有候选结点一定是一条链。

 证明：

若 $u, v$ 是不相邻两个结点，且都是候选结点。设边 $(u,v)$ 两边的关键结点数量分别为 $a,b$。

则 $a + b = k, a \le f(v) \le \frac k2,b \le f(u) \le \frac k2 \Rightarrow a = b = \frac k2$。

所有平分 $k$ 个点的边 $(u,v)$ 一定是一条链，如图。

![](https://i.loli.net/2021/03/22/cs9ZJOrImCibEAz.png)

至此，可以得到一个 $O(n^2)$ 做法：

设 $F(u,v)$ 表示链 $u - v$ 两边的结点数的较小值。

枚举一条链 $u - v$，如果 $F(u,v) \ge \frac k2$，那么可以在两边分别放 $\frac k2$ 参会者，就至少能得到链长个候选结点。  
因此用链长更新 $k=2F(u,v)$ 时的答案。

然后最仙的想法就是把树的重心作为根，然后对于任意一条链 $u - v$，$F(u,v) = \min\{size_u,size_v\}$，证明略（主要是分析深度递增型链）。

从空集开始，把所有点按 $size$ 从大到小加入，动态维护当前树的直径 $D$。

若刚加入的点为 $u$，那么就用 $D$ 更新 $k=2size_u$ 时的答案。

动态维护树的直径的方法也很简单：设当前的直径的两个端点是 $a, b$，加入结点 $u$ 后，如果直径改变，则要么是 $u - a$，要么是 $u - b$，算两次树上距离即可。

代码：

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define mem(a, b) memset(a, b, sizeof a)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)

using namespace std;
const int N = 2e5 + 5;

int n, sz[N], rt, d[N], fa[18][N], ans[N];
vector <int> G[N], nds[N];
void dfs(int u, int fa) {
    sz[u] = 1;
    int ma = 0;
    for(int v : G[u]) if(v ^ fa)
        dfs(v, u), sz[u] += sz[v], ma = max(ma, sz[v]);
    if(max(ma, n - sz[u]) <= n / 2) rt = u;
}
void Dfs(int u) {
    sz[u] = 1;
    rep(i, 1, 17) fa[i][u] = fa[i - 1][fa[i - 1][u]];
    for(int v : G[u]) if(v ^ fa[0][u])
        fa[0][v] = u, d[v] = d[u] + 1, Dfs(v), sz[u] += sz[v];
}
int dis(int u, int v) {
    if(d[u] < d[v]) swap(u, v);
    int res = d[u] - d[v];
    per(i, 17, 0) if(d[u] - (1 << i) >= d[v]) u = fa[i][u];
    if(u == v) return res;
    per(i, 17, 0) if(fa[i][u] ^ fa[i][v])
        u = fa[i][u], v = fa[i][v], res += 2 << i;
    return res + 2;
}

int main() {
    cin >> n;
    int u, v;
    rep(i, 2, n) {
        scanf("%d%d", &u, &v);
        G[u].push_back(v), G[v].push_back(u);
    }
    dfs(1, 0), Dfs(rt);
    rep(i, 1, n) nds[sz[i]].push_back(i);
    u = v = rt;
    int D = 0, d;
    per(i, n, 1) {
        for(int x : nds[i]) {
            if((d = dis(x, u)) > D) v = x, D = d;
            if((d = dis(x, v)) > D) u = x, D = d;
        }
        ans[i] = D + 1;
    }
    rep(i, 1, n) printf("%d\n", i & 1 ? 1 : ans[i / 2]);
    return 0;
}
```