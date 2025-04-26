---
title: 一道题
date: 2021-03-13 22:05:04
updated: 2021-03-13 22:05:04
tags: [数据结构]
categories: 考试
---
> 给定一棵 $n$ 个节点的树，每个节点 $u$ 上有一个可容纳 $k_u$ 个球的桶，$m$ 次操作：给一个节点到根路径上的所有节点的桶内放一个**有颜色的**球，如果节点的桶满了则不能放进这个节点，$q$ 次询问某节点的桶内的球有多少种颜色。
>
> $n, m, q  \le 10^5$

因为节点上的桶有容量，所以只有某时刻前的操作会对该节点产生影响。

记 $Max_u$ 表示$u$号节点上的桶**恰好**装满的时刻，考虑求出每个节点的 $Max$，将所有询问以 $Max$ 为关键字升序排序，问题就转化为单点加球，子树查询颜色数的问题。

- 求 $Max$：

  $Max_u$ 按照定义是所有覆盖 $u$ 节点的操作的时间第 $k_u$ 小，而所有覆盖 $u$ 节点的操作是下端点在 $u$ 子树内的操作，如果能将这些操作放到同一区间上，就可以用主席数实现区间第 $k$ 小。

  如果对每个点都至多有一次操作以它为下端点，将这个节点的 dfs 序作为对应操作的下标，操作的时间为值，问题就转化成了区间第 $k$ 小。

  但对每个点都可能作为多次操作的下端点，需要将每个结点扩充成一个区间，区间长度为这个节点对应操作的数量，然后就可以类别上一段所说的方法求出 $Max$。

  ```cpp
  rep(i, 1, m) {
      scanf("%d%d", &x[i], &c[i]);
      tim[dfn[x[i]]].push_back(i);
  }
  rep(i, 1, n) {
      R[i] = R[i-1]; //R[i] 是 dfs 序为 i 的节点对应区间的右端点
      for(int t : tim[i]) SegTree::upd(t, R[i]), R[i]++;
  }
  rep(i, 1, n) Max[i] = SegTree::qry(k[i], R[dfn[i]-1], R[suf[i]]);
  ```

  

- 单点加球，子树查询颜色数的问题：

  考虑静态查询子树查询颜色数的问题，做法是每个有颜色的结点处 $+1$，每个结点和同种颜色， ```dfs``` 序小于这个结点且 ```dfs``` 序最大的结点的 ```lca``` 处 $-1$。

  如图

    ![.jpg](https://i.loli.net/2020/10/24/8mOxgRBsJMpzLI1.jpg)

  在 $4，5$ 号结点的 ```lca```，$1$ 号结点处 $-1$。

  在 $5，6$ 号结点的 ```lca```，$1$ 号结点处 $-1$。

  ...

  然后查询子树和就是子树颜色数。

  再考虑单点加球，子树查询颜色数的问题。其实只需要加一个 ```set``` 数组和树状数组，```set``` 数组维护当前每种颜色结点的 ```dfs``` 序，单点加球时在 ```set``` 中找出前驱和后继，这个点处 $+1$，它和前驱的 ```lca``` 处 $-1$，和后继的 ```lca``` 处 $-1$，前驱和后继的 ```lca``` 处 $+1$，然后查询子树和。

  ```cpp
  int ci = c[i], dfnx = dfn[x[i]];
  if(s[ci].count(dfnx)) return;
  s[ci].insert(dfnx), add(dfnx, 1);
  auto it = s[ci].find(dfnx);
  int pre = 0, suf = 0;
  if(it != s[ci].begin()) pre = *(--it), it++;
  if(it != --s[ci].end()) suf = *(++it), it--;
  if(pre) upd(pre, dfnx, -1);
  if(suf) upd(suf, dfnx, -1);
  if(pre && suf) upd(pre, suf, 1);
  ```

时间复杂度 $O(n\log n)$。

完整代码：

```cpp
#include <bits/stdc++.h>
#define rep(i, l, r) for(int i = (l); i <= (r); i++)
#define per(i, r, l) for(int i = (r); i >= (l); i--)
#define For(i, l, r) for(int i = (l), i##e = (r); i < i##e; i++)

using namespace std;
const int N = 1e5 + 5;
int n, m, q, k[N];

namespace Seg {
    int sz, T[N], c[N*20], ls[N*20], rs[N*20];
    #define mid ((l + r) >> 1)
    #define lch l, mid, ls[o]
    #define rch mid + 1, r, rs[o]
    void upd(int oo, int p, int l, int r, int& o) {
        c[o = ++sz] = c[oo] + 1, ls[o] = ls[oo], rs[o] = rs[oo];
        if(p > mid) upd(rs[oo], p, rch);
        else if(l ^ r) upd(ls[oo], p, lch);
    }
    int qry(int k, int l, int r, int o, int oo) {
        if(l == r) return k > 1 ? m : l;
        int cnt = c[ls[o]] - c[ls[oo]];
        return k <= cnt ? qry(k, lch, ls[oo]) : qry(k - cnt, rch, rs[oo]);
    }
    void upd(int p, int t) {
        upd(T[t], p, 0, m, T[t+1]);
    }
    int qry(int k, int l, int r) {
        return qry(k, 0, m, T[r], T[l]);
    }
};
namespace Tree {
    int idx, dfn[N], suf[N], rnk[N], d[N], fa[20][N];
    vector <int> G[N]; 
    void dfs(int u) {
        rnk[dfn[u] = ++idx] = u;
        rep(i, 1, 19) fa[i][u] = fa[i-1][fa[i-1][u]];
        for(int v : G[u]) if(v ^ fa[0][u])
            fa[0][v] = u, d[v] = d[u] + 1, dfs(v);
        suf[u] = idx;
    }
    int lca(int u, int v) {
        if(d[u] < d[v]) swap(u, v);
        per(i, 19, 0) if(d[u] - (1 << i) >= d[v]) u = fa[i][u];
        if(u == v) return u;
        per(i, 19, 0) if(fa[i][u] ^ fa[i][v]) u = fa[i][u], v = fa[i][v];
        return fa[0][u];
    }
    
    int Max[N], R[N], x[N], c[N], g[N];
    vector <int> tim[N];
    void pretreat() {
        cin >> m;
        rep(i, 1, m) {
            scanf("%d%d", &x[i], &c[i]), g[i] = c[i];
            tim[dfn[x[i]]].push_back(i);
        }
        sort(g + 1, g + m + 1);//离散化颜色
        rep(i, 1, m) c[i] = lower_bound(g + 1, g + m + 1, c[i]) - g;
        rep(i, 1, n) {
            R[i] = R[i-1];//R[i] 是 dfs 序为 i 的节点对应区间的右端点
            for(int t : tim[i]) Seg::upd(t, R[i]), R[i]++;
        }
        rep(i, 1, n) Max[i] = Seg::qry(k[i], R[dfn[i]-1], R[suf[i]]);
    }
    
    int C[N];
    void add(int i, int v) {
        for(; i <= n; i += i & -i) C[i] += v;
    }
    int qry(int i, int r = 0) {
        for(; i; i &= i - 1) r += C[i];
        return r;
    }
    int Qry(int u) {
        return qry(suf[u]) - qry(dfn[u] - 1);
    }
    
    set <int> s[N];
    void upd(int u, int v, int x) {
        add(dfn[lca(rnk[u], rnk[v])], x);
    }
    void ins(int i) {//单点加球
        int ci = c[i], dfnx = dfn[x[i]];
        if(s[ci].count(dfnx)) return;
        s[ci].insert(dfnx), add(dfnx, 1);
        auto it = s[ci].find(dfnx);
        int pre = 0, suf = 0;
        if(it != s[ci].begin()) pre = *(--it), it++;
        if(it != --s[ci].end()) suf = *(++it), it--;
        if(pre) upd(pre, dfnx, -1);
        if(suf) upd(suf, dfnx, -1);
        if(pre && suf) upd(pre, suf, 1);
    }
};

int x[N], u, v, ans[N]; vector <int> id[N];
int main() {
    cin >> n;
    rep(i, 2, n) {
        scanf("%d%d", &u, &v);
        Tree::G[u].push_back(v);
        Tree::G[v].push_back(u);
    }
    Tree::dfs(1);
    rep(i, 1, n) scanf("%d", &k[i]);
    Tree::pretreat();
    cin >> q;
    rep(i, 1, q) scanf("%d", &x[i]), id[Tree::Max[x[i]]].push_back(i);
    rep(i, 1, m) {
        Tree::ins(i);
        for(int j : id[i]) ans[j] = Tree::Qry(x[j]);
    }
    rep(i, 1, q) printf("%d\n", ans[i]);
    return 0;
}
```