---
title: SMAWK 简介
date: 2021-11-05 20:50:26
tags:
---

如果一个 $N\times M$ 的矩阵 $A$ 满足四边形不等式：
$$
\forall i_1<i_2,j_1<j_2,A_{i_1,j_1}+A_{i_2,j_2}\le A_{i_1,j_2}+A_{i_2,j_1}
$$
则称矩阵 $A$ 是完全单调的。

SMAWK 算法解决的问题是求 $A$ 每行的最小值，这个问题当然可以用基于决策单调性的分治法做，但 SMAWK 的复杂度更为优秀。假设可以 $O(1)$ 计算矩阵 $A$ 中的一个元素，SMAWK 的复杂度为 $O(N+M)$​。

矩阵 $A$ 的一个性质是决策单调性，即每行取到最小值的列编号随行编号递增。证明不困难，考虑
$$
A_{i,j}+A_{i+1,j+1}\le A_{i,j+1}+A_{i+1,j}\Rightarrow A_{i,j+1}-A_{i,j}\ge A_{i+1,j+1}-A_{i+1,j}
$$
即 $A_{*,j}$ 比 $A_{*,j+1}$ 随行数增加得更快，如果第 $i$ 行的最小值在第 $j$ 列取到，那么在 $i$ 之后的行，$j$ 之前的列都比第 $j$ 列劣。

SWAWK 的核心操作是 reduce，可以去掉一些没用的列，使得剩余列数不超过行数。

reduce 操作维护一个存列编号的栈，它的性质为：对于栈内自底向上第 $i$ 个元素 $x$ 和第 $i+1$ 个元素 $y$，满足 $A_{i,x}< A_{i,y}$。初始栈为空，然后从小到大插入每一列。假设当前要插入 $c$，栈顶元素为 $top$，栈的大小为 $size$，如果 $A_{size,top}\ge A_{size,c}$，那么可以弹掉 $top$，原因：假设 $top$ 在栈中的前一个元素为 $x$，由于 $A_{size-1,x}< A_{size-1,top}$，在第 $1\rightarrow size-1$ 行中 $top$ 没有 $x$ 优秀，同理，在第 $size\rightarrow n$ 行中 $top$ 没有 $c$ 优秀，故第 $top$ 列是没用的。

弹掉没用的元素之后，如果当前栈的大小为 $n$，说明此时 $A_{n,top}< A_{n,c}$，元素 $c$ 就不用插入了。否则就插入 $c$，这不会破环栈的性质。最后栈内的剩余元素就是可能有用的列。

reduce 操作之后先求出偶数行的最小值和取到最小值的列，这是一个子问题，之后可以通过决策单调性 $O(n)$ 求出奇数行的最小值。

```cpp
int ans[MAXROW]; // 存每一行的最小值在哪一列
int A(int x, int y) {
    // O(1) 计算 A[x][y]
}
// 求第 0, k, 2k, 3k, ... 行的最小值
void SMAWK(int k, const vector<int>& row, const vector<int>& col) {
    vector<int> Stack;
    for(int c : col) {
        while(!Stack.empty()) {
            int r = Stack.size() - 1;
            if(A(r, c) > A(r, Stack.back())) break;
            Stack.pop_back();
        }
        if(Stack.size() < row.size()) Stack.push_back(c);
    }
    // ↑ reduce
    if(k < row.size()) SMAWK(k << 1, row, Stack);
    int r = k, c = 0;
    auto chkmin = [&]() {
        if(A(r, Stack[c]) < A(r, ans[r])) ans[r] = Stack[c];
    };
    while(r + k < row.size()) {
        while(Stack[c] < ans[r + k]) chkmin(), c++;
        r += k << 1;
    }
    if(r < row.size()) while(c < Stack.size()) chkmin(), c++;
}
```