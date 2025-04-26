---
title: NIM 游戏 & SG 定理
date: 2021-03-31 10:45:39
updated: 2021-03-31 10:45:39
tags: [知识总结,博弈论,阶梯 NIM 游戏]
categories: 算法
---
定义**必胜状态**为**当前局面先手必胜的状态**，**必败状态**为**当前局面先手必败的状态**。

### 有向图游戏

> 在一个**有向无环图**中，只有一个起点，上面有一个棋子，两个玩家轮流沿着有向边推动棋子，不能走的玩家判负。

对于点 $u$ 和它的 $k$ 个后继点 $v_1,v_2,\cdots,v_k$，定义 ```SG``` 函数：
$$
SG(u)=\text{mex}\{SG(v_1),SG(v_2),\cdots,SG(v_k)\}
$$
特别地，当 $u$ 没有后继状态时 $SG(u)=0$。

显然，先手必胜当且仅当 $SG(u) \ne 0$。

对于 $n$ 个有向图游戏组合而成的游戏，即两个玩家轮流推动 $n$ 个棋子之一，设这个游戏的初始状态为 $S$，棋子的起点分别为 $s_1,s_2,\cdots,s_n$，则有 **```SG``` 定理**：

$SG(S) = SG(s_1) \oplus SG(s_2) \oplus \cdots \oplus SG(s_n)$。

证明：由于组合而成的游戏也是有向图游戏，因此所有的状态存在一个拓扑序，以下使用**数学归纳法**证明：

- 归纳奠基：对于没有后继的状态 $S$，$\forall i \in [1,n], SG(s_i)=0, SG(S)=0$，命题成立。

- 归纳假设：设当前状态 $S$ 满足 $SG(s_1) + SG(s_2) + \cdots + SG(s_n) = m$，对于所有满足 $SG(s_1') + SG(s_2') + \cdots + SG(s_n') < m$ 的状态以及所有满足 $SG(s_1') + SG(s_2') + \cdots + SG(s_n') = m$ 并且拓扑序小于 $S$ 的状态命题成立。

- 归纳递推：设 $SG(s_1) \oplus SG(s_2) \oplus \cdots \oplus SG(s_n) =k$。

  先证明 $\forall k' < k$, $S$ 都存在一个后继状态 $S'$ 满足 $SG(S')=k'$。 
  
  设 $k \oplus k'$ 在二进制下的最高位为第 $m$ 位，因为 $k' < k$, 所以 $k'$ 的第 $m$ 位为 $0$，$k$ 的第 $m$ 位为 $1$。

  根据异或定义，存在 $i$ 满足 $SG(s_i)$ 的第 $m$ 位为 $1$。

  那么 $SG(s_i) \oplus k \oplus k' < SG(s_i)$，根据 ```SG``` 函数定义，$s_i$ 一定存在后继点 $s_i'$，满足 $SG(s_i') = SG(s_i) \oplus k \oplus k'$。

  把 $s_i$ 推到 $s_i'$ 后满足 $SG(s_1) + SG(s_2) + \cdots + SG(s_i') + \cdots + SG(s_n) < m$，因此 $SG(S')=SG(s_1) \oplus SG(s_2) \oplus \cdots \oplus SG(s_i') \oplus \cdots \oplus SG(s_n)=k'$。

  再证明 $S$ 不存在后继状态 $S'$ 满足 $SG(S')=k$。

  如果把 $s_i$ 推到 $s_i'$，一定 $SG(s_i') \ne SG(s_i)$。

  如果 $SG(s_i') < SG(s_i)$，那么 $SG(s_1) + SG(s_2) + \cdots + SG(s_i') + \cdots + SG(s_n) < m$，因此 $SG(S')=SG(s_1) \oplus SG(s_2) \oplus \cdots \oplus SG(s_i') \oplus \cdots \oplus SG(s_n) \ne k$。

  如果 $SG(s_i') > SG(s_i)$，那么 $SG(s_1) + SG(s_2) + \cdots + SG(s_i') + \cdots + SG(s_n) > m$，因此 $SG(S')$ 未知，但我们只需证 $SG(S') \ne k$。根据 ```SG``` 函数定义，$s_i'$ 一定存在后继点 $s_i''$，满足 $SG(s_i'') = SG(s_i)$，此时 $SG(s_1) + SG(s_2) + \cdots + SG(s_i'') + \cdots + SG(s_n) = m$，并且 $S''$ 的拓扑序小于 $S$，因此 $SG(S'')=SG(s_1) \oplus SG(s_2) \oplus \cdots \oplus SG(s_i'') \oplus \cdots \oplus SG(s_n)=k$，所以一定有 $SG(S') \ne k$。

### NIM 游戏

很多的公平组合游戏都可以转化为多个有向图游戏的组合，比如 **NIM 游戏**。

> $n$ 堆物品，每堆有 $a_i$ 个，两个玩家轮流取走任意一堆的任意个物品，但不能不取。
>
> 取走最后一个物品的人获胜。

对于每一堆石子，定义结点 $x$ 表示有 $x$ 个物品的状态。所有编号大的结点向编号小的结点连边。

这样，NIM 游戏就转化 $n$ 个有向图游戏的组合了。

易得 $SG(x)=x$，根据 ```SG``` 定理，当且仅当 $a_1 \oplus a_2 \oplus \cdots \oplus a_n \ne 0$ 时，先手必胜。

### 阶梯 NIM 游戏

>  $n$ 堆物品，每堆有 $a_i$ 个，两个玩家轮流取走任意第 $i$ 堆（$i > 1$）的任意个物品移到第 $i-1$ 堆中，但不能不取。
>
> 最后没有物品可取的人输。

结论：**此游戏等价于 $\lfloor \frac n2 \rfloor$ 堆物品，每堆有  $a_{2i}$ 个物品的 NIM 游戏。**

不妨假设在 NIM 游戏中 ```A``` 必胜，```B``` 必败，则 ```A``` 可以使用以下策略：

- 如果前一步 ```B```  从第 $2i$ 堆里取物品或者当前为第一步，```A``` 按照 NIM 里的策略从对应的堆里取物品。
- 如果前一步 ```B``` 从第 $2i+1$ 堆里取物品，```A``` 将这些物品再移到第 $2i-1$ 堆。

这样奇数堆中的物品就不会影响胜负情况，而在偶数堆中取物品相当于直接删除（因为它们到奇数堆后就不会影响胜负情况）。