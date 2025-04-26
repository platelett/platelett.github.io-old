---
title: 从 C++11 到 C++17
date: 2021-09-01 16:30:02
updated: 2021-09-01 16:30:02
tags: []
---
建议在编程中尝试使用 `C++17`，可以一定程度上简化代码编写，提高编程效率。

从 `C++11` 到 `C++14` 和 `C++17` 的部分实用特性：

### lambda 的泛型参数和捕获的增强。

允许 lambda 函数的形式参数声明中使用 `auto`。

```cpp
auto lambda = [](auto x, auto y) { return x + y; };
// since C++14
```

允许 lambda 函数的捕获部分中定义变量同时初始化。

```cpp
int a = 1, b = 2, c = 3;
auto lambda1 = [value = a + b] { return value + c; };
// since C++14
```

允许成员函数中的 lambda 函数以拷贝的方式捕获 `*this`。

```cpp
struct node { int value; void func(); };
int node::func() {
    auto lambda = [this]() { return ++value; }; // by reference
    return lambda();
}
// since C++11
struct node { int value; void func(); };
int node::func() {
    auto lambda = [*this]() { return ++value; }; // by copy
    return lambda();
}
// since C++17
```

### 函数返回类型推导。

`C++11` 允许 lambda 函数返回值使用 `auto`，`C++14` 允许一般函数也可以这样做，甚至可以递归，但递归调用必须在函数定义中的至少一个 `return` 语句之后。

```cpp
auto factorial(int n) {
    if(n == 0) return 1;
    return factorial(n - 1) * n;
}
// since C++14
```

错误示例：

```cpp
auto factorial(int n) {
    if(n) return factorial(n - 1) * n;
    return 1;
}
```

### 构造函数模板推导

构造一个模板类对象不需要指明模板参数。

```cpp
pair a(1, 2.2);
vector b{1, 2, 3};
set c{1, 2, 3};
tuple d(1, 2.2, a);
// since C++17
```

### 结构化绑定

将结构体拆包，相当于 `C++11` 中 `tie` 函数的增强。

```cpp
vector a{pair(1, 2), pair(2, 3)};
auto [first, second] = a[0];
cout << first << ' ' << second << endl;
for(auto& [first, second] : a) first++, second++; // by reference
for(auto [first, second] : a) cout << first << ' ' << second << endl; // by copy
// since C++17
```

### `if` 语句初始化

`if` 可以像 `for` 循环一样有一条初始化语句。

```cpp
if(int value = func(); value < 100) cout << value;
// since C++17
```

### 二进制字面量和其他新增标准字面量

数字可以使用二进制形式指定，其格式使用前缀 `0b` 或 `0B`。

在一个字符串后加 `s`，表示 `string` 类型。

在一个数字后加 `i`，`if`，`il` 分别表示 `complex<double>`、`complex<float>` 和 `complex<long double>` 复数类型。

```cpp
auto str = "hello world"s; // string
auto value = 0b1010; // 10
auto z = 1i; // complex<double>
// since C++14
```

### 折叠表达式

可以用来方便的定义参数个数不定的函数。

```cpp
template<class ...T> auto sum(T ...value) { return value + ...; }
// since C++17
```

### namespace 嵌套

```cpp
namespace A {
    namespace B {
        namespace C {
            int func();
        }
    }
}
namespace A::B::C {
    int func() { return 100; }
}
// since C++17
```

### 字符串转化

`from_chars` 是把 `const char*` 转化成整数或浮点数。

`to_chars` 是把整数或浮点数转化成 `const char*`。

