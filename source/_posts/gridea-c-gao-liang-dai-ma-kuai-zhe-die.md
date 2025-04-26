---
title: Gridea C++ 高亮代码块折叠
date: 2021-08-23 18:53:32
updated: 2021-08-23 18:53:32
tags: []
hidden: true
---
代码全部平铺在页面上有点影响阅读，想将代码块折叠起来。

在网上找到了内嵌 `html` 代码来实现折叠，但不支持语法高亮，我找到了一种仅适用于 `Gridea` 博客的方法来实现 `C++` 语法高亮，如下：

```plain
<details><summary>查看代码</summary>
<pre><code class="language-cpp">#include &lt;bits/stdc++.h&gt;<br>
int main() {
    int a, b;
    scanf("%d%d", &a, &b);
    printf("%d\n", a + b);
}
</code></pre></details>
```

`C++` 代码不能直接粘中间，需要进行两个改动：

- 把 `<` 替换成 `&lt;`，`>` 替换成 `&gt;`。

- 对于每一个空行，在上一个非空行的结尾加上 `<br>` 并删除该空行。

我在下面提供了一个把 `markdown` 源码中所有或指定的代码块替换成折叠型的小工具。

实际效果

<details><summary>查看代码</summary>
<pre><code class="language-cpp">#include &lt;bits/stdc++.h&gt;<br>
int main() {
    int a, b;
    scanf("%d%d", &a, &b);
    printf("%d\n", a + b);
}
</code></pre></details>

<details><summary><span style="font-size: large; font-weight: bold; color: rgb(33,150,243);">小工具</span></summary><pre><code class="language-cpp">#include &lt;bits/stdc++.h&gt;<br>
using namespace std;<br>
int main(int argc, char* argv[]) {
    if(argc &lt; 3) exit(0);
    freopen(argv[1], "r", stdin), freopen(argv[2], "w", stdout);
    ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
    set&lt;int&gt; rows;
    int tmp;
    for(int i = 3; i &lt; argc; i++) sscanf(argv[i], "%d", &tmp), rows.insert(tmp);
    string str, CodeBlock;
    bool isCodeBlock = false;
    for(int row = 1; getline(cin, str); row++)
    if(isCodeBlock) {
        if(str == "```") cout &lt;&lt; CodeBlock + "&lt;/code&gt;&lt;/pre&gt;&lt;/details&gt;\n", isCodeBlock = false;
        else if(str == "") CodeBlock.insert(CodeBlock.size() - (CodeBlock.back() == '\n'), "&lt;br&gt;");
        else {
            for(char ch : str) if(ch == '&lt;') CodeBlock += "&lt;";
            else if(ch == '&gt;') CodeBlock += "&gt;";
            else CodeBlock += ch;
            CodeBlock += '\n';
        }
    } else {
        if(str == "```cpp" && (argc == 3 || rows.count(row)))
            CodeBlock = "&lt;details&gt;&lt;summary&gt;&lt;span style=\"font-size: large; font-weight: bold; color: rgb(33,150,243);\"&gt;查看代码&lt;/span&gt;&lt;/summary&gt;&lt;pre&gt;&lt;code class=\"language-cpp\"&gt;", isCodeBlock = true;
        else cout &lt;&lt; str &lt;&lt; '\n';
    }
}
</code></pre></details>

假设小工具被保存为 `tool.cpp`，编译后可以在命令行中使用。

- `tool <input-file> <output-file>`，其中 `<input-file>` 为 `markdown` 源文件，`<output-file>` 为处理后的文件，会将所有的 `cpp` 代码块换成折叠代码块。

- `tool <input-file> <output-file> row1 row2 row3 ...`，其中第 `row1`，`row2`，`row3`，... 行必须是 ` ```cpp `，将指定的代码块换成折叠代码块。
