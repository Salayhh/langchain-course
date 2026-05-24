# LangChain Hello World - 流式输出入门

这份代码是 LangChain 的入门示例，演示如何用**提示词模板（PromptTemplate）** + **大语言模型（LLM）**构建一个最简单的 AI 应用，并支持**流式输出（Streaming）**。

---

## 项目结构

```
.
├── main.py           # 主代码
├── pyproject.toml    # 项目依赖配置
├── .env              # 环境变量（API 密钥）
└── uv.lock           # 依赖锁定文件
```

---

## 前置条件

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) 包管理器（本项目使用 uv）
- 一个 Moonshot API 密钥（或替换为 OpenAI、Ollama 等其他模型）

---

## 安装与运行

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置 API 密钥

编辑 `.env` 文件，填入你的 Moonshot API 密钥：

```env
MOONSHOT_API_KEY="your-api-key-here"
```

> 如果你使用 OpenAI，可以直接用 `OPENAI_API_KEY`，并修改代码中的 `base_url`。
> 如果你使用本地 Ollama 模型，可以参考代码中的注释切换。

### 3. 运行程序

```bash
uv run main.py
```

你会看到模型**逐字输出**总结内容，而不是等全部生成完才一次性显示。

---

## 代码逐行解析

### 导入依赖

```python
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
```

| 模块 | 作用 |
|------|------|
| `dotenv` | 从 `.env` 文件加载环境变量，避免把密钥硬编码到代码里 |
| `PromptTemplate` | 提示词模板，用来定义给 AI 的指令格式 |
| `ChatOpenAI` | 兼容 OpenAI 接口格式的聊天模型（Moonshot、DeepSeek 等也用这个） |
| `ChatOllama` | 本地 Ollama 模型（如 llama3、qwen 等），不需要联网 |

### 加载环境变量

```python
load_dotenv()  # 读取 .env 文件中的 MOONSHOT_API_KEY
```

### 定义提示词模板

```python
summary_template = """
given the information {information} about a person I want you to create:
1. A short summary
2. two interesting facts about them
"""

summary_prompt_template = PromptTemplate(
    input_variables=["information"], template=summary_template
)
```

- `{information}` 是一个**占位符**，运行时会用实际内容替换。
- `input_variables=["information"]` 告诉模板：这个模板需要一个叫 `information` 的变量。

### 选择模型

```python
llm = ChatOpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
    model="kimi-k2.5"
)
```

这里使用的是 **Moonshot（月之暗面）**的 API，通过 `ChatOpenAI` 调用，因为 Moonshot 兼容 OpenAI 的接口格式。

**如果你想换成本地 Ollama 模型**，可以改成：

```python
llm = ChatOllama(model="llama3")  # 或 qwen2.5、phi4 等
```

### 构建链（Chain）

```python
chain = summary_prompt_template | llm
```

这是 LangChain 的 **LCEL（LangChain Expression Language）**语法：

- `|` 符号表示"把左边的输出传给右边"。
- 这里表示：先渲染提示词模板 → 再把结果发给 LLM。

### 流式调用

```python
for chunk in chain.stream(input={"information": information}):
    print(chunk.content, end="", flush=True)
```

| 要点 | 说明 |
|------|------|
| `chain.stream()` | 流式调用，模型生成一段就返回一段 |
| `chunk.content` | 当前片段的文本内容 |
| `end=""` | 取消 `print` 默认的换行，让片段连续显示 |
| `flush=True` | 强制立即输出到终端，而不是等缓冲区满 |

对比非流式写法：

```python
# 非流式：等模型全部生成完，一次性返回
response = chain.invoke(input={"information": information})
print(response.content)
```

流式的好处是**用户不用等待**，大段文本可以边生成边看。

---

## 关键概念速查

| 概念 | 一句话解释 |
|------|-----------|
| **PromptTemplate** | 带变量的提示词模板，方便复用和动态填充 |
| **Chain / LCEL** | 用 `\|` 把多个步骤串联成流水线 |
| **invoke** | 一次性调用，等全部结果返回 |
| **stream** | 流式调用，边生成边返回 |
| **ChatOpenAI** | 兼容 OpenAI 接口的聊天模型封装 |

---

## 扩展练习

1. **换模型**：把 `ChatOpenAI` 换成 `ChatOllama(model="llama3")`，实现本地运行。
2. **加输出解析器**：用 `StrOutputParser()` 让输出更干净：
   ```python
   from langchain_core.output_parsers import StrOutputParser
   chain = summary_prompt_template | llm | StrOutputParser()
   for text in chain.stream(...):
       print(text, end="", flush=True)
   ```
3. **加更多变量**：在模板里加 `{style}` 变量，让用户指定总结风格（正式/幽默/简洁）。
