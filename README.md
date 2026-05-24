# LangChain Search Agents Tutorial

This branch (`project/search-agent`) demonstrates how to build search agents using LangChain's `create_agent` interface. The tutorial progresses through three key concepts, showing how to evolve from a basic custom tool implementation to using structured outputs with built-in LangChain integrations.

---

## 代码逐行解释（面向初学者）

如果你刚接触 LangChain，下面是对 [main.py](main.py) 每一部分的详细拆解。

### 1. 导入依赖

```python
import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
```

- `os`：用来读取环境变量（比如 API Key）。
- `typing.List`：类型提示，表示一个列表。
- `pydantic.BaseModel` / `Field`：用来定义**结构化数据模型**。你可以把它理解为 Python 版的 JSON Schema，告诉程序"数据应该长什么样"。
- `dotenv.load_dotenv`：自动加载项目根目录下 `.env` 文件里的环境变量，避免把密钥硬编码在代码里。

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
```

这是 LangChain 生态的核心模块：
- `create_agent`：LangChain 封装好的"创建智能体"函数，几行代码就能搭一个 Agent。
- `tool`：装饰器，用来把普通 Python 函数变成 Agent 可以调用的"工具"。
- `HumanMessage`：代表**用户输入的消息**。LangChain 用消息列表来管理对话历史。
- `ChatOpenAI`：用来连接 OpenAI 风格 API 的大语言模型。这里我们连接的是 **Moonshot（Kimi）** 的 API。
- `TavilySearch`：LangChain 官方封装的 **Tavily 搜索引擎** 工具，让 Agent 能联网查资料。

---

### 2. 定义结构化输出模型

```python
class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")
```

- 定义了一个 `Source` 类，表示"一个信息来源"。
- `BaseModel` 是 Pydantic 的基类，它会自动帮你做**类型校验**。
- `Field(description=...)`：给字段加一段人类可读的描述。这段描述**会被送给大模型看**，让模型知道这里该填什么。

```python
class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )
```

- 这是最终输出的"模板"。我们告诉 Agent：你回答问题时，必须按这个格式交卷——
  - `answer`：你的文字回答
  - `sources`：你用了哪些来源，每个来源给一个 URL
- `default_factory=list`：如果没有来源，默认给一个空列表，而不是报错。

> **为什么要结构化输出？**  如果不加约束，模型可能随便吐一段文字，来源混在正文里，程序很难提取。有了 Pydantic 模型，输出就是标准的 Python 对象，可以直接用代码处理。

---

### 3. 初始化大语言模型（LLM）

```python
llm = ChatOpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key=os.getenv('MOONSHOT_API_KEY'),
    model="kimi-k2.5",
    extra_body={"thinking": {"type": "disabled"}}
)
```

- `base_url`：指向 Moonshot（Kimi）的 API 地址，而不是默认的 OpenAI 官方地址。
- `api_key`：从环境变量读取密钥，安全又方便。
- `model`：指定使用 `kimi-k2.5` 模型。
- `extra_body={"thinking": {"type": "disabled"}}`：这是一个**兼容性修复**。Kimi K2.5 默认会开启"深度思考"模式，但 LangChain 的 `response_format` 会强制要求模型**立刻输出 JSON**，两者冲突会导致 400 错误。关闭 thinking 后，模型就能乖乖按格式输出。

> 关于这个 Bug 的详细原理，见下方 [fix kimi toolcalling bug](#fix-kimi-toolcalling-bug) 章节。

---

### 4. 创建工具和智能体

```python
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
```

- `tools = [TavilySearch()]`：给 Agent 配一个"工具箱"，目前只有一个工具——Tavily 搜索引擎。
- `create_agent(...)`：LangChain 的核心魔法，三行代码搭好一个 Agent：
  - `model=llm`：指定大脑（Kimi）
  - `tools=tools`：指定手脚（搜索引擎）
  - `response_format=AgentResponse`：指定输出的格式（用我们定义的 Pydantic 模型）

> **Agent 的工作逻辑**：你问一个问题 → Agent 判断需不需要搜索 → 如果需要，调用 TavilySearch → 拿到结果后，Kimi 整理答案 → 最终按 `AgentResponse` 格式输出。

---

### 5. 主函数：运行 Agent

```python
def main():
    print("Hello from langchain-course!")
    result = agent.invoke({
        "messages": HumanMessage(content="搜索美伊战争最新动态")
    })
    print(result)
```

- `agent.invoke(...)`：**同步调用** Agent。传入一个字典，其中 `messages` 是一个 `HumanMessage` 对象，代表用户的问题。
- Agent 收到问题后，会自动决定要不要搜索、搜索什么关键词、怎么整理答案。
- 最终返回的 `result` 是一个字典，里面包含了完整的对话历史、工具调用记录和最终的 `structured_response`。

```python
if __name__ == "__main__":
    main()
```

- Python 的标准入口写法。直接运行 `python main.py` 时，会执行 `main()` 函数。

---

### 6. 整体流程图

```
用户提问 (HumanMessage)
       │
       ▼
┌──────────────┐
│   Agent      │ ← create_agent 创建的调度器
│ (Kimi 大脑)  │
└──────┬───────┘
       │ 发现需要搜索
       ▼
┌──────────────┐
│ TavilySearch │ ← 联网搜索最新资讯
│   (工具)     │
└──────┬───────┘
       │ 返回搜索结果
       ▼
┌──────────────┐
│   Agent      │ ← 整理、总结、引用来源
│ (Kimi 大脑)  │
└──────┬───────┘
       │ 按 AgentResponse 格式输出
       ▼
┌──────────────────────────┐
│  AgentResponse           │
│  ├── answer: str         │
│  └── sources: [Source]   │
└──────────────────────────┘
```

---

## Learning Objectives

- Understand the LangChain `create_agent` interface
- Build custom tools using the `@tool` decorator
- Integrate third-party search APIs (Tavily)
- Use LangChain's built-in tool integrations
- Implement structured outputs with Pydantic models

## Commits Overview

| # | Commit | Title | What You'll Learn | Key Changes |
|---|--------|-------|-------------------|-------------|
| 1 | [d2bd87f](https://github.com/emarco177/ice_breaker/commit/d2bd87f825f3558f6674fcde854767206bfcfe63) | intro to search agents | Project setup, creating custom search tools with `@tool` decorator, direct Tavily API integration, building an agent with `create_agent`, invoking agents with messages | Initial project setup with dependencies, created custom `search` function using `@tool`, direct `TavilyClient` integration, created agent using `create_agent(model=llm, tools=tools)`, invoked agent with `HumanMessage` |
| 2 | [8f78259](https://github.com/emarco177/ice_breaker/commit/8f782592c4e6bd083a4973588b82514a87836975) | langchain tavily built in tool | Using LangChain's built-in tool integrations instead of custom tools, code simplification and best practices | Replaced custom `@tool` with `TavilySearch` from `langchain_tavily`, removed boilerplate code, cleaner implementation |
| 3 | [ba398aa](https://github.com/emarco177/ice_breaker/commit/ba398aa03795ea3fbb341e2b599dce5aae59ebe4) | added structured output | Implementing structured outputs with Pydantic models, type-safe agent responses using `response_format` parameter | Added `Source` and `AgentResponse` BaseModels, configured `response_format` in `create_agent`, enforced predictable output structure |

## Running the Code

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set up environment variables:
   ```bash
   # Create a .env file with:
   OPENAI_API_KEY=your_openai_key
   TAVILY_API_KEY=your_tavily_key
   ```

3. Run the agent:
   ```bash
   python main.py
   ```

## Example Query

The agent searches for AI engineer job postings in the Bay Area on LinkedIn:
```python
"美伊战争最新动态"
```

## Key Takeaways

- **Custom Tools**: You can create custom tools using the `@tool` decorator for specialized functionality
- **Built-in Integrations**: LangChain provides pre-built tools that reduce boilerplate and improve maintainability
- **Structured Outputs**: Using Pydantic models with `response_format` ensures type-safe, predictable agent responses
- **Agent Interface**: The `create_agent` function provides a simple, consistent interface for building agents with different capabilities

## fix kimi toolcalling bug

直接使用kimi api会报错：
```
openai.BadRequestError: Error code: 400 - {'error': {'message': "tool_choice 'required' is incompatible with thinking enabled", 'type': 'invalid_request_error'}}
During task with name 'model' and id '3ecb65f6-dd44-9ebb-8095-b7b107a3fe33'
```

必须要将llm部分改成：

```python
llm = ChatOpenAI(base_url="https://api.moonshot.cn/v1",
                 api_key=os.getenv('MOONSHOT_API_KEY'),
                 model="kimi-k2.5",
                 extra_body={"thinking": {"type": "disabled"}})
```
这个冲突的本质，其实是 **“必须立刻行动”** 与 **“允许先思考”** 这两种指令，在 API 协议层面产生了**逻辑悖论**。

我们可以把大模型当成一个办事员，拆开来看看两边分别下了什么命令：

---

### 1. LangChain 的死命令：`tool_choice="required"`

当你写下 `response_format=AgentResponse` 时，LangChain 为了确保模型交出来的作业绝对符合你定义的 Pydantic 格式（包含 `answer` 和 `sources`），它在底层玩了个“障眼法”：

* 它把你的 `AgentResponse` 结构体包装成了一个**隐式工具（Implicit Tool）**。
* 并在发给 Kimi 的 API 请求中，强制加上了参数：`"tool_choice": "required"`。

> **这个参数的意思是：** “你不要和我说任何废话，你的**第一句回答（输出的第一个 Token 起）**，必须且只能是一个标准的工具调用 JSON 格式。

---

### 2. Kimi-k2.5 的本能：思维链（Thinking Enabled）

作为新一代的推理模型（Reasoning Model），Kimi-k2.5 默认开启了深度思考。

* 它的工作原理是：看到问题后，不急着回答，而是先在脑子里高强度运转，输出一段**思维链（Chain of Thought）**。
* 这段思考过程表现为一串“思考 Token”（在前端通常表现为折叠的“思考中...”文本）。

> **这个模式的意思是：** “别催我，我在正式干活（调用工具或回答）之前，**必须先输出一段文字来理清我的思路**。”

---

### 💥 悖论卡死了 API

当这两个参数同时传给 Moonshot 的服务器时，API 网关直接懵了。因为它们在时序上是完全冲突的：

| 阶段 | Kimi 想做的事情（Thinking） | LangChain 强迫它做的事情（Required） |
| --- | --- | --- |
| **第 1 步** | 输出“思考内容：我应该先搜什么……” | 必须立刻输出 `{"name": "AgentResponse", "args": ...}` |
| **结果** | ❌ 违背了 LangChain 的格式强力约束 | ❌ 强行剥夺了模型的思考过程 |

大模型如果先思考，就违反了“必须立刻调用工具”的铁律；如果大模型不思考直接调用工具，又违反了推理模型的设计初衷。

由于 Moonshot 的服务器底层做了解析校验，发现这种**既要大模型深度思考、又逼着它一句话不输出直接吐出工具 JSON** 的参数组合是无法实现的，为了防止后续产生畸形的报错，它索性在最前端直接拦截，抛出了这个 400 错误：

> `"tool_choice 'required' is incompatible with thinking enabled"`
> *（调兵指令“必须强制指定工具”与“开启深度思考模式”互不兼容）*

---

### 🛠️ 为什么你的修改能成功？

* **方案一（关闭思考）**：去掉了 Kimi 的“思考本能”，让它退化成一个传统的、一问一答的普通大模型。它不需要先写思维链了，自然就能完美配合 LangChain 的 `"required"` 强力约束，一上来就吐出 JSON。
* **方案二（改用提示词）**：把 LangChain 的 `"required"` 撤掉（将 `tool_choice` 恢复为默认的 `"auto"`）。Kimi 就可以高高兴兴地**先在脑子里思考（输出思考 Token），思考完了，再自动决定在最后调用工具或返回 JSON**。

## Output

{'messages': [HumanMessage(content='搜索美伊战争最新动态', additional_kwargs={}, response_metadata={}, id='621b4a97-5c7b-4693-b02b-c8eafd50ff69'), AIMessage(content='', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 47, 'prompt_tokens': 1542, 'total_tokens': 1589, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 1280}, 'cached_tokens': 1280}, 'model_provider': 'openai', 'model_name': 'kimi-k2.5', 'system_fingerprint': None, 'id': 'chatcmpl-6a12d55b6fc276ab303e4b0b', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--019e5991-6ae1-7861-8e1f-8ce2788ac395-0', tool_calls=[{'name': 'tavily_search', 'args': {'query': '美伊战争 最新动态 最新消息', 'search_depth': 'advanced', 'time_range': 'month', 'topic': 'news'}, 'id': 'tavily_search:0', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 1542, 'output_tokens': 47, 'total_tokens': 1589, 'input_token_details': {'cache_read': 1280}, 'output_token_details': {}}), ToolMessage(content='{"query": "美伊战争 最新动态 最新消息", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "https://www.wsj.com/livecoverage/iran-us-china-news-2026", "title": "Iran War Live News Updates: Trump Arrives in China Against Background of Unresolved Iran War - WSJ", "score": 0.9722044, "published_date": "Wed, 13 May 2026 03:17:30 GMT", "content": "Military leaders in Iran have said their control of the Strait of Hormuz will ultimately be more lucrative than exporting oil. That’s because 20% of the world’s oil typically moves through the narrow waterway that connects the Persian Gulf to the world. Countries across Europe and Asia count on importing fuel from Iran’s neighbors, including Saudi Arabia, Kuwait and Iraq.\\n\\nHere’s what else to know:\\n\\nIsrael said Prime Minister Benjamin Netanyahu had secretly visited the United Arab Emirates and met with the country’s president. Later, the U.A.E. denied any such visit had taken place.\\n\\nA China Cosco Shipping supertanker crossed the strait without paying tolls, seen by the company as a goodwill gesture by Iran toward Beijing ahead of the Trump-Xi summit. [...] Saudi Arabia and Kuwait have attacked Iran-backed militias in Iraq multiple times during the war.\\n\\nThe Pentagon is pushing to acquire 10,000 low-cost cruise missiles amid concerns the U.S. is running low on munitions.\\n\\nImage 2Image 3Image 4Image 5Image 6Image 7\\n\\nImage 8\\n\\nImage 9 [...] # Iran War Live News Updates: Trump Arrives in China Against Background of Unresolved Iran War\\n\\nDJIA 49693.20 -0.14%\\n\\nS&P 500 7444.25 0.58%\\n\\nNasdaq 26402.34 1.20%\\n\\nRussell 2000 2843.93 0.04%\\n\\nU.S. 10 Yr -5/32 4.469%\\n\\nVIX 17.87 -0.67%\\n\\nGold 4700.30 -0.14%\\n\\nBitcoin 79264.71 -0.43%\\n\\nCrude Oil 101.13 0.11%\\n\\nDollar Index 95.17 -0.02%\\n\\nKBW Nasdaq Bank Index 162.54 -1.26%\\n\\nS&P GSCI Index Spot 749.75 -0.95%\\n\\nThe Wall Street Journal\\n\\nSpecial OfferSign In\\n\\n   English Edition  Edition Use Alt + Down Arrow to expand.   \\n   Print Edition\\n   Video\\n   Audio\\n   Latest Headlines\\n   Puzzles\\n   More  More   \\n\\n   World\\n   Business\\n   U.S.\\n   Politics\\n   Economy\\n   Tech\\n   Markets & Finance\\n   Opinion\\n   Free Expression\\n   Arts\\n   Lifestyle\\n   Real Estate\\n   Personal Finance\\n   Health\\n   Style\\n   Sports", "raw_content": null}, {"url": "https://www.wsj.com/livecoverage/iran-war-trump-news", "title": "Iran War Live News Updates: U.S. Strikes Tankers Trying to Break Blockade, Awaits Iran\'s Peace Plan Response - WSJ", "score": 0.96267307, "published_date": "Fri, 08 May 2026 19:37:51 GMT", "content": "Despite the exchanges of fire between the U.S. and Iran this week, President Trump has said the cease-fire with Iran was still in effect. Shipping in the Strait of Hormuz has slowed sharply amid the turmoil, with no commercial vessels run by registered shipping companies crossing the strait for days, according to S&P Global Intelligence data.\\n\\nHere’s what else to know:\\n\\nThe U.S. is working to prevent Iran from using empty tankers to serve as floating oil storage as it enforces a blockade of Iranian ports\\n\\nWashington will facilitate a new round of talks between Israel and Lebanon next week.\\n\\nU.S.-Israeli strikes have set back Iran’s ability to build a nuclear weapon, a watchdog said.\\n\\nImage 2\\n\\nImage 3Image 4Image 5Image 6Image 7Image 8Image 9 [...] ## Rubio said Washington expects a response from Tehran on Friday to a proposed memorandum of understanding to end the war\\n\\nImage 1\\n\\nU.S. Navy\\n\\nU.S. forces struck two empty Iranian-flagged oil tankers attempting to circumvent the American naval blockade against Iranian ports on Friday, according to Central Command. The airstrikes came as Secretary of State Marco Rubio warned that the U.S. would fire on any Iranian vessels that threaten American ships and said Washington was expecting a response from Tehran about a proposed memorandum of understanding to end the war. Iranian officials have indicated to mediators that they would likely agree on the MoU but with several modifications. [...] # Iran War Live News Updates: U.S. Strikes Tankers Trying to Break Blockade, Awaits Iran\'s Peace Plan Response\\n\\nDJIA 49609.16 0.02%\\n\\nS&P 500 7398.93 0.84%\\n\\nNasdaq 26247.08 1.71%\\n\\nRussell 2000 2861.21 0.76%\\n\\nU.S. 10 Yr 25/32 4.359%\\n\\nVIX 17.19 0.64%\\n\\nGold 4723.70 0.27%\\n\\nBitcoin 80344.44 0.25%\\n\\nCrude Oil 94.68 -0.14%\\n\\nDollar Index 94.68 -0.33%\\n\\nKBW Nasdaq Bank Index 165.78 -0.79%\\n\\nS&P GSCI Index Spot 732.21 0.62%\\n\\nThe Wall Street Journal\\n\\nLimited-Time SaleSign In\\n\\n   English Edition  Edition Use Alt + Down Arrow to expand.   \\n   Print Edition\\n   Video\\n   Audio\\n   Latest Headlines\\n   Puzzles\\n   More  More", "raw_content": null}, {"url": "https://www.jpost.com/middle-east/iran-news/2026-05-11/live-updates-895743", "title": "Live Updates: Latest from Israel, Iran, and Middle East - The Jerusalem Post", "score": 0.891055, "published_date": "Mon, 11 May 2026 00:00:05 GMT", "content": "Ayatollah Ali Khamenei was killed by an Israeli strike on a Tehran bunker. Iran’s defense minister and several IRGC generals were also killed in the largest-ever aerial attack by the IAF\\n\\nIran retaliated by firing across the Middle East at Gulf nations and US military bases in the region\\n\\nTwelve IDF soldiers and 23 civilians have been killed, and at least 7,693 more injured in ballistic missile attacks across Israel since February 28 • 13 US soldiers killed, CENTCOM says\\n\\nA ceasefire deal was announced on April 7 and went into effect on April 8.\\n\\n### See more on\\n\\nHezbollah|\\n\\nIDF|\\n\\nIran|\\n\\nLebanon|\\n\\nLebanon War|\\n\\nhezbollah iran|\\n\\nhezbollah lebanon|\\n\\nIDF drone|\\n\\niran israel war|\\n\\niran negotiations|\\n\\nIran nuclear talks|\\n\\nIsrael Iran War|\\n\\nus iran|\\n\\ndrone|\\n\\nmissiles|\\n\\nLive Updates| [...] Go to the full article > >\\n\\n## Trump calls latest Iranian proposal to end war \'totally unacceptable\'\\n\\n### The Iranian proposal stressed the need to end the war on all fronts and to lift sanctions on Tehran, Iran\'s semi-official Tasnim news agency reported, citing an informed source.\\n\\nByTOBIAS HOLCMAN, REUTERS\\n\\nUS President Donald Trump speaks to the media before boarding Air Force One for departure to Miami, Florida, at Palm Beach International Airport in West Palm Beach, Florida, US, May 2, 2026.\\n\\n(photo credit: REUTERS/Nathan Howard)\\n\\nGo to the full article > >\\n\\n## Hezbollah strikes Iron Dome battery with FPV drone [...] # Live Updates: Trump calls latest Iranian proposal \'unacceptable,\' Hezbollah drone strikes Iron Dome battery\\n\\n## Trump\'s visit to China will focus on Iranian issue • IDF, US ran covert rescue ops. from Iraq during Iran war • War with Iran not over, enriched uranium needs to be removed, Netanyahu says\\n\\nUS President Donald Trump speaks during an event to mark Military Mother\'s Day, in the East Room at the White House in Washington, DC, US, May 6, 2026.\\n\\n(photo credit: Reuters/Kylie Cooper)\\n\\n## War with Iran not over, enriched uranium needs to be removed, Netanyahu tells 60 Minutes\\n\\n### When asked how Iran\'s supply of enriched uranium could be removed from the country, Netanyahu replied simply that \\"You go in, and you take it out.\\"\\n\\nByTZVI JASPER", "raw_content": null}, {"url": "https://www.hindustantimes.com/world-news/us-iran-war-news-live-donald-trump-china-strait-hormuz-petrol-diesel-price-today-israel-lebanon-ceasefire-uae-araghchi-101778892776637.html", "title": "US-Iran war highlights: Israel, Hezbollah fire attacks, strikes day after Lebanon truce extended | World News - Hindustan Times", "score": 0.8311431, "published_date": "Sat, 16 May 2026 01:24:28 GMT", "content": "US-Iran war LIVE updates: The United Arab Emirates on Saturday said its decision to exit from the OPEC was a sovereign and strategic choice based on a comprehensive assessment of its production policy and future capabilities, Reuters news agency reported.\\n\\nUAE energy minister Suhail Al Mazrouei said in a post on X that the move was not politically motivated and did not reflect divisions with partners.\\n\\nMay 16, 2026 2:44:43 PMIST\\n\\n## US-Iran war LIVE updates: UAE rejects Iran war role claims\\n\\nUS-Iran war LIVE updates: The United Arab Emirates rejected \\"attempts to justify Iranian terrorist attacks\\" after Tehran accused the wealthy Gulf state of playing an active role in the war. [...] US-Iran war LIVE updates: The Israeli military said on Saturday it had launched fresh strikes targeting Hezbollah positions across southern Lebanon, a day after the two sides agreed to extend an existing ceasefire. “The IDF has begun striking Hezbollah infrastructure sites in several areas in southern Lebanon,” the military said in a statement.\\n\\nMay 16, 2026 2:06:51 PMIST\\n\\n## US-Iran war LIVE updates: Trump warns Taiwan against independence, Taipei responds\\n\\nUS-Iran war LIVE updates: Taiwan on Saturday declared itself an “independent” and sovereign nation after US President Donald Trump warned the island against formally declaring independence during his visit to China. [...] Lebanon\'s state-run National News Agency (NNA) reported strikes on more than two dozen villages on Saturday, including one more than 50 kilometres (30 miles) from the border.\\n\\nMay 16, 2026 10:19:44 PMIST\\n\\n## US-Iran war LIVE updates: Pak minister arrives in Tehran to ‘facilitate’ peace talks, says state media\\n\\nUS-Iran war LIVE updates: Pakistan\'s interior minister Mohsin Naqvi arrived in Tehran on Saturday \\"to facilitate\\" the peace talks between Iran and the United States, Iran\'s state media reported, according to AFP.\\n\\n“Mohsin Naqvi arrived today in the Islamic Republic of Iran on an official two-day visit as part of Pakistan\'s ongoing efforts to facilitate talks and promote regional peace,” Tasnim news agency reported.\\n\\nMay 16, 2026 9:07:48 PMIST", "raw_content": null}, {"url": "https://www.nbcnews.com/video/stalemate-lingers-as-iran-waits-to-respond-to-latest-u-s-peace-proposal-263055941549", "title": "Stalemate lingers as Iran waits to respond to latest U.S. peace proposal - NBC News", "score": 0.67405045, "published_date": "Sat, 09 May 2026 22:49:58 GMT", "content": "00:57\\n\\n Copied\\n\\nNo breakthroughs to end the war with Iran have been made as the U.S. awaits Iran’s response to its latest peace proposal. Despite exchanging fire this week, both sides are maintaining that the month-long ceasefire still holds. NBC News’ Julie Tsikrin has the latest.May 9, 2026\\n\\nGet more newson\\n\\n UP NEXT\\n\\n  ## Great Americans: A conversation with Brad Paisley\\n\\n  04:07\\n ## Deadly volcano eruption in Indonesia\\n\\n  01:20\\n ## Pentagon releases declassified UFO files\\n\\n  02:03\\n ## U.S. and Iran exchange fire as Trump says ceasefire still holds\\n\\n  01:50\\n ## Hiker’s body found after suspected bear attack in Glacier National Park\\n\\n  01:17\\n ## Evidence of possible remains found in the Kristin Smart case\\n\\n  01:30\\n\\nGet more newson\\n\\n## Best of NBC News\\n\\nPlay All\\n\\n## Weekend Nightly", "raw_content": null}], "response_time": 0.93, "request_id": "16c1fc86-d7ca-4f9b-9fbe-26a8806ba06e"}', name='tavily_search', id='5d511106-7c9c-4a6d-9599-fc234231b0fd', tool_call_id='tavily_search:0'), AIMessage(content='', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 49, 'prompt_tokens': 4614, 'total_tokens': 4663, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 1536}, 'cached_tokens': 1536}, 'model_provider': 'openai', 'model_name': 'kimi-k2.5', 'system_fingerprint': None, 'id': 'chatcmpl-6a12d55f6fc276ab303e4b0d', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--019e5991-7c32-7c42-a45b-3f3802283066-0', tool_calls=[{'name': 'tavily_search', 'args': {'query': 'US Iran war 2026 latest news updates May', 'search_depth': 'advanced', 'time_range': 'month', 'topic': 'news'}, 'id': 'tavily_search:1', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 4614, 'output_tokens': 49, 'total_tokens': 4663, 'input_token_details': {'cache_read': 1536}, 'output_token_details': {}}), ToolMessage(content='{"query": "US Iran war 2026 latest news updates May", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "https://understandingwar.org/research/middle-east/iran-update-special-report-may-8-2026/", "title": "Iran Update Special Report, May 8, 2026 - Institute for the Study of War", "score": 0.9393874, "published_date": "Fri, 08 May 2026 23:18:10 GMT", "content": "The United States has continued efforts to maintain an effective naval blockade against Iran as Iran continues to assert long-term sovereignty over the Strait of Hormuz. US Central Command (CENTCOM) announced on May 8 that US naval forces have redirected 57 vessels since the blockade began on April 13. CENTCOM also reported that US forces disabled two Iranian tankers, Sea Star III and Sevda, on May 8 after both vessels attempted to enter an Iranian port along the Gulf of Oman. Iran would have presumably used the tankers to expand its floating oil storage capacity as the US blockade continues to strain the regime’s ability to store crude oil. Iran has already reactivated older tankers and repurposed empty vessels as temporary offshore oil storage to alleviate some of the mounting pressure [...] The United States and Iran remain divided over key issues, particularly Iran’s nuclear program, the status of Iran’s highly enriched uranium (HEU) stockpile, and Iranian efforts to assert sovereignty over the Strait of Hormuz. Iranian Foreign Affairs Ministry Spokesperson Esmail Baghaei stated on May 8 that Iran is still reviewing the recent US proposal and will respond once Iran reaches a final decision. Baghaei also stated on May 7 that Iran has decided to pursue negotiations in a “new format” through Pakistan and added that Iranian decisionmakers chose not to enter nuclear talks “for the time being.” US President Donald Trump warned on May 8 that the United States could conduct renewed attacks if Iran does not quickly accept a deal. Secretary of State Marco Rubio stated on May 8 that [...] 1. Iranian officials continue to frame control over the Strait of Hormuz as a key strategic interest and a critical component of long-term Iranian deterrence. Iran likely views control over the strait as essential to restoring deterrence against the United States and Israel following the degradation of its other forms of deterrence.\\n2. The United States and Iran remain divided over key issues, particularly Iran’s nuclear program, the status of Iran’s highly enriched uranium (HEU) stockpile, and Iranian efforts to assert sovereignty over the Strait of Hormuz.\\n3. The United States has continued efforts to maintain an effective naval blockade against Iran as Iran continues to assert long-term sovereignty over the Strait of Hormuz.", "raw_content": null}, {"url": "https://www.hindustantimes.com/world-news/us-iran-war-news-live-donald-trump-china-strait-hormuz-petrol-diesel-price-today-israel-lebanon-ceasefire-uae-araghchi-101778892776637.html", "title": "US-Iran war highlights: Israel, Hezbollah fire attacks, strikes day after Lebanon truce extended | World News - Hindustan Times", "score": 0.9365236, "published_date": "Sat, 16 May 2026 01:24:28 GMT", "content": "May 16, 2026 6:47:00 PMIST\\n\\n## US-Iran war LIVE updates: Iran interior minister holds talks with with Pak counterpart, discusses prospects for resuming peace negotiations with US\\n\\nUS-Iran war LIVE updates: Iran interior minister held talks with his Pakistani counterpart on Saturday.\\n\\nThe two allegedly held detailed discussions on Iran-Pakistan relations and the prospects of Tehran resuming peace talks with the United States, Reuters news agency reported.\\n\\nMay 16, 2026 6:12:36 PMIST\\n\\n## US-Iran war LIVE updates: Fire breaks out in engine oil plant in Iran\\n\\nUS-Iran war LIVE updates: A fire broke out at an engine oil plant in northwestern Iran on Saturday, Reuters news agency reported citing Mehr news.\\n\\nThere were no immediate reports of casualties in the incident. [...] May 16, 2026 7:39:53 AMIST\\n\\n## US-Iran LIVE Updates: India hikes export duty on petrol, cuts levy on diesel, aviation turbine fuel\\n\\nUS-Iran LIVE Updates: India raised its export duty on petrol and cut levies on diesel and aviation fuel on Friday, the Ministry of Finance said in a statement. This notification from the government comes within a day of raising retail fuel prices for the first time since the war between the US, Israel and Iran broke out in February.\\n\\nCiting the government notification, news agency PTI reported that the duty on exports of petrol has been raised to ₹3 per litre, a first since the Iran war started.\\n\\nMeanwhile, the duty on export of diesel has been reduced to ₹16.5 per litre, from ₹23 per litre, and aviation turbine fuel to ₹16 per litre from ₹33 per litre. [...] May 16, 2026 7:18:27 AMIST\\n\\n## US-Iran LIVE Updates: Iran foreign minister flags ‘lack of trust’ amid talks with US\\n\\nUS-Iran LIVE Updates: Iran’s foreign minister said a lack of trust is the biggest obstacle in negotiations to end the war with the US., saying Friday that Tehran would be open to diplomatic help, particularly from China, to help ease tensions.\\n\\nAbbas Araghchi added that contradictory messages have “made us reluctant about the real intentions of Americans.”\\n\\n“We are in doubt about their seriousness,” he told reporters in New Delhi, adding that negotiations would move forward if Washington was ready for a “fair and balanced deal.”\\n\\nMay 16, 2026 7:06:02 AMIST\\n\\n## US-Iran LIVE Updates: UAE rejects Iran war role claims", "raw_content": null}, {"url": "https://understandingwar.org/research/middle-east/iran-update-special-report-may-17-2026/", "title": "Iran Update Special Report, May 17, 2026 - Institute for the Study of War", "score": 0.92794675, "published_date": "Sun, 17 May 2026 19:25:33 GMT", "content": "## Toplines\\n\\nThe United States and Iran continue to pursue fundamentally incompatible negotiating positions. Unspecified sources familiar with the US response to the latest Iranian counterproposal told Islamic Revolutionary Guards Corps (IRGC)-affiliated Fars News Agency on May 17 that the United States laid out its five main conditions in response to the latest Iranian proposal. The reported US response rejected war reparations, required Iran to transfer 400 kilograms of uranium to the United States, limited Iran to one operational nuclear facility, and refused to release more than 25 percent of frozen Iranian assets. The war would only end if negotiations were successful, and there would be no guarantees that the United States or Israel would not attack Iran in the future. [...] Iran sent its most recent counterproposal to the United States on May 10, which US President Donald Trump later called “totally unacceptable.” IRGC Baghiyatollah Sociocultural Headquarters Commander and former IRGC commander Major General Mohammad Ali Jafari stated on May 11 that Iran will not enter negotiations with the United States until the United States accepts Iran’s terms. Jafari stated that Iran’s terms include an end to the war on “all fronts,” the lifting of sanctions, the release of frozen Iranian assets, compensation for war-related damages, and recognition of Iran’s sovereignty over the Strait of Hormuz. [...] 1. The United States and Iran continue to pursue fundamentally incompatible negotiating positions. Unspecified sources familiar with the US response to the latest Iranian counterproposal told Islamic Revolutionary Guards Corps (IRGC)-affiliated Fars News Agency on May 17 that the United States laid out its five main conditions in response to the latest Iranian proposal.", "raw_content": null}, {"url": "https://understandingwar.org/research/middle-east/iran-update-special-report-may-20-2026/", "title": "Iran Update Special Report, May 20, 2026 - Institute for the Study of War", "score": 0.9242583, "published_date": "Wed, 20 May 2026 22:38:10 GMT", "content": "The United States continues to interdict vessels violating the US blockade of Iranian ports. Three US officials told the Wall Street Journal that US forces seized a US-sanctioned, Iranian-linked oil tanker, the M/T Skywave (IMO: 9328716),between May 19 and 20. Commercially available ship tracking data showed that the Skywave transited the Strait of Malacca on May 14, before returning to the Indian Ocean on May 19, presumably after being seized by US forces. It is unclear where the vessel was before May 14. The United States Treasury Department sanctioned the Skywave, under a different vessel name, on March 13 for its role in transporting Iranian oil. US Central Command (CENTCOM) reported on May 20 that US Marines boarded the Iranian-flagged oil tanker M/T Celestial Sea (IMO: 9397030) on [...] The IRGC threatened on May 20 to expand the war “far beyond the region” if US-Israeli strikes on Iran resume. This threat almost certainly seeks to deter renewed US-Israeli strikes but could also reflect IRGC planning for a potential future conflict. IRGC-affiliated media previously used similar messaging ahead of the most recent round of conflict to try to exploit US concerns about a protracted regional war to try to deter a US attack. IRGC-affiliated media Tasnim News warned on February 2 that a US attack on Iran could trigger a regional war because Iran could retaliate by attacking Israel, international shipping, or US bases in regional countries. Iran may similarly calculate that threats of a conflict extending “beyond the region” could deter against renewed US-Israeli strikes or [...] oil tanker M/T Celestial Sea (IMO: 9397030) on May 20 on suspicion of attempting to reach an Iranian port. US forces released the vessel after searching it and directing it to change course. CENTCOM reported that US forces redirected 91 vessels and disabled four others since the start of the blockade on Iranian ports on April 13.", "raw_content": null}, {"url": "https://understandingwar.org/research/middle-east/iran-update-special-report-april-29-2026/", "title": "Iran Update Special Report, April 29, 2026 - Institute for the Study of War", "score": 0.78670514, "published_date": "Thu, 30 Apr 2026 01:25:24 GMT", "content": "Israeli leaders have asked the United States not to extend the temporary ceasefire in Lebanon past its scheduled expiration on May 17, according to Israeli media. Unnamed Israeli political officials argued on April 29 that the IDF’s current restraint under the ceasefire is weakening Israeli deterrence against Hezbollah, whose attacks are “undermining the prospects of reaching agreements” with the Lebanese government. The Israeli government asked US President Donald Trump to approve of expanded IDF operations in Lebanon if negotiations with the Lebanese government do not “bear fruit” by May 17. IDF Chief of Staff Eyal Zamir said during a visit to Israeli soldiers in Taybeh, Marjaayoun District, that there is “no ceasefire” in Lebanon and cited Hezbollah’s attacks targeting Israeli soldiers [...] Skip to contentSkip to Content\\n\\nMiddle East\\n\\nIran & Proxies\\n\\nIran Update\\n\\n# Iran Update Special Report, April 29, 2026\\n\\n###### April 29, 2026\\n\\n## Jump to\\n\\nPrevious\\n\\n# Iran Update Special Report, April 29, 2026\\n\\nThe Institute for the Study of War (ISW) and The Critical Threats Project (CTP) at the American Enterprise Institute are publishing daily updates to provide analysis on the war with Iran. The updates focus on US and Israeli strikes on Iran and the Axis of Resistance’s response to the strikes. The updates cover events from the past 24-hour period. [...] 5. Iran has responded to growing pressure on the economy caused by the war, sanctions, and the blockade by shoring up internal unity, presumably as an initial step in a more comprehensive internal security plan.\\n6. Iran is reportedly exploring alternative methods to circumvent the US blockade of its ports, which is a suboptimal way to alleviate some of the economic pressure, including seizing US commercial vessels or Israeli-linked vessels, relying on trade with Russia and other countries bordering the Caspian Sea, or even the Houthis attacking vessels through the Bab el Mandeb Strait.\\n7. Iran has almost certainly used the ceasefire to reorganize and regenerate its missile and drone forces, which were badly degraded on the operational level prior to the ceasefire.", "raw_content": null}], "response_time": 1.01, "request_id": "dce37488-ac5b-49cb-89e6-507cf5426674"}', name='tavily_search', id='b994ebab-f87c-481e-be05-39aa79156562', tool_call_id='tavily_search:1'), AIMessage(content='', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 842, 'prompt_tokens': 7497, 'total_tokens': 8339, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 4608}, 'cached_tokens': 4608}, 'model_provider': 'openai', 'model_name': 'kimi-k2.5', 'system_fingerprint': None, 'id': 'chatcmpl-6a12d5646fc276ab303e4b0f', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--019e5991-8eb9-7eb2-907c-74b3ee9fd144-0', tool_calls=[{'name': 'AgentResponse', 'args': {'answer': '根据最新搜索结果，以下是美伊战争的最新动态（截至2026年5月）：\n\n## 当前局势概述\n\n**战争背景**：美伊战争于2026年2月28日爆发，以色列空袭伊朗德黑兰地堡，杀死了伊朗最高领袖哈梅内伊、国防部长及多名伊斯兰革命卫队将领。伊朗随后对中东地区的美军基地和海湾国家进行报复性打击。4月7日宣布停火，4月8日正式生效。\n\n## 最新动态（2026年5月）\n\n### 1. 谈判僵局\n- 美国和伊朗在关键问题上立场根本对立，特别是伊朗核计划、高浓缩铀库存地位以及霍尔木兹海峡主权问题\n- 伊朗于5月10日提出最新反提案，特朗普总统称该提案"完全不可接受"\n- 美国提出五项主要条件：拒绝战争赔偿、要求伊朗转移400公斤铀、限制伊朗仅运营一个核设施、拒绝解冻超过25%的伊朗资产，且不保证未来不攻击伊朗\n\n### 2. 海上封锁持续\n- 美国自4月13日起对伊朗港口实施海军封锁\n- 截至5月20日，美军已拦截91艘船只，并击毁4艘\n- 5月19-20日，美军扣押了一艘与伊朗有关联的油轮M/T Skywave\n- 5月20日，美军登上一艘伊朗油轮M/T Celestial Sea进行检查后放行\n\n### 3. 地区局势\n- **黎巴嫩**：以色列与真主党的停火协议延长至5月17日后，双方仍持续交火\n- **阿联酋**：否认参与战争，并宣布退出OPEC\n- **巴基斯坦**：内政部长于5月16日抵达德黑兰，协助促进美伊和平谈判\n\n### 4. 各方表态\n- **伊朗**：外长阿拉格齐表示"缺乏信任"是谈判最大障碍，对美国真实意图存疑\n- **特朗普**：警告若伊朗不尽快接受协议，美国可能发动新一轮攻击；同时表示停火仍然有效\n- **以色列**：总理内塔尼亚胡表示战争尚未结束，需要移除伊朗的高浓缩铀\n\n### 5. 经济影响\n- 霍尔木兹海峡航运急剧放缓，全球20%的石油运输受影响\n- 印度于5月16日调整燃油出口关税\n- 伊朗正面临严重的石油储存压力，被迫将老旧油轮用作浮动储油设施\n\n## 关键分歧点\n1. 伊朗要求结束"所有战线"的战争、解除制裁、释放被冻结资产、战争赔偿及承认对霍尔木兹海峡的主权\n2. 美国拒绝接受伊朗的反提案条件\n3. 双方在核问题上的立场差距巨大\n\n目前局势仍处于脆弱的停火状态，但谈判进展缓慢，战争风险依然存在。', 'sources': [{'url': 'https://www.wsj.com/livecoverage/iran-us-china-news-2026'}, {'url': 'https://www.wsj.com/livecoverage/iran-war-trump-news'}, {'url': 'https://www.jpost.com/middle-east/iran-news/2026-05-11/live-updates-895743'}, {'url': 'https://www.hindustantimes.com/world-news/us-iran-war-news-live-donald-trump-china-strait-hormuz-petrol-diesel-price-today-israel-lebanon-ceasefire-uae-araghchi-101778892776637.html'}, {'url': 'https://understandingwar.org/research/middle-east/iran-update-special-report-may-8-2026/'}, {'url': 'https://understandingwar.org/research/middle-east/iran-update-special-report-may-17-2026/'}, {'url': 'https://understandingwar.org/research/middle-east/iran-update-special-report-may-20-2026/'}]}, 'id': 'AgentResponse:2', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 7497, 'output_tokens': 842, 'total_tokens': 8339, 'input_token_details': {'cache_read': 4608}, 'output_token_details': {}}), ToolMessage(content='Returning structured response: answer=\'根据最新搜索结果，以下是美伊战争的最新动态（截至2026年5月）：\\n\\n## 当前局势概述\\n\\n**战争背景**：美伊战争于2026年2月28日爆发，以色列空袭伊朗德黑兰地堡，杀死了伊朗最高领袖哈梅内伊、国防部长及多名伊斯兰革命卫队将领。伊朗随后对中东地区的美军基地和海湾国家进行报复性打击。4月7日宣布停火，4月8日正式生效。\\n\\n## 最新动态（2026年5月）\\n\\n### 1. 谈判僵局\\n- 美国和伊朗在关键问题上立场根本对立，特别是伊朗核计划、高浓缩铀库存地位以及霍尔木兹海峡主权问题\\n- 伊朗于5月10日提出最新反提案，特朗普总统称该提案"完全不可接受"\\n- 美国提出五项主要条件：拒绝战争赔偿、要求伊朗转移400公斤铀、限制伊朗仅运营一个核设施、拒绝解冻超过25%的伊朗资产，且不保证未来不攻击伊朗\\n\\n### 2. 海上封锁持续\\n- 美国自4月13日起对伊朗港口实施海军封锁\\n- 截至5月20日，美军已拦截91艘船只，并击毁4艘\\n- 5月19-20日，美军扣押了一艘与伊朗有关联的油轮M/T Skywave\\n- 5月20日，美军登上一艘伊朗油轮M/T Celestial Sea进行检查后放行\\n\\n### 3. 地区局势\\n- **黎巴嫩**：以色列与真主党的停火协议延长至5月17日后，双方仍持续交火\\n- **阿联酋**：否认参与战争，并宣布退出OPEC\\n- **巴基斯坦**：内政部长于5月16日抵达德黑兰，协助促进美伊和平谈判\\n\\n### 4. 各方表态\\n- **伊朗**：外长阿拉格齐表示"缺乏信任"是谈判最大障碍，对美国真实意图存疑\\n- **特朗普**：警告若伊朗不尽快接受协议，美国可能发动新一轮攻击；同时表示停火仍然有效\\n- **以色列**：总理内塔尼亚胡表示战争尚未结束，需要移除伊朗的高浓缩铀\\n\\n### 5. 经济影响\\n- 霍尔木兹海峡航运急剧放缓，全球20%的石油运输受影响\\n- 印度于5月16日调整燃油出口关税\\n- 伊朗正面临严重的石油储存压力，被迫将老旧油轮用作浮动储油设施\\n\\n## 关键分歧点\\n1. 伊朗要求结束"所有战线"的战争、解除制裁、释放被冻结资产、战争赔偿及承认对霍尔木兹海峡的主权\\n2. 美国拒绝接受伊朗的反提案条件\\n3. 双方在核问题上的立场差距巨大\\n\\n目前局势仍处于脆弱的停火状态，但谈判进展缓慢，战争风险依然存在。\' sources=[Source(url=\'https://www.wsj.com/livecoverage/iran-us-china-news-2026\'), Source(url=\'https://www.wsj.com/livecoverage/iran-war-trump-news\'), Source(url=\'https://www.jpost.com/middle-east/iran-news/2026-05-11/live-updates-895743\'), Source(url=\'https://www.hindustantimes.com/world-news/us-iran-war-news-live-donald-trump-china-strait-hormuz-petrol-diesel-price-today-israel-lebanon-ceasefire-uae-araghchi-101778892776637.html\'), Source(url=\'https://understandingwar.org/research/middle-east/iran-update-special-report-may-8-2026/\'), Source(url=\'https://understandingwar.org/research/middle-east/iran-update-special-report-may-17-2026/\'), Source(url=\'https://understandingwar.org/research/middle-east/iran-update-special-report-may-20-2026/\')]', name='AgentResponse', id='a0b1e5e6-74cc-4c32-a382-8566e419c2af', tool_call_id='AgentResponse:2')], 'structured_response': AgentResponse(answer='根据最新搜索结果，以下是美伊战争的最新动态（截至2026年5月）：\n\n## 当前局势概述\n\n**战争背景**：美伊战争于2026年2月28日爆发，以色列空袭伊朗德黑兰地堡，杀死了伊朗最高领袖哈梅内伊、国防部长及多名伊斯兰革命卫队将领。伊朗随后对中东地区的美军基地和海湾国家进行报复性打击。4月7日宣布停火，4月8日正式生效。\n\n## 最新动态（2026年5月）\n\n### 1. 谈判僵局\n- 美国和伊朗在关键问题上立场根本对立，特别是伊朗核计划、高浓缩铀库存地位以及霍尔木兹海峡主权问题\n- 伊朗于5月10日提出最新反提案，特朗普总统称该提案"完全不可接受"\n- 美国提出五项主要条件：拒绝战争赔偿、要求伊朗转移400公斤铀、限制伊朗仅运营一个核设施、拒绝解冻超过25%的伊朗资产，且不保证未来不攻击伊朗\n\n### 2. 海上封锁持续\n- 美国自4月13日起对伊朗港口实施海军封锁\n- 截至5月20日，美军已拦截91艘船只，并击毁4艘\n- 5月19-20日，美军扣押了一艘与伊朗有关联的油轮M/T Skywave\n- 5月20日，美军登上一艘伊朗油轮M/T Celestial Sea进行检查后放行\n\n### 3. 地区局势\n- **黎巴嫩**：以色列与真主党的停火协议延长至5月17日后，双方仍持续交火\n- **阿联酋**：否认参与战争，并宣布退出OPEC\n- **巴基斯坦**：内政部长于5月16日抵达德黑兰，协助促进美伊和平谈判\n\n### 4. 各方表态\n- **伊朗**：外长阿拉格齐表示"缺乏信任"是谈判最大障碍，对美国真实意图存疑\n- **特朗普**：警告若伊朗不尽快接受协议，美国可能发动新一轮攻击；同时表示停火仍然有效\n- **以色列**：总理内塔尼亚胡表示战争尚未结束，需要移除伊朗的高浓缩铀\n\n### 5. 经济影响\n- 霍尔木兹海峡航运急剧放缓，全球20%的石油运输受影响\n- 印度于5月16日调整燃油出口关税\n- 伊朗正面临严重的石油储存压力，被迫将老旧油轮用作浮动储油设施\n\n## 关键分歧点\n1. 伊朗要求结束"所有战线"的战争、解除制裁、释放被冻结资产、战争赔偿及承认对霍尔木兹海峡的主权\n2. 美国拒绝接受伊朗的反提案条件\n3. 双方在核问题上的立场差距巨大\n\n目前局势仍处于脆弱的停火状态，但谈判进展缓慢，战争风险依然存在。', sources=[Source(url='https://www.wsj.com/livecoverage/iran-us-china-news-2026'), Source(url='https://www.wsj.com/livecoverage/iran-war-trump-news'), Source(url='https://www.jpost.com/middle-east/iran-news/2026-05-11/live-updates-895743'), Source(url='https://www.hindustantimes.com/world-news/us-iran-war-news-live-donald-trump-china-strait-hormuz-petrol-diesel-price-today-israel-lebanon-ceasefire-uae-araghchi-101778892776637.html'), Source(url='https://understandingwar.org/research/middle-east/iran-update-special-report-may-8-2026/'), Source(url='https://understandingwar.org/research/middle-east/iran-update-special-report-may-17-2026/'), Source(url='https://understandingwar.org/research/middle-east/iran-update-special-report-may-20-2026/')])}

## output explain

这个长字典实际上是 LangChain 智能体的“思维与行动轨迹”。下面为你把这个复杂的 Output 拆解成人类可读的**五个核心步骤**：

---

## 🧭 智能体执行轨迹拆解

整个运行过程像一个精密的流水线，总共交替进行了多轮对话和工具调用：

### 第一步：用户下达指令 (`HumanMessage`)

* **内容**：你输入的 `"搜索美伊战争最新动态"`。

### 第二步：智能体第一次搜网 (`AIMessage` -> `ToolMessage`)

* **决策**：Kimi 模型意识到自己不知道最新动态，于是决定调用 `tavily_search` 工具。
* **关键词**：`"美伊战争 最新动态 最新消息"`。
* **结果**：Tavily 返回了《华尔街日报》、《耶路撒冷邮报》等媒体的最新新闻切片（内容包含哈梅内伊遇袭、4月停火协议等）。

### 第三步：智能体觉得不够，进行第二次搜网 (`AIMessage` -> `ToolMessage`)

* **决策**：Kimi 读完第一批资料后，发现这些信息多集中在4月和5月初，为了确保准确性，它**自动发起了第二次更精确的英文搜索**。
* **关键词**：`"US Iran war 2026 latest news updates May"`。
* **结果**：Tavily 返回了来自美国战争研究所（ISW）等更硬核的5月中下旬报告（内容包含海上封锁、美军扣押油轮、5月17日谈判僵局等详情）。

### 第四步：智能体组装结构化数据 (`AIMessage`)

* **决策**：收集到足够信息后，Kimi 开始履行你定义的 `response_format=AgentResponse` 约定。
* **实现方式**：在底层，它把你的 Pydantic 结构当成了一个名叫 `AgentResponse` 的“特殊工具”来进行调用，并在参数中填满了它总结出来的 `answer`（详细的Markdown报告）和 `sources`（7个权威网站链接）。

### 第五步：LangChain 交付成品 (`structured_response`)

* **结果**：这是输出字典的最后一个字段。LangChain 拦截了第四步的特殊调用，直接帮你实例化好了 Python 的 Pydantic 对象：
```python
AgentResponse(answer='根据最新搜索结果...', sources=[Source(url='...'), ...])

```



```

---

## 💡 关键字段解读（写代码时怎么拿数据）

当你拿到这个 `result` 字典后，在代码中可以通过以下方式提取你需要的信息：

1.  **获取纯净的结构化成品（最常用）**：
    ```python
    final_object = result["structured_response"]
    print(final_object.answer)   # 打印完美格式化的新闻大纲
    print(final_object.sources)  # 拿到 Source(url=...) 的 Python 列表

```

2. **查看 Token 消耗（算账/优化时看）**：
在 `usage_metadata` 中可以看到：
* `input_tokens`: 7497（因为注入了大量网页搜索结果，上下文变长了）。
* `cache_read`: 4608（Kimi 的缓存机制生效了，帮你**省了这部分 Token 的钱**，好评！）。
* `output_tokens`: 842（最终模型吐出的字数）。



## 🎯 总结

你的代码现在不仅是一个“会联网的 GPT”，更是一个“能自己决定搜几次、搜完还能把网页清洗干净，最后规规矩矩填进 Python 标准对象里”的高级 AI 助手。方案一的兼容性修改非常成功！