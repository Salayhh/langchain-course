import os

from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


llm = ChatOpenAI(base_url="https://api.moonshot.cn/v1",
                 api_key=os.getenv('MOONSHOT_API_KEY'),
                 model="kimi-k2.5",
                 extra_body={"thinking": {"type": "disabled"}})
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="搜索美伊战争最新动态"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
