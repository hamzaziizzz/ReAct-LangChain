from typing import Union, List

from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool
from langchain.tools.render import render_text_description
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of the given text by character count
    """
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(list_of_tools: List[BaseTool], name_of_tool: str) -> BaseTool:
    for tool_to_search in list_of_tools:
        if tool_to_search.name == name_of_tool:
            return tool_to_search
    raise ValueError(f"Tool {name_of_tool} not found")


if __name__ == '__main__':
    print("Hello ReAct Langchain")
    tools: List[BaseTool] = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join(t.name for t in tools)
    )

    # llm = ChatOpenAI(temperature=0, callbacks=[AgentCallbackHandler()])
    llm = ChatOllama(temperature=0, model="gemma3n:e4b", callbacks=[AgentCallbackHandler()])

    intermediate_steps = list()

    agent = {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
    } | prompt | llm.bind(stop=["\nObservation:", "Observation:"]) | ReActSingleInputOutputParser()

    # agent = agent | prompt | llm.bind(stop=["\nObservation:", "Observation:"]) | ReActSingleInputOutputParser()

    agent_step = None
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish, None] = agent.invoke(
            {
                "input": "What is the text length of 'DOG' in characters?",
                "agent_scratchpad": intermediate_steps
            }
        )
        # print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            print(f"Observation: {observation}")
            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
