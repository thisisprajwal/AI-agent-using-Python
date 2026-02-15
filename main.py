from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

# 1. Define Schema
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# 2. Setup Agent Prompt (Focused only on research)
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research assistant. Use tools to find detailed info."),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(llm=llm, prompt=agent_prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 3. Setup Structuring Prompt (Focused only on formatting)
structuring_prompt = ChatPromptTemplate.from_template(
    "You are a data formatter. Turn this research into the required JSON format.\n"
    "{format_instructions}\n\n"
    "Research Content: {research_content}"
)

# --- EXECUTION ---
query = input("What can I help you research? ")

# Step A: Run the Agent
print("\n--- Agent is researching ---")
agent_result = agent_executor.invoke({"query": query})
raw_text = agent_result["output"]

# Step B: Format the Result
print("\n--- Structuring Data ---")
structuring_chain = structuring_prompt | llm | parser

try:
    structured_response = structuring_chain.invoke({
        "research_content": raw_text,
        "format_instructions": parser.get_format_instructions()
    })
    print("\nFINAL STRUCTURED OUTPUT:")
    print(structured_response.model_dump_json(indent=2))
except Exception as e:
    print(f"Parsing failed. Raw research was: {raw_text}")
    print(f"Error: {e}")