import os
from datetime import datetime

from langchain.agents import create_agent
from langchain.messages import SystemMessage
from langchain.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.memory import InMemorySaver


@tool(name_or_callable="Hour", description="Pegar a hora atual")
def hora():
    return datetime.now().strftime("%H:%M:%S")

@tool(name_or_callable="Date", description="Pegar a data atual")
def data():
    return datetime.now().strftime("%d/%m/%Y")

@tool(name_or_callable="Day", description="Pegar o nome do dia atual")
def day():
    dias = [
        "segunda-feira",
        "terça-feira",
        "quarta-feira",
        "quinta-feira",
        "sexta-feira",
        "sábado",
        "domingo"
    ]

    agora = datetime.now()

    return dias[agora.weekday()]


def Agent():
    llm = ChatNVIDIA(
        model="z-ai/glm-5.2",
        api_key=os.environ["NVIDIA_API_KEY"],
        temperature=1,
        top_p=1,
        max_tokens=16384,
        seed=42,
        model_kwargs={"chat_template_kwargs": {"enable_thinking": True}},
    )

    agent = create_agent(
        model=llm,
        tools=[],
        checkpointer=InMemorySaver(),
        system_prompt=SystemMessage("Voce e um assistente pessoal chamado nexus, e voce so fala portugues, e voce executa a tarefa primeiro e depois fala com o usuario, ou fale com o usuario sobre o estado da tarefa?"),
    )
    return agent