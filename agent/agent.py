from models import model
from tools import calculate_right_refund , get_today_date
from tools import issue_refund_ticket,get_user_purchase_history
from dotenv import load_dotenv
from langchain.agents import create_agent
from prompts import agent_system_prompt
from objects import Answer
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, ToolMessage
from uuid import uuid4
import logging

EXIT_WORDS = {"exit", "quit", "bye", "goodbye", "q"}

load_dotenv()

agent = create_agent(
    model=model,
    system_prompt=agent_system_prompt,
    tools=[calculate_right_refund,get_user_purchase_history,issue_refund_ticket,get_today_date],
    response_format=Answer,
    checkpointer=InMemorySaver()
)

def run_conversation():
    config = {"configurable": {"thread_id": str(uuid4())}}

    text = "I would like to request a refund"
    print(text)
    
    question = HumanMessage(content=text)
    conversation = True
    seen = 0  # number of messages already processed in this thread

    while conversation:

        response = agent.invoke({"messages":question}, config=config)
        structured_response = response.get("structured_response")

        # messages produced during this turn only (the checkpointer keeps history)
        new_messages = response["messages"][seen:]
        seen = len(response["messages"])
        ticket_issued = any(
            isinstance(m, ToolMessage) and m.name == "issue_refund_ticket"
            for m in new_messages
        )

        if structured_response is None:
            # Small local models (Ollama ignores tool_choice) sometimes reply in
            # plain text instead of calling the Answer tool. Fall back to the
            # last message and keep the conversation going.
            logging.warning("model replied in plain text (no Answer tool call)")
            print(response["messages"][-1].content)
            conversation = True
        else:
            logging.info(f"structured answer, conversation={structured_response.conversation}")
            print(structured_response.answer)
            conversation = structured_response.conversation

        # the refund decision is final once a ticket is issued, whatever the model says
        if ticket_issued:
            conversation = False

        if conversation:
            user_input = input("reply: \n")
            words = set(user_input.lower().replace(",", " ").replace(".", " ").split())
            if words & EXIT_WORDS:
                break
            question = HumanMessage(content=user_input)

    print("the job is done!")

#run_conversation()