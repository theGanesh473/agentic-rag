import streamlit as st
from rag_agent import agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

st.set_page_config(page_title="KRYOS AI", page_icon="🤖")
st.title("🤖 KRYOS AI")
st.caption("Built by Ganesh Sankar")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "tool":
        continue
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask me anything…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            history = []
            for m in st.session_state.messages[:-1]:
                if m["role"] == "user":
                    history.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    history.append(AIMessage(content=m["content"]))

            final_answer = ""
            tool_output  = ""

            for event in agent.stream(
                {"messages": history + [HumanMessage(content=user_input)]},
                stream_mode="values",
            ):
                last = event["messages"][-1]
                if isinstance(last, ToolMessage):
                    tool_output = last.content if isinstance(last.content, str) else str(last.content)
                elif isinstance(last, AIMessage) and last.content:
                    final_answer = last.content


        st.write(final_answer)

    st.session_state.messages.append({"role": "tool",      "content": tool_output})
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
