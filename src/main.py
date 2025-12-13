import streamlit as st
from pipeline import run_thesis_pipeline

st.title("CorpusAgent")

# --- Session state setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# store full pipeline results per turn for debugging
if "debug_runs" not in st.session_state:
    st.session_state.debug_runs = []

# --- Show chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input ---
prompt = st.chat_input("Ask a question about the news corpus (2016–2021)...")

if prompt:
    # 1) Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2) Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3) Run your full thesis pipeline for this question
    with st.spinner("Running retrieval + NLP pipeline..."):
        pipeline_result = run_thesis_pipeline(prompt)

    final_answer = pipeline_result["final_answer"]

    # 4) Show assistant's answer to the user
    with st.chat_message("assistant"):
        st.markdown(final_answer)

    # 5) Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})

    # 6) Store pipeline internals for debugging
    st.session_state.debug_runs.append(pipeline_result)

# --- Debug window / backend pipeline view ---
st.markdown("---")
with st.expander("🔧 Pipeline debug (latest run)", expanded=False):
    if st.session_state.debug_runs:
        last = st.session_state.debug_runs[-1]

        st.write("### Question")
        st.write(last["question"])

        st.write("### Retrieval summary")
        retrieval = last["retrieval"]
        st.write(f"- Tool calls: {len(retrieval.get('tool_calls', []))}")
        st.write(f"- Search results: {len(retrieval.get('search_results', []))}")
        st.write(f"- Retrieved documents: {len(retrieval.get('documents', []))}")

        st.write("### NLP Plan")
        st.json(last.get("nlp_plan", {}))

        st.write("### Mocked NLP tool outputs (keys)")
        st.write(list(last.get("mocked_tool_outputs", {}).keys()))

        st.write("### Year-level summaries")
        st.json(last.get("year_summaries", {}))

    else:
        st.info("No pipeline runs yet. Ask a question to see debug info here.")
