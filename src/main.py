import streamlit as st
from pipeline import run_thesis_pipeline

from viz.runs import list_pipeline_runs, fetch_pipeline_run
from viz.artifacts import list_pngs
from viz.schema_summary import schema_summary
from viz.planner import plan_visualizations
from viz.codegen import generate_viz_code
from viz.code_runner import run_generated_code

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

    # 3) Run full thesis pipeline for this question
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

        if "doc_selection_plan" in last:
            st.write("### Document selection plan")
            st.json(last["doc_selection_plan"])

        st.write("### NLP Plan")
        nlp_plan = last.get("nlp_plan") or {}
        st.json(nlp_plan)

        extra_tools = nlp_plan.get("desired_additional_tools", [])
        if extra_tools:
            st.write("### Additional NLP tools the planner would like to have")
            st.json(extra_tools)

        if "doc_selection_plan" in last:
            st.write("### Document selection plan")
            st.json(last["doc_selection_plan"])

        st.write("### Mocked NLP tool outputs (keys)")
        st.write(list(last.get("mocked_tool_outputs", {}).keys()))

        st.write("### Year-level summaries")
        st.json(last.get("year_summaries", {}))

    else:
        st.info("No pipeline runs yet. Ask a question to see debug info here.")

# --- Visualization playground ---
st.markdown("---")
with st.expander("📊 Visualization playground", expanded=False):
    runs = list_pipeline_runs(limit=30)
    st.write("### DB schema summary")
    st.code(schema_summary())

    if not runs:
        st.info("No runs in DB yet. Ask a question first.")
    else:
        labels = [
            f"{r['created_at']} | {str(r['run_id'])[:8]} | {str(r['question'])[:80]}"
            for r in runs
        ]
        idx = st.selectbox("Pick a run", range(len(runs)), format_func=lambda i: labels[i])
        run_id = str(runs[idx]["run_id"])
        run = fetch_pipeline_run(run_id)
        st.session_state.setdefault("viz_plan_by_run", {})

        # --- Plan Charts ---
        plan_btn = st.button("Plan charts (LLM) for selected run")
        if plan_btn:
            st.session_state.pop("viz_plan", None)  # clear previous plan
            with st.spinner("Generating visualization plan (LLM)..."):
                st.session_state["viz_plan"] = plan_visualizations(
                    question=run["question"],
                    final_answer=run["final_answer"],
                    run_id=run["run_id"],
                )

        # show the plan when available
        if "viz_plan" in st.session_state:
            with st.expander("Visualization Plan (JSON)", expanded=False):
                st.json(st.session_state["viz_plan"])

        only_selected = st.checkbox("Use only selected docs (relevance_score=1.0)", value=True)

        # --- Code generation ---
        if "viz_plan" in st.session_state:
            if st.button("Generate viz Python code (LLM)"):
                with st.spinner("Generating Python code (LLM)..."):
                    code = generate_viz_code(
                        question=run["question"],
                        final_answer=run["final_answer"],
                        run_id=run["run_id"],
                    )
                st.session_state["viz_code"] = code

        if "viz_code" in st.session_state:
            with st.expander("Generated Python code", expanded=False):
                st.code(st.session_state["viz_code"], language="python")

        # --- LLM -> Python code generation ---
        st.markdown("### LLM-generated Python charts")

        if "viz_code" in st.session_state:

            if st.button("Run generated code (render PNGs)"):
                with st.spinner("Running generated code..."):
                    res = run_generated_code(run_id, st.session_state["viz_code"])

                st.write(f"Exit code: {res.exit_code}")

                with st.expander("stdout", expanded=False):
                    st.code(res.stdout_path.read_text(encoding='utf-8'), language="text")

                with st.expander("stderr", expanded=False):
                    st.code(res.stderr_path.read_text(encoding='utf-8'), language="text")

        imgs = list_pngs(run_id)
        if imgs:
            for p in imgs:
                st.image(str(p), caption=p.name)
        else:
            st.caption("No images yet for this run.")
