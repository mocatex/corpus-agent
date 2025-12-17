"""
This module implements a multi-step pipeline for a temporal news analytics thesis system.
It uses OpenAI's LLM to assess corpus compatibility, plan document retrieval,
plan NLP analysis, and synthesize final answers. It integrates with OpenSearch
and Postgres for document retrieval, and includes mocked NLP tool outputs.
"""

from openai import OpenAI
from dotenv import load_dotenv
from tools_backend import (
    search_opensearch,
    fetch_run_documents_postgres,
    store_run_articles,
    update_run_articles_nlp_features,
    set_run_articles_relevance,
    store_run_metadata,
    fetch_run_metadata,
)
from mocked_tools import mock_nlp_tool_outputs
import json
import uuid

load_dotenv()
client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_opensearch",
            "description": "Search the news corpus via OpenSearch and return matching article IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 100},
                },
                "required": ["query"],
            },
        },
    },
]


# --- Corpus compatibility assessment ---
def assess_corpus_compatibility(question: str) -> dict:
    """
    Let the LLM decide whether the question can be reasonably answered
    from a corpus that only covers news between 2016 and 2021.

    This is needed for questions without explicit years (e.g., historical events),
    where we still want the system to say: 'out of corpus range'.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a temporal scope classifier for a news analytics thesis system. "
                "The underlying corpus contains ONLY news articles from 2016 to 2021. "
                "Given a user question, you must decide whether a reasonable answer could be derived "
                "mainly from news coverage in 2016–2021. "
                "Examples of NOT answerable from this corpus: questions about John F. Kennedy's election campaign, "
                "World War II, the fall of the Berlin Wall, or time ranges like 'from 2002 to 2008'. "
                "If the question focuses on events, people, or time spans clearly outside 2016–2021, "
                "mark it as not answerable from this corpus. "
                "If it spans a long period but includes 2016–2021 (e.g., 'from 2000 to 2020'), consider it answerable, "
                "but note that only the 2016–2021 part can be covered. "
                "Return ONLY a JSON object with the schema: "
                "{ "
                "  'can_answer_from_corpus': boolean, "
                "  'primary_time_period': string, "
                "  'explanation': string "
                "}."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        response_format={"type": "json_object"},
    )
    assessment = json.loads(resp.choices[0].message.content)
    return assessment



def retrieve_documents(q: str, run_id: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Retrieval Planner for a temporal news analytics thesis system operating on a newspaper corpus covering the years 2016–2021. "
                "Given a user question, decide whether you need to call tools. When you call tools, your goal is to: "
                "1) Formulate a focused keyword query that reflects the entities, topics, and temporal scope in the question. "
                "2) Use search_opensearch with a reasonable top_k (ideally <= 100) to retrieve document IDs. "
                "Do NOT try to answer the question yourself in this step; just plan retrieval via tools. "
                "If the question refers to years outside 2016–2021, conceptually restrict retrieval to content from 2016–2021, as the corpus only covers that range."
            )
        },
        {"role": "user", "content": q}
    ]

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    msg = resp.choices[0].message

    # If the model wants to call a tool:
    if msg.tool_calls:
        print("Tools detected, handling tool calls...")
        search_results = []
        documents = []

        for tc in msg.tool_calls:
            print(f"➡ Tool call received: {tc.function.name} with args {tc.function.arguments}")
            fn_name = tc.function.name
            args = json.loads(tc.function.arguments)

            if fn_name == "search_opensearch":
                print("Calling search_opensearch...")
                result = search_opensearch(
                    query=args["query"],
                    top_k=args.get("top_k", 100),
                )
                search_results.extend(result)

        # After OpenSearch, we decide to fetch from Postgres:
        if search_results:
            store_run_articles(run_id=run_id, question=q, search_results=search_results)
            documents = fetch_run_documents_postgres(run_id=run_id)
            print(f"Loaded {len(documents)} documents from DB for run_id={run_id}")
        else:
            documents = []

        return {
            "tool_calls": msg.tool_calls,
            "search_results": search_results,
            "documents": documents,
        }

    print("No tools were called by the LLM. Returning empty results.")
    return {
        "tool_calls": [],
        "search_results": [],
        "documents": [],
    }


def plan_nlp_analysis(question: str, retrieval_result: dict) -> dict:
    """
    Mocked NLP pipeline planner.

    Given the original question and a light summary of the retrieval result,
    decide (at a high level) which NLP tools *would* be applied next.
    This is a planning step only; no real NLP is executed.
    """
    summary = {
        "question": question,
        "num_search_results": len(retrieval_result.get("search_results", [])),
        "num_documents": len(retrieval_result.get("documents", [])),
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert NLP Pipeline Planner for a temporal news analytics research system over a news corpus (2016–2021). "
                "Your role is to analyze the user's question together with a short summary of the retrieved documents and decide which analytical tools must be executed next in the pipeline. "
                "You do NOT answer the user's question directly; you ONLY select and justify the next processing steps. "
                "Each decision must consider: the temporal scope of the question, whether the question is descriptive, comparative, causal, predictive, or exploratory, "
                "and whether sentiment, topics, entities, events, framing, or narrative change are required. "
                "Available (mocked) tools include: "
                "'sentiment_over_time', 'volatility_of_sentiment', 'emotion_distribution_over_time', "
                "'topic_trend_over_time', 'dynamic_topic_shift_detection', "
                "'event_detection', 'burst_detection', "
                "'named_entity_trend_tracking', 'relationship_graph_extraction', "
                "'framing_shift_detection', 'stance_detection_over_time', "
                "'salient_quote_extraction', 'fact_density_estimation', "
                "'source_bias_comparison', 'temporal_segmentation', "
                "'document_clustering', 'keyphrase_evolution_tracking', "
                "'anomaly_detection_in_discourse', 'confidence_scoring_of_results'. "
                "You must return ONLY a valid JSON object with the following schema: "
                "{ 'task_type': string, 'time_horizon': string, 'chosen_tools': [string], "
                "  'explanation': string, 'desired_additional_tools': [string] }. "
                "Rules: Select ONLY tools that are strictly necessary, do NOT include extra text outside the JSON, "
                "keep the explanation concise (max. 2 sentences), set 'time_horizon' explicitly (e.g., '2016–2021', 'pre/post event', 'monthly over 5 years'), "
                "and set 'task_type' to one of: ['descriptive', 'comparative', 'causal', 'predictive', 'exploratory']. "
                "If you would like to use an NLP tool that is NOT in the available tools list, briefly describe it in natural language and add it to 'desired_additional_tools'. "
                "If no extra tools are needed, return an empty list there."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(summary),
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        response_format={"type": "json_object"},
    )
    plan = json.loads(resp.choices[0].message.content)
    return plan




def build_doc_metadata(documents: list[dict]) -> list[dict]:
    """
    Build lightweight semantic metadata for each document to help the LLM
    select a subset. This does NOT include full text, only compact features.

    The metadata schema is:

    {
      "doc_id": int,
      "year": int | str,
      "title": str,
      "text": str,
      "os_score": float | None,
      "sentiment": { "label": str, "score": float },
      "topics": [str],
      "entities": { "ORG": [str], "PERSON": [str], "GPE": [str] },
      "events": any,
      "embedding_ref": str | None
    }

    Currently, sentiment, topics, entities, events and embeddings are mocked
    or left empty; in a future version these fields will be populated by the
    NLP analytics layer.
    """
    meta = []

    for d in documents:
        body = (d.get("body") or "")
        year = d.get("year", "unknown")

        # Get Full Article Text to append
        full_text = body.replace("\n", " ")

        extra = d.get("extra_metadata") or {}
        mock_nlp = extra.get("mock_nlp") if isinstance(extra, dict) else {}

        sentiment_score = d.get("sentiment_score")
        sentiment_label = None

        # label from DB if available
        if isinstance(mock_nlp, dict):
            sentiment_label = mock_nlp.get("sentiment_label")
        topic_label = mock_nlp.get("topic") if isinstance(mock_nlp, dict) else None
        emotion_label = mock_nlp.get("emotion") if isinstance(mock_nlp, dict) else None

        meta.append(
            {
                "doc_id": d.get("id"),
                "year": year,
                "title": (d.get("title") or "")[:200],
                "text": full_text,
                "bm25_score": d.get("os_score"),
                "rank": d.get("rank"),
                "sentiment": {
                    "label": sentiment_label,
                    "score": round(sentiment_score, 3),
                },
                "topics": [topic_label],
                "emotion": emotion_label,
                "entities": {
                    "ORG": [],
                    "PERSON": [],
                    "GPE": [],
                },
                "events": None,
                "embedding_ref": None,
            }
        )

    return meta

def select_documents_agentically(
    question: str,
    documents: list[dict],
    max_docs_total: int = 30,
    batch_size: int = 50,
) -> dict:
    """
    Let the LLM decide which subset of retrieved documents should be kept
    for summarization and final analysis.

    Steps:
    1) Simple algorithmic filtering (drop very short docs, filter invalid years).
    2) Build compact semantic metadata for each remaining document.
    3) Send the metadata to the LLM in batches and ask it to mark which
       documents in each batch are relevant to the question.
    4) Combine the per-batch decisions into a final set of selected IDs,
       capped at max_docs_total.

    The LLM sees metadata including full text: year, title, text, BM25/OS score,
    mocked sentiment, topics, etc.
    """

    # --- 1) Simple algorithmic filtering ---
    filtered_docs: list[dict] = []
    for d in documents:
        body = (d.get("body") or "")
        if len(body) < 100:
            # Drop super short documents
            continue

        year = d.get("year")
        try:
            if year is not None and year != "unknown":
                year_int = int(year)
                if year_int < 2016 or year_int > 2021:
                    # Filter out years outside the corpus range as a safety net
                    continue
        except (TypeError, ValueError):
            # If year cannot be parsed, keep the doc (or you could choose to skip it)
            pass

        filtered_docs.append(d)

    # --- 2) Build metadata for the remaining docs ---
    metadata = build_doc_metadata(filtered_docs)

    # If there are fewer docs than the budget, just keep all
    if len(metadata) <= max_docs_total:
        return {
            "selected_ids": [m["doc_id"] for m in metadata if m.get("doc_id") is not None],
            "reasoning": "Number of filtered documents is already below the maximum budget; keeping all.",
            "batches": [],
        }

    # --- 3) Batched LLM selection over metadata ---
    batches = [
        metadata[i : i + batch_size] for i in range(0, len(metadata), batch_size)
    ]

    all_relevant_ids = set()
    batch_decisions = []

    for batch_index, batch in enumerate(batches):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document selection planner for a temporal news analytics system. "
                    "You are given a user question and a batch of PER-ARTICLE metadata (doc_id, year, title, "
                    "text, BM25 score, rank, sentiment, topics, emotion, entities, etc.). "
                    "Your task is to mark which documents in THIS BATCH are most relevant to the question. "
                    "Focus on semantic relevance to the question, sentiment contrast when useful, and coverage "
                    "of the key topics mentioned in the question. "
                    "Return ONLY a JSON object with the schema: "
                    "{ 'relevant_ids': [int], 'reasoning': string }."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "batch_index": batch_index,
                        "max_docs_total": max_docs_total,
                        "documents": batch,
                    }
                ),
            },
        ]

        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            response_format={"type": "json_object"},
        )
        batch_plan = json.loads(resp.choices[0].message.content)
        relevant_ids = [
            doc_id for doc_id in batch_plan.get("relevant_ids", []) if doc_id is not None
        ]
        all_relevant_ids.update(relevant_ids)
        batch_decisions.append(
            {
                "batch_index": batch_index,
                "relevant_ids": relevant_ids,
                "reasoning": batch_plan.get("reasoning", ""),
            }
        )

    # --- 4) Cap at max_docs_total, preserving original metadata order ---
    selected_ids_ordered: list[int] = []
    relevant_id_set = set(all_relevant_ids)
    for m in metadata:
        doc_id = m.get("doc_id")
        if doc_id in relevant_id_set:
            selected_ids_ordered.append(doc_id)
        if len(selected_ids_ordered) >= max_docs_total:
            break

    return {
        "selected_ids": selected_ids_ordered,
        "reasoning": "Batched LLM-based selection over filtered metadata.",
        "batches": batch_decisions,
    }


def summarize_documents_per_year(question: str, retrieval_result: dict) -> dict:
    """
    Use the LLM to create short summaries per year, based on a small subset
    of documents from that year. The number of docs per year is chosen
    by a separate summarization planning step.
    """
    documents = retrieval_result.get("documents", [])

    # Group docs by year
    docs_by_year = {}
    for doc in documents:
        year = doc.get("year", "unknown")
        docs_by_year.setdefault(year, []).append(doc)

    year_summaries = {}

    for year, docs in sorted(docs_by_year.items(), key=lambda x: x[0]):
        subset = docs
        if not subset:
            continue

        context_chunks = []
        for d in subset:
            title = d.get("title", "") or ""
            body = (d.get("body") or "").replace("\n", " ")
            context_chunks.append(f"TITLE: {title}\nTEXT: {body[:2000]}")

        prompt = (
                "You are a news summarizer for a temporal analytics thesis. "
                "Given a user question and several news articles from a single year, "
                "write a concise 3–5 sentence summary of how that year's coverage relates to the question. "
                "Focus on main themes, sentiment and notable events; do not invent facts.\n\n"
                f"User question: {question}\n"
                f"Year: {year}\n\n"
                "Articles:\n" + "\n\n".join(context_chunks)
        )

        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system",
                 "content": "You summarize news articles with a focus on temporal trends and perception."},
                {"role": "user", "content": prompt},
            ],
        )
        year_summaries[year] = resp.choices[0].message.content.strip()

    return year_summaries


def synthesize_final_answer(question: str, year_summaries: dict, nlp_plan: dict, mocked_tool_outputs: dict) -> str:
    """
    Final LLM synthesis step:

    - Takes the original question
    - Uses compressed, year-level summaries
    - Uses high-level analytics from the mocked tools
    - Produces an answer that is explicitly grounded ONLY in the provided context
    """
    payload = {
        "question": question,
        "year_summaries": year_summaries,
        "nlp_plan": nlp_plan,
        "analytics": mocked_tool_outputs,
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful research assistant for a temporal news analytics thesis. "
                "You must answer the user's question using ONLY the provided context (summaries and analytics). "
                "If the context is insufficient, explicitly say so and describe what additional data would be needed. "
                "Cite years or high-level trends, but do not fabricate specific numbers, quotes, or events."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload),
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
    )
    return resp.choices[0].message.content.strip()


def run_thesis_pipeline(q: str) -> dict:
    print(f"=== Running thesis pipeline for question: {q!r} ===")

    run_id = str(uuid.uuid4())

    scope_assessment = assess_corpus_compatibility(q)
    print("[Scope] Corpus compatibility assessment:", scope_assessment)

    if not scope_assessment.get("can_answer_from_corpus", True):
        final_answer = (
            "I cannot reliably answer this question from the thesis corpus, because it focuses on news articles "
            "from 2016 to 2021, while your question is primarily about a different time period. "
            f"{scope_assessment.get('explanation', '')} "
            "To answer this properly, I would need a corpus that includes the relevant years."
        )
        return {
            "run_id": run_id,
            "question": q,
            "retrieval": {"tool_calls": [], "search_results": [], "documents": []},
            "retrieval_plan": {},
            "nlp_plan": None,
            "mocked_tool_outputs": {},
            "year_summaries": {},
            "final_answer": final_answer,
            "scope_assessment": scope_assessment,
        }

    # 1) Retrieval (writes pipeline_run_articles + loads initial documents)
    retrieval_result = retrieve_documents(q, run_id=run_id)
    retrieval_plan = retrieval_result.get("retrieval_plan", {})

    documents = retrieval_result.get("documents", [])

    # 2) NLP planning + mocked tools (run-level)
    nlp_plan = plan_nlp_analysis(q, retrieval_result)
    mocked_tool_outputs = mock_nlp_tool_outputs(retrieval_result, nlp_plan)

    store_run_metadata(
        run_id=run_id,
        question=q,
        nlp_plan=nlp_plan,
        mocked_tool_outputs=mocked_tool_outputs,
        final_answer="" # Placeholder; will be updated later
    )

    # 2b) Per-article NLP features -> temp table
    update_run_articles_nlp_features(run_id=run_id, documents=documents)

    # Re-fetch to include sentiment_score / extra_metadata / os_score / rank (DB-backed)
    documents = fetch_run_documents_postgres(run_id=run_id)

    # 3) Agentic selection
    doc_selection_plan = select_documents_agentically(q, documents, max_docs_total=50)
    selected_ids = set(doc_selection_plan.get("selected_ids", []))

    # After selection, update relevance scores in DB, set to 1.0 for selected, 0.0 for others
    set_run_articles_relevance(run_id=run_id, selected_article_ids=list(selected_ids))

    if selected_ids:
        documents = [d for d in documents if d.get("id") in selected_ids]

    # Put final docs back into retrieval_result for downstream summarization
    retrieval_result["documents"] = documents

    # 4) Summarization
    year_summaries = summarize_documents_per_year(q, retrieval_result)

    # 5) Final synthesis (use persisted run-level metadata if present)
    run_meta = fetch_run_metadata(run_id=run_id) or {}
    final_answer = synthesize_final_answer(
        q,
        year_summaries,
        run_meta.get("nlp_plan", nlp_plan),
        run_meta.get("mocked_tool_outputs", mocked_tool_outputs),
    )

    # Update final answer in DB
    store_run_metadata(
        run_id=run_id,
        question=q,
        nlp_plan=nlp_plan,
        mocked_tool_outputs=mocked_tool_outputs,
        final_answer=final_answer
    )

    return {
        "run_id": run_id,
        "question": q,
        "retrieval": retrieval_result,
        "retrieval_plan": retrieval_plan,
        "doc_selection_plan": doc_selection_plan,
        "nlp_plan": nlp_plan,
        "mocked_tool_outputs": mocked_tool_outputs,
        "year_summaries": year_summaries,
        "final_answer": final_answer,
        "scope_assessment": scope_assessment,
    }


if __name__ == "__main__":
    q = "How has the perception of Instagram changed over the course of 2016?"
    result = run_thesis_pipeline(q)
    print("=== Final pipeline result (truncated view) ===")
    print(f"Tool calls: {len(result['retrieval']['tool_calls'])}")
    print(f"Search results: {len(result['retrieval']['search_results'])}")
    print(f"Retrieved documents: {len(result['retrieval']['documents'])}")
    print("NLP plan:", result["nlp_plan"])
    print("Final Answer:\n", result["final_answer"])
