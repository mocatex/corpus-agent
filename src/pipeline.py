# rag_llm_pipeline.py
from openai import OpenAI
from tools_backend import search_opensearch, fetch_articles_postgres
import json
import os

client = OpenAI()


print("API KEY LOADED:", os.getenv("OPENAI_API_KEY") is not None)

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
    {
        "type": "function",
        "function": {
            "name": "fetch_articles_postgres",
            "description": "Fetch full article texts and metadata for given article IDs from Postgres.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    }
                },
                "required": ["ids"],
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
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
    )
    assessment = json.loads(resp.choices[0].message.content)
    return assessment

def plan_summarization(question: str, retrieval_result: dict) -> dict:
    """
    Let the LLM decide how many docs per year to summarize (budget),
    based on question difficulty and document distribution.
    """
    documents = retrieval_result.get("documents", [])
    docs_by_year = {}
    for doc in documents:
        year = doc.get("year", "unknown")
        docs_by_year.setdefault(year, 0)
        docs_by_year[year] += 1

    summary = {
        "question": question,
        "total_documents": len(documents),
        "docs_per_year": docs_by_year,
        "difficulty": retrieval_result.get("retrieval_plan", {}).get("difficulty"),
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a summarization budget planner for a temporal news analytics system. "
                "Given the question, its difficulty, and the number of documents per year, "
                "decide how many documents per year should be passed into the LLM summarizer. "
                "Trade-off: more docs -> richer summaries but higher cost. "
                "Rules: "
                "- For 'simple' questions, prefer 1–3 docs per year. "
                "- For 'moderate', 3–5 docs per year. "
                "- For 'hard', 5–10 docs per year, but keep total across all years under ~60 docs if possible. "
                "Return ONLY JSON with schema: "
                "{ 'max_docs_per_year': integer, 'reasoning': string }."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(summary),
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
    )
    plan = json.loads(resp.choices[0].message.content)
    return plan

def retrieve_documents(q: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Retrieval Planner for a temporal news analytics thesis system operating on a newspaper corpus covering the years 2016–2021. "
                "Given a user question, decide whether you need to call tools. When you call tools, your goal is to: "
                "1) Formulate a focused keyword query that reflects the entities, topics, and temporal scope in the question. "
                "2) Use search_opensearch with a reasonable top_k (ideally <= 100) to retrieve document IDs. "
                "3) Optionally call fetch_articles_postgres to turn those IDs into full documents. "
                "Do NOT try to answer the question yourself in this step; just plan retrieval via tools. "
                "If the question refers to years outside 2016–2021, conceptually restrict retrieval to content from 2016–2021, as the corpus only covers that range."
            )
        },
        {"role": "user", "content": q}
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
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

        #After OpenSearch, we decide to fetch from Postgres:
        if search_results:
            ids = [hit["id"] for hit in search_results]
            print(f"Fetching {len(ids)} documents from Postgres...")
            documents = fetch_articles_postgres(ids=ids)
        else:
            documents = []

        return {
            "tool_calls": msg.tool_calls,  # or a simplified version
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
                "{ 'task_type': string, 'time_horizon': string, 'chosen_tools': [string], 'explanation': string }. "
                "Rules: Select ONLY tools that are strictly necessary, do NOT include extra text outside the JSON, "
                "keep the explanation concise (max. 2 sentences), set 'time_horizon' explicitly (e.g., '2016–2021', 'pre/post event', 'monthly over 5 years'), "
                "and set 'task_type' to one of: ['descriptive', 'comparative', 'causal', 'predictive', 'exploratory']."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(summary),
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
    )
    plan = json.loads(resp.choices[0].message.content)
    return plan


def mock_nlp_tool_outputs(retrieval_result: dict, nlp_plan: dict) -> dict:
    """
    Very lightweight mocked execution of a subset of NLP tools.

    This function does NOT run any real NLP. It only fabricates plausible-looking
    outputs to demonstrate what the downstream analytics layer might see.
    """
    documents = retrieval_result.get("documents", [])
    chosen_tools = nlp_plan.get("chosen_tools", [])

    # Group documents by year (assuming Postgres returns 'year' in each row)
    docs_by_year = {}
    for doc in documents:
        year = doc.get("year", "unknown")
        docs_by_year.setdefault(year, []).append(doc)

    outputs = {}

    # --- Mock sentiment_over_time ---
    if "sentiment_over_time" in chosen_tools:
        series = []
        for year, year_docs in sorted(docs_by_year.items(), key=lambda x: x[0]):
            # Fake "sentiment" as a deterministic function of average body length
            if year_docs:
                lengths = [len((d.get("body") or "")) for d in year_docs]
                avg_len = sum(lengths) / max(len(lengths), 1)
                avg_sentiment = (avg_len % 800) / 400.0 - 1.0  # maps roughly to [-1, +1]
            else:
                avg_sentiment = 0.0
            series.append(
                {
                    "year": year,
                    "avg_sentiment": round(avg_sentiment, 3),
                    "n_docs": len(year_docs),
                }
            )
        outputs["sentiment_over_time"] = {"series": series}

    # --- Mock topic_trend_over_time ---
    if "topic_trend_over_time" in chosen_tools:
        topics = []
        for idx, label in enumerate(["Economy / Markets", "Technology / Innovation", "Politics / Regulation"], start=1):
            timeline = []
            for year, year_docs in sorted(docs_by_year.items(), key=lambda x: x[0]):
                weight = (len(year_docs) + idx) % 10 / 10.0
                timeline.append({"year": year, "weight": round(weight, 2)})
            topics.append(
                {
                    "topic_id": idx,
                    "label": label,
                    "timeline": timeline,
                }
            )
        outputs["topic_trend_over_time"] = {"topics": topics}

    # --- Mock event_detection ---
    if "event_detection" in chosen_tools:
        events = []
        for year, year_docs in sorted(docs_by_year.items(), key=lambda x: x[0]):
            if not year_docs:
                continue
            # Just take the first document title as a "major event" placeholder
            first_doc = year_docs[0]
            events.append(
                {
                    "year": year,
                    "title": first_doc.get("title", "Unknown event"),
                    "approx_date": first_doc.get("date", f"{year}-01-01"),
                    "evidence_doc_ids": [first_doc.get("id")],
                }
            )
        outputs["event_detection"] = {"events": events}

    # --- Mock salient_quote_extraction ---
    if "salient_quote_extraction" in chosen_tools:
        quotes = []
        for doc in documents[:10]:
            body = (doc.get("body") or "").replace("\n", " ")
            snippet = body[:200]
            if not snippet:
                continue
            quotes.append(
                {
                    "doc_id": doc.get("id"),
                    "year": doc.get("year"),
                    "quote": snippet + ("..." if len(body) > 200 else ""),
                }
            )
        outputs["salient_quote_extraction"] = {"quotes": quotes}

    # --- Mock volatility_of_sentiment ---
    if "volatility_of_sentiment" in chosen_tools and "sentiment_over_time" in outputs:
        series = outputs["sentiment_over_time"]["series"]
        deltas = []
        for i in range(1, len(series)):
            prev = series[i - 1]["avg_sentiment"]
            curr = series[i]["avg_sentiment"]
            deltas.append(abs(curr - prev))
        volatility = sum(deltas) / max(len(deltas), 1) if deltas else 0.0
        outputs["volatility_of_sentiment"] = {"avg_delta": round(volatility, 3)}

    return outputs


def summarize_documents_per_year(question: str, retrieval_result: dict) -> dict:
    """
    Use the LLM to create short summaries per year, based on a small subset
    of documents from that year. The number of docs per year is chosen
    by a separate summarization planning step.
    """
    documents = retrieval_result.get("documents", [])

    # LLM decides how many docs/year
    summarization_plan = plan_summarization(question, retrieval_result)
    max_docs_per_year = int(summarization_plan.get("max_docs_per_year", 3))
    print(f"[Summarization] Using max_docs_per_year={max_docs_per_year}")
    # you can also return summarization_plan in run_thesis_pipeline for debugging

    # Group docs by year
    docs_by_year = {}
    for doc in documents:
        year = doc.get("year", "unknown")
        docs_by_year.setdefault(year, []).append(doc)

    year_summaries = {}

    for year, docs in sorted(docs_by_year.items(), key=lambda x: x[0]):
        subset = docs[:max_docs_per_year]
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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You summarize news articles with a focus on temporal trends and perception."},
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
        model="gpt-4o-mini",
        messages=messages,
    )
    return resp.choices[0].message.content.strip()

def run_thesis_pipeline(q: str) -> dict:
    print(f"=== Running thesis pipeline for question: {q!r} ===")

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
            "question": q,
            "retrieval": {
                "tool_calls": [],
                "search_results": [],
                "documents": [],
            },
            "retrieval_plan": {},
            "nlp_plan": None,
            "mocked_tool_outputs": {},
            "year_summaries": {},
            "final_answer": final_answer,
            "scope_assessment": scope_assessment,
        }

    retrieval_result = retrieve_documents(q)
    retrieval_plan = retrieval_result.get("retrieval_plan", {})
    print("[Plan] Retrieval plan:", retrieval_plan)

    print(
        f"Retrieval summary -> "
        f"search_results: {len(retrieval_result['search_results'])}, "
        f"documents: {len(retrieval_result['documents'])}"
    )

    nlp_plan = plan_nlp_analysis(q, retrieval_result)
    print("[Plan] NLP plan (mocked):", nlp_plan)

    mocked_tool_outputs = mock_nlp_tool_outputs(retrieval_result, nlp_plan)
    print("[Mock NLP] Outputs keys:", list(mocked_tool_outputs.keys()))

    year_summaries = summarize_documents_per_year(q, retrieval_result)
    print("[Summaries] Years:", list(year_summaries.keys()))

    final_answer = synthesize_final_answer(q, year_summaries, nlp_plan, mocked_tool_outputs)

    return {
        "question": q,
        "retrieval": retrieval_result,
        "retrieval_plan": retrieval_plan,
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