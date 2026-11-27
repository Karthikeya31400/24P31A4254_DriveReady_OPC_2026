# Day 5: RAG with ChromaDB, LangGraph State Checkpointing & Multi-Session Agent Evaluation

> **Drive Ready – AI Agents | One Person Company (OPC) Initiative**  
> *Department of Technical Training, Aditya University*

---

## Overview

Day 5 marks the synthesis of full-stack AI agent engineering by integrating Retrieval-Augmented Generation (RAG) using ChromaDB, thread-isolated conversational memory via LangGraph's `InMemorySaver`, multi-tenant session management, and automated regression testing. The agent acts as an autonomous, enterprise-ready digital assistant for **KLG Ancient World** ([klgancientworld.com](https://klgancientworld.com)), capable of knowledge retrieval, route consultation, live fee calculation, Google Sheets booking persistence, and multi-user thread isolation.

---

## Tasks Covered

### Task 1: Knowledge Ingestion & Vector Storage with ChromaDB
* **Implementation:** Built a vector database using an in-memory ChromaDB client (`chromadb.Client()`) and collection (`klg_ancient_world_kb`).
* **Domain Corpus:** Ingested 18 structured enterprise knowledge facts for KLG Ancient World spanning:
  * Ancient scripts studied (Brahmi, Grantha, Latin, Sanskrit).
  * Narrative heritage (Puranic narratives, Sthala Puranas).
  * Sacred Landscape Routes (Alqueva, St. Peter of the Orchards).
  * MythiLens mobile application ecosystem (AI Artifact Scanner, Inscription Translator, Quests & Campaigns, offline mode).
  * Catalogue pricing, research desk operating hours, booking rules, and negative retail guardrails.
* **Semantic Retrieval Verification:**
  * Query: `"Can I use the mobile app when there is no internet?"`
  * Result: Correctly retrieves offline-first facts without requiring exact keyword matching:
    > *"MythiLens supports offline-first usage so visitors can explore sacred routes without an active internet connection."*

### Task 2: Vector Search Tool (`search_klg_kb`)
* **Implementation:** Wrapped the ChromaDB vector collection into a LangChain `@tool` for agent retrieval:
  ```python
  @tool
  def search_klg_kb(question: str) -> str:
      """Search KLG Ancient World's knowledge base and FAQs..."""
      results = klg_kb.query(query_texts=[question], n_results=3)
      return "\n".join(results["documents"][0])
  ```
* **Tool Check:** Invoking `search_klg_kb` for MythiLens features successfully retrieves top-3 contextual documents (AI Artifact Scanner, Translator, Quests & Leaderboard).

### Task 3: Multi-Session Agent with LangGraph Memory Checkpointing
* **Architecture:** Constructed an autonomous agent with LangChain's `create_agent` using `gemini-3.1-flash-lite`, integrated with a LangGraph state checkpointer:
  ```python
  agent = create_agent(
      model=llm,
      tools=[get_access_fee, is_open, log_booking, search_klg_kb],
      system_prompt=SYSTEM,
      checkpointer=InMemorySaver()
  )
  ```
* **Thread-Isolated Multi-Turn Execution:**
  * **Thread 1 (`karthikeya`):**
    1. *Query 1:* `"Hi, I'm Karthikeya. Can I use MythiLens without internet?"` $\rightarrow$ Agent uses `search_klg_kb` to confirm offline capability.
    2. *Query 2:* `"What is my name and what app feature did I ask about?"` $\rightarrow$ Recalls user name (Karthikeya) and the offline feature discussion from thread state.
    3. *Query 3:* `"Book 2 alqueva route passes for 24 Dec in English."` $\rightarrow$ Calls `log_booking`, saves row to Google Sheets, and confirms total (Rs 500).
  * **Thread 2 (`sravani` - Multi-Tenant Privacy Isolation):**
    1. *Query 1:* `"Hello! I'm Sravani. Which ancient scripts do you study?"` $\rightarrow$ Introduces Brahmi, Grantha, Latin, Sanskrit via `search_klg_kb`.
    2. *Query 2 (Cross-Tenant Leakage Test):* `"Do you remember Karthikeya's question?"`
    3. *Outcome:* Agent strictly replies: *"I don't have access to previous conversations or other users' questions."* This demonstrates zero cross-tenant memory leakage.

### Task 4: Guardrail Testing & Out-of-Scope Enforcement
* **Implementation:** Tested adversarial and out-of-domain inquiries through separate threads:
  * `"Do you offer helicopter tours over Alqueva?"` $\rightarrow$ Fallback response: *"I'll check with the KLG team."*
  * `"Do you sell physical car tyres?"` $\rightarrow$ Fallback response: *"I'll check with the KLG team."*
* **Takeaway:** Explicit system instructions combined with RAG retrieval grounding prevent hallucinated service offerings.

### Task 5: Automated Regression Sign-Off Suite (`run_agent_tests`)
* **Implementation:** Extended the test suite to 15 comprehensive test cases covering script knowledge, route pricing, desk timings, offline capabilities, and out-of-scope policies.
* **State Isolation:** Each test case executes with a unique `thread_id` (`eval_thread_{idx}`) to ensure complete memory isolation across evaluation questions.
* **Evaluation Metrics:**
  * **Sign-off Score:** `12/15 passed (80%)`.
  * **Analysis of Test Variances:**
    * Tests that passed verified core facts: Brahmi, Grantha, Alqueva, St. Peter of the Orchards, Quests, 24/7 digital app hours on Sundays, and out-of-domain rejections.
    * Minor failures highlighted evolving agent capabilities against legacy assertions:
      * Guided tour pricing: The agent answered with the newly configured knowledge base price (`Rs 250`), while the legacy Day 3 test assertion expected the old fallback *"check with the KLG team"*.
      * Feature expansion: The agent described comprehensive MythiLens features (Artifact Scanner, Translator) instead of only matching legacy single-word substrings.

---

## Tech Stack & Setup

### Required Libraries
```bash
pip install -q chromadb langgraph langchain langchain-core langchain-google-genai pydantic gspread google-auth
```

### Environment Variable Setup
```python
import os
from google.colab import userdata

os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
```

---

## How to Run

### Option 1: Run Locally
```bash
# Navigate to Day-5 folder
cd Day-5

# Run the task script
python task.py
```
*(Note: Running Tasks 3 & 5 requires Google account authentication for the Google Sheets booking logger).*

### Option 2: Run in Google Colab / Interactive Python
Upload `task.py` and your test CSV to Google Colab or your interactive Python environment, ensure `GEMINI_API_KEY` is configured in secrets, authenticate Google Sheets permissions via `auth.authenticate_user()`, and execute the script.

---

## Key Observations & Learnings

1. **RAG Eliminates Prompt Bloat:** Instead of hardcoding lengthy factual manuals directly into the system prompt, storing domain knowledge in ChromaDB and providing `search_klg_kb` reduces context token overhead and keeps prompts focused.
2. **State Checkpointing via `InMemorySaver`:** LangGraph's checkpointer architecture manages conversational state cleanly using `thread_id`. Sessions can be paused, resumed, or branched without custom session dictionaries.
3. **Multi-Tenant Security:** Unique `thread_id` keys strictly compartmentalize conversational context, guaranteeing user privacy and preventing cross-user data leakage.
4. **Living Evaluation Suites:** As an agent gains new tools and factual knowledge (e.g., learning that Alqueva route passes cost Rs 250), test suites must evolve; assertions expecting fallback text must be updated to expect accurate factual answers.
