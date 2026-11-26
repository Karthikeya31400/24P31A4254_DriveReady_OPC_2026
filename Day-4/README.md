# Day 4: Tool Calling, Pydantic Structured Outputs, External Integrations & Agent Tracing

> **Drive Ready – AI Agents | One Person Company (OPC) Initiative**  
> *Department of Technical Training, Aditya University*

---

## Overview

Day 4 marks the transition from single-turn LCEL chains to building fully autonomous, multi-turn AI agents. It covers the core building blocks of agentic workflows: defining custom LangChain tools with `@tool`, extracting strongly-typed domain records using Pydantic schemas, integrating external persistence via Google Sheets (`gspread`), diagnosing how poor tool docstrings degrade agent performance, and observing autonomous tool selection, multi-tool chaining, and error-recovery through complete execution traces for **KLG Ancient World** ([klgancientworld.com](https://klgancientworld.com)).

---

## Tasks Covered

### Task 1: Defining Custom Domain Tools with LangChain
* **Implementation:** Defined modular, self-describing tools using LangChain's `@tool` decorator:
  * `get_access_fee(item: str, qty: int = 1)`: Look up prices for KLG script guides (`brahmi guide`: Rs 0, `grantha guide`: Rs 0) and Sacred Landscape Routes (`alqueva route`: Rs 250, `st. peter route`: Rs 200). Handles out-of-catalogue items gracefully.
  * `is_open(day: str)`: Checks research desk and route hours across days of the week, indicating Sunday closure with 24/7 digital app availability.
* **Direct Tool Invocations:**
  * `get_access_fee({"item": "alqueva route", "qty": 2})` $\rightarrow$ `"2 x alqueva route = Rs 500"`
  * `is_open({"day": "sunday"})` $\rightarrow$ `"KLG desk hours on sunday: Closed (Digital App Available 24/7)."`

### Task 2: Structured Output Extraction with Pydantic
* **Implementation:** Built a strict validation model with Pydantic (`HeritageBooking`):
  ```python
  class HeritageBooking(BaseModel):
      name: str = Field(description="Visitor's first name")
      item: Literal["brahmi guide", "grantha guide", "alqueva route", "st. peter route"]
      qty: int = Field(description="Number of passes or copies requested", ge=1)
      date: str = Field(description="Preferred visit or access date as stated by the user")
      language: Literal["English", "Telugu"] = Field(default="English")
  ```
* **Extraction Pipeline:** Integrated with the LLM via `llm.with_structured_output(HeritageBooking)`.
* **Robust Parsing of Unstructured Inputs:**
  * `"hi karthik here, book 2 alqueva route passes for 15 dec in english"` $\rightarrow$ `name='Karthik'`, `item='alqueva route'`, `qty=2`, `date='15 dec'`, `language='English'`
  * `"i am ravi need 1 brahmi guide on 20th dec telugu pls"` $\rightarrow$ `name='ravi'`, `item='brahmi guide'`, `qty=1`, `date='20th dec'`, `language='Telugu'`
  * `"Sravani - 3 passes for st. peter route this sunday"` $\rightarrow$ `name='Sravani'`, `item='st. peter route'`, `qty=3`, `date='this sunday'`, `language='English'`
  * `"can arjun get 4 grantha guide copies for 1 jan telugu"` $\rightarrow$ `name='arjun'`, `item='grantha guide'`, `qty=4`, `date='1 jan'`, `language='Telugu'`
  * `"meera wants 2 st. peter route tickets for next friday"` $\rightarrow$ `name='meera'`, `item='st. peter route'`, `qty=2`, `date='next friday'`, `language='English'`

### Task 3: External Tool Integration (Google Sheets Booking Persistence)
* **Implementation:** Connected the agent to Google Sheets via `gspread` and service credentials.
* **Spreadsheet Schema:** Automatically verifies or initializes `KLG Ancient World Bookings` with columns:
  `["name", "item", "qty", "date", "language", "total"]`.
* **Tool Logic (`log_booking`):**
  * Validates inputs using the `HeritageBooking` schema.
  * Calculates total access fee dynamically.
  * Appends confirmed bookings directly to the remote sheet and returns a confirmation string.

### Task 4: Breaking the Agent — The Criticality of Tool Semantics
* **Experiment:** Created an intentionally degraded tool (`tool1`) with a vague name and docstring (`"""Does stuff."""`).
* **Test Query:** `"How much are 2 Alqueva route passes?"`
* **Agent Behavior & Breakdown:**
  * The model lacked semantic guidance to structure the arguments properly. It passed `item: "Alqueva route pass"` instead of the canonical catalogue key `"alqueva route"`.
  * `tool1` returned `"Not found."`.
  * The agent failed to complete the lookup and fell back to: *"I'll check with the KLG team."*
* **Core Takeaway:** LLM tool selection and parameter generation are directly dependent on unambiguous tool names, docstrings, and parameter typing.

### Task 5: Autonomous Multi-Turn Agent Tracing
* **Implementation:** Built a complete autonomous agent using LangChain's `create_agent` equipped with `[get_access_fee, is_open, log_booking]` and grounded with enterprise system guidelines.
* **Customer Scenarios & Execution Traces:**
  1. **Parallel Multi-Tool Calling:**
     * *User:* `"Are you open on Sunday? And how much are 2 Alqueva route passes?"`
     * *Trace:* Agent autonomously invoked `is_open(day='sunday')` AND `get_access_fee(item='alqueva route', qty=2)` in parallel, combining both tool outputs into a unified answer.
  2. **End-to-End Booking Workflow:**
     * *User:* `"I'm Karthikeya. Book 2 Alqueva route passes for 24 Dec in English."`
     * *Trace:* Agent called `log_booking(...)`, persisted row to Google Sheets, and responded with confirmation and total (`Rs 500`).
  3. **Out-of-Scope Rejection:**
     * *User:* `"How much for a Visigothic tablet?"`
     * *Trace:* Recognizing the item is absent from catalogue and tools, the agent grounded itself and responded: *"I'll check with the KLG team."*
  4. **Missing Information Detection:**
     * *User:* `"Book 2 st. peter route passes please"`
     * *Trace:* Agent called `get_access_fee` to report price (`Rs 400`), but refrained from calling `log_booking`, prompting the user for missing details: *"Could you please share your name, preferred date, and language so I can complete your booking?"*.
  5. **Graceful Pydantic Validation Error Recovery:**
     * *User:* `"Book 0 brahmi guides for Ravi on 25 Dec"`
     * *Trace:* `log_booking` rejected `qty=0` via Pydantic (`Input should be greater than or equal to 1`). The agent intercepted the error and replied politely: *"You need to book at least 1 Brahmi guide! Would you like to update the quantity?"*.

---

## Tech Stack & Setup

### Required Libraries
```bash
pip install -q langchain langchain-core langchain-google-genai pydantic gspread google-auth
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
# Navigate to Day-4 folder
cd Day-4

# Run the task script
python task.py
```
*(Note: Google Sheets authentication requires Google account credentials when running Task 3 & 5).*

### Option 2: Run in Google Colab / Interactive Python
Upload `task.py` to Google Colab or your interactive Python environment, ensure `GEMINI_API_KEY` is configured in secrets, authenticate Google Sheets permissions via `auth.authenticate_user()`, and execute the script.

---

## Key Observations & Learnings

1. **Tool Descriptions are Prompts:** The LLM does not see tool implementation code—it only sees the function name, docstring, and parameter schema. Ambiguous docstrings immediately cause tool-call failures.
2. **Schema-Enforced Extraction:** Using Pydantic models with `with_structured_output` guarantees deterministic JSON fields (`Literal` constraints, integer bounds like `ge=1`) even from unstructured, messy user messages.
3. **Multi-Tool Orchestration:** Modern agents can autonomously decide whether a query requires zero, one, or multiple parallel tool calls (e.g., checking operating hours and calculating fees in one exchange).
4. **Resilient Error Recovery:** When tool executions raise validation errors (such as `qty=0`), passing the error back into the agent context allows the model to explain the error naturally and guide the user to correct their input.
