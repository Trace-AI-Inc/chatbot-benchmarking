# Allybot LLM Benchmarking Suite

This project benchmarks multiple Large Language Models (LLMs) to evaluate how accurately they can answer questions based on:
- The **Allybot C2 Cleaning Robot User Manual** (PDF)
- A **Trace AI FAQ dataset** (CSV)

The goal is to test how different models handle varying difficulty levels, paraphrased queries, and fuzzy prompts using structured context.

---

## Supported LLMs

This suite currently benchmarks the following models:

| Model              | Provider     | Notes                                         |
|-------------------|--------------|-----------------------------------------------|
| GPT-3.5 Turbo      | OpenAI       | Fast and cheap baseline                       |
| GPT-4 Turbo        | OpenAI       | Higher intelligence, broader context support  |
| Claude 3 Haiku     | Anthropic via [OpenRouter](https://openrouter.ai) | Incredibly fast and cost-effective            |
| Gemini 1.5 Flash   | Google       | High-speed model with generous context window |

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Trace-AI-Inc/chatbot-benchmarking.git
cd chatbot-benchmarking
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

If you have a requirements.txt, use:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install langchain openai langchain-google-genai PyMuPDF pandas python-dotenv
```

### 4. Configure Environment Variables

Create a .env file in the root directory with the following contents:

```bash
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-gemini-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
```

### 5. Run the Benchmark

```bash
python allybot_llm_benchmark.py
```