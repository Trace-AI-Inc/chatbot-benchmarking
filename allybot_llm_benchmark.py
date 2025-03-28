import fitz  # PyMuPDF
import time
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import AIMessage
from langchain.chat_models import ChatAnthropic  # For Claude (via OpenRouter)

# Load .env variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 1. Load PDF manual text
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 1b. Load FAQ CSV
def load_faq_text(csv_path):
    df = pd.read_csv(csv_path)
    faq_text = "\n".join([f"Q: {row['question']}\nA: {row['answer']}" for _, row in df.iterrows()])
    return faq_text


manual_path = "AllybotManual.pdf"  # Ensure it's in the same directory
manual_text = extract_pdf_text(manual_path)
manual_text = manual_text[:12000]  # Truncate if needed
faq_path = "traceAI_faqs.csv"  # Ensure it's in the same directory
faq_text = load_faq_text(faq_path)
# faq_text = load_faq_text("allybot_faq.csv")[:6000]  # optional truncation

# 2. Define test questions
questions = [
    # 🟢 Easy
    "What does Trace AI do?",
    "How does Trace AI use AI?",
    "Who can benefit from Trace AI?",
    "Does Trace AI work with Excel?",
    "Is Trace AI secure?",
    "Can Trace AI integrate with my current software?",
    "Does Trace AI need training to use?",
    "How long does implementation take?",
    "Is support available after setup?",
    "Can I use Trace AI on mobile?",
    "How do I turn on the Allybot C2?",
    "What is the initial password for unlocking the screen?",
    "How many cleaning modes are there?",
    "What happens when the clean water tank hits 1%?",
    "How do I start a full clean?",
    "What does the emergency stop button do?",
    "Can I use the mobile app with this robot?",
    "How often should I clean the HEPA filter?",
    "How do I know when the sewage tank is full?",
    "How do I update the robot's firmware?",

    # 🟡 Medium
    "What kind of businesses would get the most out of using Trace AI?",
    "How does Trace AI help with tracking documentation and approvals?",
    "If I’m not tech-savvy, can I still use Trace AI?",
    "Does Trace AI work with Microsoft Teams or SharePoint?",
    "Can Trace AI help reduce audit risks?",
    "How does Trace AI simplify the approval workflow?",
    "What kind of customer support can I expect from Trace AI?",
    "Can I upload invoices or receipts into Trace AI?",
    "How customizable is Trace AI to my workflow?",
    "Can multiple users access Trace AI at once?",
    "What's the best way to use the mapping feature in a new environment?",
    "What are the steps for timer-based cleaning routines?",
    "Can the robot clean multiple rooms in one go?",
    "What should I do if the robot loses its position?",
    "How does the robot handle glass walls or mirrors?",
    "What kind of maintenance does the roller brush need?",
    "Can I disable vacuuming during mopping?",
    "How do I manually save a map without the charging station?",
    "Does the robot automatically detect when to recharge?",
    "How do I replace the mopping pad?",

    # 🔴 Hard / Ambiguous
    "I’m drowning in paperwork—can this help with that?",
    "Will it catch things like missing signatures?",
    "Do I have to manually check if something’s been approved?",
    "What if I want to change the approval steps later?",
    "Will it work with the systems I already use at work?",
    "Can it replace having to remind people to sign things?",
    "How much tech experience do I need to use this?",
    "Can this help with audits even if we’re a small business?",
    "Do I need to install anything or is it all online?",
    "What happens after I upload a document?",
    "The robot’s making weird noises — what should I look into?",
    "If there's water everywhere after a clean, what might be broken?",
    "Where does it show up if there's a problem with navigation?",
    "It keeps spinning in place — is this a sensor issue?",
    "What’s the deal with the laser thing that makes the map?",
    "Something about the LiDAR seems off, how do I fix it?",
    "Why is it not sucking stuff up during the cleaning run?",
    "How do I use the app to like, do cleaning every day at 8am?",
    "It bumped into my chair and stopped — is that normal?",
    "Can this robot work on rugs or do they mess it up?",

    # 🧠 Contextual Recall
    "What are the temperature and humidity limits for safe operation?",
    "If I remove the battery while it’s powered on, will it remember the map?",
    "Is there a warranty on the water sensor pin?",
    "Can I use the robot in a multi-floor building with saved maps?",
    "What if I want to set cleaning permissions for someone else?"
]


# 3. Initialize LLMs
openai_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
gpt4_llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0.2,
)
claude_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="anthropic/claude-3-haiku",
    temperature=0.2,
)

# 4. System context from manual
system_prompt = SystemMessage(
    content=f"You are a helpful assistant answering questions using the Allybot C2 Cleaning Robot User Manual and FAQs.\n\n"
            f"---\n📘 MANUAL CONTENT:\n{manual_text}\n\n"
            f"---\n❓FAQ CONTENT:\n{faq_text}"
)

# 5. Run benchmark
results = []

for i, question in enumerate(questions, 1):
    print(f"\n🧪 [{i}/{len(questions)}] Question: {question}")
    
    for model_name, llm in [
        ("GPT-3.5 Turbo", openai_llm),
        ("GPT-4 Turbo", gpt4_llm),
        ("Claude 3 Haiku", claude_llm),
        ("Gemini 1.5 Flash", gemini_llm)
    ]:
        print(f"🔍 Asking with {model_name}...")

        try:
            response = llm([system_prompt, HumanMessage(content=question)]).content
        except Exception as e:
            response = f"❌ Error: {e}"
            print(response)

        results.append({
            "Model": model_name,
            "Question": question,
            "Response": response
        })
        time.sleep(1)


# 6. Save results
df = pd.DataFrame(results)
df.to_csv("allybot_benchmark_results.csv", index=False)
print("✅ Benchmark complete. Results saved to allybot_benchmark_results.csv")
