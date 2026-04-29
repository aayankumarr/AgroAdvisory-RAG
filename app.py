import streamlit as st
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

# ------------------------------
# 1. Load Vector DB
# ------------------------------
PERSIST_DIR = "./agroadvisory_chroma"
COLLECTION = "agroadvisory"

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = Chroma(
    collection_name=COLLECTION,
    persist_directory=PERSIST_DIR,
    embedding_function=embedder,
)

retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})

# ------------------------------
# 2. Prompt builder
# ------------------------------
def build_enhanced_prompt(user_query, retrieved_chunks, structured_data, table_data=None):
    retrieved_context = "\n\n".join(retrieved_chunks)
    structured_context = json.dumps(structured_data, indent=2)

    table_context = ""
    if table_data:
        table_context = "\n\nAdditional Table Data (from research documents):\n"
        table_context += json.dumps(table_data, indent=2)

    prompt = f"""
You are KrishiSaathi, a knowledgeable and empathetic agri-advisor assisting small and marginal farmers in Rewa district of Madhya Pradesh.

Your job is to provide clear, precise, and practical advice using **two types of information**:
1. 📘 Scientific and institutional knowledge from research reports and agri studies.
2. 🌦️ Real-time structured data including:
   - Weather forecasts (rainfall, temperature, humidity, wind)
   - Mandi prices of relevant commodities
   - Government schemes and agronomic events (if present)

---

## 🔍 Farmer's Question:
{user_query}

---

## 📘 Relevant Knowledge from Research Documents:
{retrieved_context}

---

## 🌐 Real-Time Structured Data (in JSON format):
{structured_context}

{table_context}

---

## ✅ Instructions:
- Read all the sections above and combine insights from both scientific and real-time data.
- Use **weather and mandi trends to guide time-sensitive advice** like sowing, irrigation, harvesting, storage.
- When applicable, refer to past practices or region-specific patterns from the research documents.
- If `table_data` is provided, **interpret it and summarize relevant numbers in your advice**.
- Answer should be **concise**, **localized**, and **understandable to a farmer with limited education**.
- Avoid repeating full paragraphs from documents. Instead, synthesize and explain in simple words.
- Use bullet points or short paragraphs where appropriate.

---

## 💬 Final Answer:
"""
    return prompt

# ------------------------------
# 3. Streamlit UI
# ------------------------------
st.set_page_config(page_title="KrishiSaathi Agro-Advisory", layout="wide")

st.title("🌾 KrishiSaathi: Agro-Advisory Assistant")
st.write("Ask questions and get localized, farmer-friendly advice for Rewa district.")

# Input box
user_query = st.text_area("👉 Enter your question:", placeholder="e.g., What is the recommended urea dose for wheat in Rabi season?")

# Mock structured data (replace with API calls later)
structured_data = {
    "weather_forecast": {"rainfall_mm": 10, "temperature_c": 25, "humidity_percent": 68},
    "mandi_prices": {"wheat": "₹2100/quintal", "gram": "₹5200/quintal"},
    "govt_schemes": ["PM-Kisan support available", "Soil health card distribution ongoing"]
}

# Optional table data
table_data = {
    "fertilizer_recommendations": [
        {"crop": "wheat", "urea_kg_per_ha": 120, "expected_yield_quintals": 42}
    ]
}

if st.button("Get Advisory"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        # Retrieve context
        docs = retriever.invoke(user_query)
        if not docs:
            st.error("⚠️ No relevant documents found in the knowledge base.")
        else:
            retrieved_chunks = [d.page_content for d in docs]
            st.info(f"Retrieved {len(retrieved_chunks)} document chunks.")

            # Build prompt
            prompt = build_enhanced_prompt(user_query, retrieved_chunks, structured_data, table_data)

            # st.write("🔎 **Prompt being sent to LLM:**")
            # st.code(prompt[:500] + " ...")  # Preview first 500 chars

            try:
                # Run LLM (Ollama)
                llm = ChatOllama(model="llama3.1:8b", temperature=0)
                response = llm.invoke(prompt)

                # Display answer
                st.subheader("🧑‍🌾 KrishiSaathi’s Advice:")
                st.write(response.content)

                # Optionally show retrieved evidence
                with st.expander("🔎 View supporting document chunks"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**{i}. Source:** {d.metadata.get('source', 'unknown')} | Page {d.metadata.get('page_num', 'N/A')}")
                        st.write(d.page_content)

            except Exception as e:
                st.error(f"❌ Error calling Ollama: {e}")
