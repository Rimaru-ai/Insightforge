import streamlit as st
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os

# Load key from Streamlit Secrets (you'll add it later in cloud UI)
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# LLM setup
llm = OpenAI(temperature=0.2, model_name="gpt-3.5-turbo-instruct")

# Static summary for now (replace with dynamic later if needed)
knowledge_base = """
📊 Business Insight Summary:

• Total Sales: ₹1383220.00
• Average Sale: ₹553.29
• Median Sale: ₹552.50
• Standard Deviation: ₹260.10

🗓️ Time Trends:
• Best Month: 2028-04
• Worst Month: 2028-11

🛒 Product Insights:
• Top-Selling Products: Widget A (₹375235.00), Widget B (₹346062.00), Widget C (₹335069.00)

🌍 Regional Performance:
• Best Performing Region: West

👥 Customer Demographics:
• Best Performing Age Group: N/A
"""

# Prompt templates
insight_prompt = PromptTemplate(
    input_variables=["data_summary", "user_question"],
    template="""
You are a data analyst. You have the following sales data summary:

{data_summary}

Based on the user's question below, extract and explain only the **relevant insight** from the data — do not give recommendations yet.

Question: {user_question}
Insight:"""
)

recommendation_prompt = PromptTemplate(
    input_variables=["insight", "user_question"],
    template="""
You are a strategic business consultant.

Here is an insight extracted from company data:
{insight}

Now, based on this insight and the user's question:
"{user_question}"

Generate a clear and actionable recommendation.
Recommendation:"""
)

# Memory & chains
memory = ConversationBufferMemory(input_key="user_question", memory_key="chat_history")
insight_chain = LLMChain(llm=llm, prompt=insight_prompt, output_key="insight", memory=memory)
recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt, output_key="recommendation")

insightforge_chain = SequentialChain(
    chains=[insight_chain, recommendation_chain],
    input_variables=["data_summary", "user_question"],
    output_variables=["insight", "recommendation"],
    verbose=False
)

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="InsightForge", page_icon="📊")
st.title("📊 InsightForge - Business Intelligence Assistant")

st.markdown("Ask a question about your company’s sales performance. Get insights + strategic recommendations.")

user_question = st.text_input("🔍 Ask a business question:")

if user_question:
    with st.spinner("Generating insights..."):
        result = insightforge_chain.invoke({
            "data_summary": knowledge_base,
            "user_question": user_question
        })

    st.subheader("🧠 Insight")
    st.success(result['insight'].strip())

    st.subheader("💡 Recommendation")
    st.info(result['recommendation'].strip())

    if st.checkbox("🗂 Show Memory Log"):
        st.text(memory.buffer)
