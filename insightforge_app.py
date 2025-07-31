# import streamlit as st
# from langchain.chains import LLMChain, SequentialChain
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory
# import os

# # Load key from Streamlit Secrets (you'll add it later in cloud UI)
# openai_api_key = st.secrets["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = openai_api_key

# # LLM setup
# llm = OpenAI(temperature=0.2, model_name="gpt-3.5-turbo-instruct")

# # Static summary for now (replace with dynamic later if needed)
# knowledge_base = """
# 📊 Business Insight Summary:

# • Total Sales: ₹1383220.00
# • Average Sale: ₹553.29
# • Median Sale: ₹552.50
# • Standard Deviation: ₹260.10

# 🗓️ Time Trends:
# • Best Month: 2028-04
# • Worst Month: 2028-11

# 🛒 Product Insights:
# • Top-Selling Products: Widget A (₹375235.00), Widget B (₹346062.00), Widget C (₹335069.00)

# 🌍 Regional Performance:
# • Best Performing Region: West

# 👥 Customer Demographics:
# • Best Performing Age Group: N/A
# """

# # Prompt templates
# insight_prompt = PromptTemplate(
#     input_variables=["data_summary", "user_question"],
#     template="""
# You are a data analyst. You have the following sales data summary:

# {data_summary}

# Based on the user's question below, extract and explain only the **relevant insight** from the data — do not give recommendations yet.

# Question: {user_question}
# Insight:"""
# )

# recommendation_prompt = PromptTemplate(
#     input_variables=["insight", "user_question"],
#     template="""
# You are a strategic business consultant.

# Here is an insight extracted from company data:
# {insight}

# Now, based on this insight and the user's question:
# "{user_question}"

# Generate a clear and actionable recommendation.
# Recommendation:"""
# )

# # Memory & chains
# memory = ConversationBufferMemory(input_key="user_question", memory_key="chat_history")
# insight_chain = LLMChain(llm=llm, prompt=insight_prompt, output_key="insight", memory=memory)
# recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt, output_key="recommendation")

# insightforge_chain = SequentialChain(
#     chains=[insight_chain, recommendation_chain],
#     input_variables=["data_summary", "user_question"],
#     output_variables=["insight", "recommendation"],
#     verbose=False
# )

# # ------------------- Streamlit UI -------------------
# st.set_page_config(page_title="InsightForge", page_icon="📊")
# st.title("📊 InsightForge - Business Intelligence Assistant")

# st.markdown("Ask a question about your company’s sales performance. Get insights + strategic recommendations.")

# user_question = st.text_input("🔍 Ask a business question:")

# if user_question:
#     with st.spinner("Generating insights..."):
#         result = insightforge_chain.invoke({
#             "data_summary": knowledge_base,
#             "user_question": user_question
#         })

#     st.subheader("🧠 Insight")
#     st.success(result['insight'].strip())

#     st.subheader("💡 Recommendation")
#     st.info(result['recommendation'].strip())

#     if st.checkbox("🗂 Show Memory Log"):
#         st.text(memory.buffer)

import streamlit as st
import pandas as pd
import numpy as np
import os

from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load OpenAI key from Streamlit Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize LLM
llm = OpenAI(temperature=0.2, model_name="gpt-3.5-turbo-instruct")

# Function to create summary from uploaded data
def generate_summary_from_df(df):
    total_sales = df['Sales'].sum()
    avg_sale = df['Sales'].mean()
    median_sale = df['Sales'].median()
    std_sale = df['Sales'].std()

    best_month = df.groupby('Month')['Sales'].sum().idxmax()
    worst_month = df.groupby('Month')['Sales'].sum().idxmin()

    product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
    top_products = ", ".join([f"{p} (₹{v:.2f})" for p, v in product_sales.head(3).items()])

    region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    best_region = region_sales.idxmax()
    worst_region = region_sales.idxmin()
    region_breakdown = "\n".join([f"• {region}: ₹{sales:,.2f}" for region, sales in region_sales.items()])

    summary = f"""
📊 Business Insight Summary:

• Total Sales: ₹{total_sales:,.2f}
• Average Sale: ₹{avg_sale:,.2f}
• Median Sale: ₹{median_sale:,.2f}
• Standard Deviation: ₹{std_sale:,.2f}

🗓️ Time Trends:
• Best Month: {best_month}
• Worst Month: {worst_month}

🛒 Product Insights:
• Top-Selling Products: {top_products}

🌍 Regional Performance:
• Best Performing Region: {best_region}
• Regional Sales Breakdown:
{region_breakdown}
• Underperforming Region: {worst_region}

👥 Customer Demographics:
• Best Performing Age Group: N/A
"""
    return summary.strip()

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

# Memory and Chains
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

# Upload CSV
st.sidebar.header("📁 Upload your sales data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
knowledge_base = None

# If uploaded, process file
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        # Clean column names: lowercase, remove spaces
        df.columns = [col.strip().lower() for col in df.columns]
        expected_cols = {'date', 'product', 'sales', 'region'}

        if expected_cols.issubset(df.columns):
            # Convert date and extract month
            df['Date'] = pd.to_datetime(df['date'])
            df['Month'] = df['Date'].dt.to_period('M').astype(str)

            # Rename for consistency with summary function
            df.rename(columns={
                'product': 'Product',
                'sales': 'Sales',
                'region': 'Region'
            }, inplace=True)

            knowledge_base = generate_summary_from_df(df)
            st.sidebar.success("✅ Summary generated from uploaded data!")

            if st.checkbox("👁 Preview DataFrame"):
                st.dataframe(df.head())
        else:
            st.sidebar.error("❌ CSV must contain 'Date', 'Product', 'Sales', and 'Region' columns (case-insensitive).")
    except Exception as e:
        st.sidebar.error(f"❌ Error reading CSV: {e}")

# Ask business question
user_question = st.text_input("🔍 Ask a business question:")

if user_question:
    if not knowledge_base:
        st.warning("Please upload a CSV with required columns to continue.")
    else:
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

