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

# import streamlit as st
# import pandas as pd
# import numpy as np
# import os

# from langchain.chains import LLMChain, SequentialChain
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory

# # Load OpenAI key from Streamlit Secrets
# openai_api_key = st.secrets["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = openai_api_key

# # Initialize LLM
# llm = OpenAI(temperature=0.2, model_name="gpt-3.5-turbo-instruct")

# # Function to create summary from uploaded data
# def generate_summary_from_df(df):
#     total_sales = df['Sales'].sum()
#     avg_sale = df['Sales'].mean()
#     median_sale = df['Sales'].median()
#     std_sale = df['Sales'].std()

#     best_month = df.groupby('Month')['Sales'].sum().idxmax()
#     worst_month = df.groupby('Month')['Sales'].sum().idxmin()

#     product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
#     top_products = ", ".join([f"{p} (₹{v:.2f})" for p, v in product_sales.head(3).items()])

#     region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
#     best_region = region_sales.idxmax()
#     worst_region = region_sales.idxmin()
#     region_breakdown = "\n".join([f"• {region}: ₹{sales:,.2f}" for region, sales in region_sales.items()])

#     summary = f"""
# 📊 Business Insight Summary:

# • Total Sales: ₹{total_sales:,.2f}
# • Average Sale: ₹{avg_sale:,.2f}
# • Median Sale: ₹{median_sale:,.2f}
# • Standard Deviation: ₹{std_sale:,.2f}

# 🗓️ Time Trends:
# • Best Month: {best_month}
# • Worst Month: {worst_month}

# 🛒 Product Insights:
# • Top-Selling Products: {top_products}

# 🌍 Regional Performance:
# • Best Performing Region: {best_region}
# • Regional Sales Breakdown:
# {region_breakdown}
# • Underperforming Region: {worst_region}

# 👥 Customer Demographics:
# • Best Performing Age Group: N/A
# """
#     return summary.strip()

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

# # Memory and Chains
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

# # Upload CSV
# st.sidebar.header("📁 Upload your sales data")
# uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
# knowledge_base = None

# # If uploaded, process file
# if uploaded_file:
#     try:
#         df = pd.read_csv(uploaded_file)
#         # Clean column names: lowercase, remove spaces
#         df.columns = [col.strip().lower() for col in df.columns]
#         expected_cols = {'date', 'product', 'sales', 'region'}

#         if expected_cols.issubset(df.columns):
#             # Convert date and extract month
#             df['Date'] = pd.to_datetime(df['date'])
#             df['Month'] = df['Date'].dt.to_period('M').astype(str)

#             # Rename for consistency with summary function
#             df.rename(columns={
#                 'product': 'Product',
#                 'sales': 'Sales',
#                 'region': 'Region'
#             }, inplace=True)

#             knowledge_base = generate_summary_from_df(df)
#             st.sidebar.success("✅ Summary generated from uploaded data!")

#             if st.checkbox("👁 Preview DataFrame"):
#                 st.dataframe(df.head())
#         else:
#             st.sidebar.error("❌ CSV must contain 'Date', 'Product', 'Sales', and 'Region' columns (case-insensitive).")
#     except Exception as e:
#         st.sidebar.error(f"❌ Error reading CSV: {e}")

# # Ask business question
# user_question = st.text_input("🔍 Ask a business question:")

# if user_question:
#     if not knowledge_base:
#         st.warning("Please upload a CSV with required columns to continue.")
#     else:
#         with st.spinner("Generating insights..."):
#             result = insightforge_chain.invoke({
#                 "data_summary": knowledge_base,
#                 "user_question": user_question
#             })

#         st.subheader("🧠 Insight")
#         st.success(result['insight'].strip())

#         st.subheader("💡 Recommendation")
#         st.info(result['recommendation'].strip())

#         if st.checkbox("🗂 Show Memory Log"):
#             st.text(memory.buffer)

# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt

# from langchain.chains import LLMChain, SequentialChain
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory

# # --- CONFIG ---
# st.set_page_config(page_title="InsightForge", page_icon="📊")
# st.title("📊 InsightForge - Business Intelligence Assistant")
# st.markdown("Ask a question about your company’s sales performance. Get insights + strategic recommendations.")

# # --- OpenAI KEY ---
# openai_api_key = st.secrets["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = openai_api_key

# # --- LangChain LLM Setup ---
# llm = OpenAI(temperature=0.2, model_name="gpt-3.5-turbo-instruct")

# # --- Memory & Chains ---
# memory = ConversationBufferMemory(input_key="user_question", memory_key="chat_history")

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

# insight_chain = LLMChain(llm=llm, prompt=insight_prompt, output_key="insight", memory=memory)
# recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt, output_key="recommendation")

# insightforge_chain = SequentialChain(
#     chains=[insight_chain, recommendation_chain],
#     input_variables=["data_summary", "user_question"],
#     output_variables=["insight", "recommendation"],
#     verbose=False
# )

# # --- Summary Generator ---
# def generate_summary_from_df(df):
#     total_sales = df['Sales'].sum()
#     avg_sale = df['Sales'].mean()
#     median_sale = df['Sales'].median()
#     std_sale = df['Sales'].std()

#     best_month = df.groupby('Month')['Sales'].sum().idxmax()
#     worst_month = df.groupby('Month')['Sales'].sum().idxmin()

#     product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
#     top_products = ", ".join([f"{p} (₹{v:.2f})" for p, v in product_sales.head(3).items()])

#     region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
#     best_region = region_sales.idxmax()
#     worst_region = region_sales.idxmin()
#     region_breakdown = "\n".join([f"• {region}: ₹{sales:,.2f}" for region, sales in region_sales.items()])

#     summary = f"""
# 📊 Business Insight Summary:

# • Total Sales: ₹{total_sales:,.2f}
# • Average Sale: ₹{avg_sale:,.2f}
# • Median Sale: ₹{median_sale:,.2f}
# • Standard Deviation: ₹{std_sale:,.2f}

# 🗓️ Time Trends:
# • Best Month: {best_month}
# • Worst Month: {worst_month}

# 🛒 Product Insights:
# • Top-Selling Products: {top_products}

# 🌍 Regional Performance:
# • Best Performing Region: {best_region}
# • Regional Sales Breakdown:
# {region_breakdown}
# • Underperforming Region: {worst_region}

# 👥 Customer Demographics:
# • Best Performing Age Group: N/A
# """
#     return summary.strip()

# # --- Chart Rendering ---
# def plot_monthly_sales(df):
#     import matplotlib.pyplot as plt

#     # Ensure 'Date' or 'Month' is datetime
#     if 'Date' in df.columns:
#         df['Date'] = pd.to_datetime(df['Date'])
#         df['Month'] = df['Date'].dt.to_period('M').astype(str)
#     elif 'Month' in df.columns:
#         df['Month'] = pd.to_datetime(df['Month']).dt.to_period('M').astype(str)

#     # Group and plot
#     monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()

#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.plot(monthly_sales['Month'], monthly_sales['Sales'], marker='o', linestyle='-')

#     ax.set_title("🗓 Monthly Sales Trend", fontsize=14, weight='bold')
#     ax.set_xlabel("Month", fontsize=12)
#     ax.set_ylabel("Sales", fontsize=12)

#     # Clean x-axis ticks
#     ax.set_xticks(monthly_sales['Month'][::max(1, len(monthly_sales)//12)])
#     ax.tick_params(axis='x', rotation=45)

#     st.pyplot(fig)


# def plot_sales_by_region(df):
#     chart_df = df.groupby('Region')['Sales'].sum().sort_values().reset_index()
#     plt.figure(figsize=(6, 3))
#     plt.bar(chart_df['Region'], chart_df['Sales'], color='skyblue')
#     plt.title("📊 Sales by Region")
#     plt.xlabel("Region")
#     plt.ylabel("Sales")
#     st.pyplot(plt)

# def plot_top_products(df):
#     chart_df = df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(5).reset_index()
#     plt.figure(figsize=(6, 3))
#     plt.bar(chart_df['Product'], chart_df['Sales'], color='green')
#     plt.title("🏆 Top 5 Products by Sales")
#     plt.xlabel("Product")
#     plt.ylabel("Sales")
#     st.pyplot(plt)

# # --- CSV Upload ---
# st.sidebar.header("📁 Upload your sales data")
# uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
# knowledge_base = None
# df = None

# if "qa_log" not in st.session_state:
#     st.session_state.qa_log = []

# if uploaded_file:
#     try:
#         df = pd.read_csv(uploaded_file)
#         df.columns = [col.strip().lower() for col in df.columns]
#         expected_cols = {'date', 'product', 'sales', 'region'}

#         if expected_cols.issubset(df.columns):
#             df['Date'] = pd.to_datetime(df['date'])
#             df['Month'] = df['Date'].dt.to_period('M').astype(str)
#             df.rename(columns={
#                 'product': 'Product',
#                 'sales': 'Sales',
#                 'region': 'Region'
#             }, inplace=True)
#             knowledge_base = generate_summary_from_df(df)
#             st.sidebar.success("✅ Summary generated from uploaded data.")
#             if st.sidebar.checkbox("👁 Preview DataFrame"):
#                 st.dataframe(df.head())
#         else:
#             st.sidebar.error("❌ CSV must include: 'Date', 'Product', 'Sales', 'Region'")
#     except Exception as e:
#         st.sidebar.error(f"❌ Error reading CSV: {e}")

# # --- Chat UI ---
# st.divider()
# st.markdown("### 🔍 Ask a business question:")

# with st.form("ask_form", clear_on_submit=True):
#     user_question = st.text_input("Your question:")
#     chart_choice = st.selectbox(
#         "📈 Which chart would you like to see?",
#         ["None", "Monthly Sales Trend", "Sales by Region", "Top Products", "All Charts"]
#     )
#     submitted = st.form_submit_button("Ask")

# # --- Run Chain on Submit ---
# if submitted and user_question:
#     if not knowledge_base:
#         st.warning("Please upload a valid CSV file first.")
#     else:
#         with st.spinner("Thinking..."):
#             result = insightforge_chain.invoke({
#                 "data_summary": knowledge_base,
#                 "user_question": user_question
#             })

#         # Append to chat log
#         st.session_state.qa_log.append({
#             "question": user_question,
#             "insight": result["insight"].strip(),
#             "recommendation": result["recommendation"].strip(),
#             "chart": chart_choice
#         })

# # --- Display History Log ---
# for entry in reversed(st.session_state.qa_log):
#     st.markdown("#### 🔍 Question")
#     st.info(entry["question"])
#     st.markdown("#### 🧠 Insight")
#     st.success(entry["insight"])
#     st.markdown("#### 💡 Recommendation")
#     st.info(entry["recommendation"])

#     if entry["chart"] != "None" and df is not None:
#         st.markdown("#### 📊 Visual Analysis")
#         if entry["chart"] == "Monthly Sales Trend":
#             plot_monthly_sales(df)
#         elif entry["chart"] == "Sales by Region":
#             plot_sales_by_region(df)
#         elif entry["chart"] == "Top Products":
#             plot_top_products(df)
#         elif entry["chart"] == "All Charts":
#             plot_monthly_sales(df)
#             plot_sales_by_region(df)
#             plot_top_products(df)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from langchain.chains import LLMChain, SequentialChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI  # NOTE: Replace with langchain_community.llms.OpenAI in future

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Load OpenAI key from Streamlit Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize LLM
llm = OpenAI(temperature=0.2, model_name="gpt-3.5-turbo-instruct")

# STEP 1: Load and Embed the 4 reference PDFs
pdf_paths = [
    "AI business model innovation.pdf",
    "BI approaches.pdf",
    "Time-Series-Data-Prediction-using-IoT-and-Machine-Le_2020_Procedia-Computer-.pdf",
    "Walmarts sales data analysis.pdf"
]

all_docs = []
for path in pdf_paths:
    if os.path.exists(path):
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(all_docs)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

# STEP 2: RAG chain setup
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

# Helper: Generate summary
def generate_summary_from_df(df):
    total_sales = df['Sales'].sum()
    avg_sale = df['Sales'].mean()
    median_sale = df['Sales'].median()
    std_sale = df['Sales'].std()

    df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')
    month_sales = df.groupby('Month')['Sales'].sum()
    best_month = month_sales.idxmax()
    worst_month = month_sales.idxmin()

    product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
    top_products = ", ".join([f"{p} (₹{v:,.2f})" for p, v in product_sales.head(3).items()])

    region_sales = df.groupby('Region')['Sales'].sum()
    best_region = region_sales.idxmax()

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

👥 Customer Demographics:
• Best Performing Age Group: N/A
"""
    return summary.strip()

# Prompts
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

# Chains
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
st.title(":bar_chart: InsightForge - Business Intelligence Assistant")

st.markdown("Ask a question about your company’s sales performance. Get insights + strategic recommendations.")

# Upload CSV
st.sidebar.header("📁 Upload your sales data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

knowledge_base = None
df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [col.strip() for col in df.columns]
        expected_cols = {'Date', 'Product', 'Sales', 'Region'}
        if expected_cols.issubset(df.columns):
            knowledge_base = generate_summary_from_df(df)
            st.sidebar.success("✅ Summary generated from uploaded data!")
            if st.sidebar.checkbox("👁️ Preview DataFrame"):
                st.dataframe(df)
        else:
            st.sidebar.error("CSV must include 'Date', 'Product', 'Sales', 'Region'")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Question box
user_question = st.text_input("🔍 Ask a business question:")

if user_question:
    if not knowledge_base:
        st.warning("Please upload a valid CSV to proceed.")
    else:
        with st.spinner("Generating insights..."):
            result = insightforge_chain.invoke({
                "data_summary": knowledge_base,
                "user_question": user_question
            })

        st.subheader("🧑‍🎓 Insight")
        st.success(result['insight'].strip())

        st.subheader("💡 Recommendation")
        st.info(result['recommendation'].strip())

        if st.checkbox("🗂 Show Memory Log"):
            st.text(memory.buffer)

# Optional RAG box for PDF reference questions
with st.expander("📚 Ask about Reference PDFs (AI/BI/IoT/Sales Reports)"):
    reference_question = st.text_input("Ask a research-based question:", key="ref_q")
    if reference_question:
        with st.spinner("🔍 Searching documents..."):
            rag_result = rag_chain.invoke({"query": reference_question})
        st.subheader("📖 RAG Answer")
        st.info(rag_result['result'])
        with st.expander("📎 Sources"):
            for doc in rag_result['source_documents']:
                st.markdown(f"- `{doc.metadata.get('source', 'Unknown Source')}`")



    

