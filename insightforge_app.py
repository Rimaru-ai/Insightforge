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

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from langchain.chains import LLMChain, SequentialChain, RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory
# from langchain_community.llms import OpenAI
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings

# # Set up OpenAI key
# openai_api_key = st.secrets["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = openai_api_key

# # Initialize LLM
# llm = OpenAI(temperature=0.2, model_name="gpt-3.5-turbo-instruct")

# # ---------------- PDF INGESTION + EMBEDDING -------------------
# @st.cache_resource(show_spinner="🔄 Processing PDFs...")
# def load_reference_rag():
#     pdf_paths = [
#         "AI business model innovation.pdf",
#         "BI approaches.pdf",
#         "Time-Series-Data-Prediction-using-IoT-and-Machine-Le_2020_Procedia-Computer-.pdf",
#         "Walmarts sales data analysis.pdf"
#     ]

#     all_docs = []
#     for path in pdf_paths:
#         if os.path.exists(path):
#             loader = PyPDFLoader(path)
#             docs = loader.load()
#             st.sidebar.success(f"✅ Loaded: {path}")
#             all_docs.extend(docs)
#         else:
#             st.sidebar.warning(f"⚠️ File not found: {path}")

#     if not all_docs:
#         return None

#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#     chunks = splitter.split_documents(all_docs)
#     embeddings = OpenAIEmbeddings()
#     vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
#     return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# rag_chain = load_reference_rag()

# # ---------------- CSV DATA INSIGHT FLOW -------------------

# def generate_summary_from_df(df):
#     total_sales = df['Sales'].sum()
#     avg_sale = df['Sales'].mean()
#     median_sale = df['Sales'].median()
#     std_sale = df['Sales'].std()

#     best_month = df.groupby('Month')['Sales'].sum().idxmax()
#     worst_month = df.groupby('Month')['Sales'].sum().idxmin()

#     product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
#     top_products = ", ".join([f"{p} (₹{v:.2f})" for p, v in product_sales.head(3).items()])

#     region_sales = df.groupby('Region')['Sales'].sum()
#     best_region = region_sales.idxmax()

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

# # Chains
# memory = ConversationBufferMemory(input_key="user_question", memory_key="chat_history")
# insight_chain = LLMChain(llm=llm, prompt=insight_prompt, output_key="insight", memory=memory)
# recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt, output_key="recommendation")
# insightforge_chain = SequentialChain(
#     chains=[insight_chain, recommendation_chain],
#     input_variables=["data_summary", "user_question"],
#     output_variables=["insight", "recommendation"],
#     verbose=False
# )

# # ---------------- STREAMLIT UI -------------------
# st.set_page_config(page_title="InsightForge", page_icon="📊")
# st.title("📊 InsightForge - Business Intelligence Assistant")

# st.markdown("Ask a question about your company’s sales performance. Get insights + strategic recommendations.")

# # Upload CSV
# st.sidebar.header("📁 Upload your sales data")
# uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
# knowledge_base = None

# if uploaded_file:
#     try:
#         df = pd.read_csv(uploaded_file)
#         df.columns = [col.strip().lower() for col in df.columns]
#         expected_cols = {'date', 'product', 'sales', 'region'}
#         if expected_cols.issubset(df.columns):
#             df.rename(columns={
#                 'date': 'Month',
#                 'product': 'Product',
#                 'sales': 'Sales',
#                 'region': 'Region'
#             }, inplace=True)
#             df['Month'] = pd.to_datetime(df['Month']).dt.to_period('M').astype(str)
#             knowledge_base = generate_summary_from_df(df)
#             st.sidebar.success("✅ Summary generated from uploaded data!")
#         else:
#             st.sidebar.error("❌ CSV must contain 'Date', 'Product', 'Sales', and 'Region' columns (case-insensitive).")
#     except Exception as e:
#         st.sidebar.error(f"❌ Error reading CSV: {e}")

# # Choose section
# tab = st.radio("Select interaction type:", ["📈 Ask Business Question", "📚 Ask Reference (PDF) Question"])

# if tab == "📈 Ask Business Question":
#     user_question = st.text_input("🔍 Ask a business question:", key="user_input")
#     chart_options = ["None", "Monthly Sales Trend", "Top Products", "Sales by Region"]
#     selected_chart = st.selectbox("📊 Which chart would you like to see?", chart_options)

#     if user_question:
#         if not knowledge_base:
#             st.warning("Please upload a CSV with required columns to continue.")
#         else:
#             with st.spinner("Generating insights..."):
#                 result = insightforge_chain.invoke({
#                     "data_summary": knowledge_base,
#                     "user_question": user_question
#                 })
#             st.subheader("🧑‍🎓 Insight")
#             st.success(result['insight'].strip())
#             st.subheader("💡 Recommendation")
#             st.info(result['recommendation'].strip())

#             if selected_chart != "None":
#                 st.subheader("📊 Visual Analysis")
#                 if selected_chart == "Monthly Sales Trend":
#                     monthly_sales = df.groupby('Month')['Sales'].sum()
#                     fig, ax = plt.subplots()
#                     monthly_sales.plot(ax=ax, marker='o')
#                     ax.set_title("🗓 Monthly Sales Trend")
#                     ax.set_xlabel("Month")
#                     ax.set_ylabel("Sales")
#                     plt.xticks(rotation=45)
#                     st.pyplot(fig)

#                 elif selected_chart == "Top Products":
#                     top_products = df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(10)
#                     fig, ax = plt.subplots()
#                     top_products.plot(kind='bar', ax=ax)
#                     ax.set_title("🏆 Top-Selling Products")
#                     ax.set_ylabel("Sales")
#                     st.pyplot(fig)

#                 elif selected_chart == "Sales by Region":
#                     region_sales = df.groupby('Region')['Sales'].sum()
#                     fig, ax = plt.subplots()
#                     region_sales.plot(kind='pie', ax=ax, autopct='%1.1f%%')
#                     ax.set_ylabel("")
#                     ax.set_title("🌍 Sales by Region")
#                     st.pyplot(fig)

#             if st.checkbox("🗂 Show Memory Log"):
#                 st.text(memory.buffer)

# elif tab == "📚 Ask Reference (PDF) Question":
#     if not rag_chain:
#         st.error("❌ No reference documents found to answer questions. Please check deployment.")
#     else:
#         pdf_question = st.text_input("📚 Ask a question based on uploaded reference documents:")
#         if pdf_question:
#             with st.spinner("Searching reference materials..."):
#                 response = rag_chain.run(pdf_question)
#                 st.subheader("📖 Answer from References")
#                 st.write(response)




# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from langchain.chains import LLMChain, SequentialChain, RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory
# from langchain_community.llms import OpenAI
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings

# # Set up OpenAI key
# openai_api_key = st.secrets["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = openai_api_key

# # Initialize LLM
# llm = OpenAI(temperature=0.2, model_name="gpt-3.5-turbo-instruct")

# # ---------------- PDF INGESTION + EMBEDDING -------------------
# @st.cache_resource(show_spinner="🔄 Processing PDFs...")
# def load_reference_rag():
#     pdf_paths = [
#         "AI business model innovation.pdf",
#         "BI approaches.pdf",
#         "Time-Series-Data-Prediction-using-IoT-and-Machine-Le_2020_Procedia-Computer-.pdf",
#         "Walmarts sales data analysis.pdf"
#     ]

#     all_docs = []
#     for path in pdf_paths:
#         if os.path.exists(path):
#             loader = PyPDFLoader(path)
#             docs = loader.load()
#             st.sidebar.success(f"✅ Loaded: {path}")
#             all_docs.extend(docs)
#         else:
#             st.sidebar.warning(f"⚠️ File not found: {path}")

#     if not all_docs:
#         return None

#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#     chunks = splitter.split_documents(all_docs)
#     embeddings = OpenAIEmbeddings()
#     vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
#     return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# rag_chain = load_reference_rag()

# # ---------------- CSV DATA INSIGHT FLOW -------------------

# def generate_summary_from_df(df):
#     total_sales = df['Sales'].sum()
#     avg_sale = df['Sales'].mean()
#     median_sale = df['Sales'].median()
#     std_sale = df['Sales'].std()

#     best_month = df.groupby('Month')['Sales'].sum().idxmax()
#     worst_month = df.groupby('Month')['Sales'].sum().idxmin()

#     product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
#     top_products = ", ".join([f"{p} (₹{v:.2f})" for p, v in product_sales.head(3).items()])

#     region_sales = df.groupby('Region')['Sales'].sum()
#     best_region = region_sales.idxmax()

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

# # Chains
# memory = ConversationBufferMemory(input_key="user_question", memory_key="chat_history")
# insight_chain = LLMChain(llm=llm, prompt=insight_prompt, output_key="insight", memory=memory)
# recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt, output_key="recommendation")
# insightforge_chain = SequentialChain(
#     chains=[insight_chain, recommendation_chain],
#     input_variables=["data_summary", "user_question"],
#     output_variables=["insight", "recommendation"],
#     verbose=False
# )

# # ---------------- STREAMLIT UI -------------------
# st.set_page_config(page_title="InsightForge", page_icon="📊")
# st.title("📊 InsightForge - Business Intelligence Assistant")

# st.markdown("Ask a question about your company’s sales performance. Get insights + strategic recommendations.")

# # Upload CSV
# st.sidebar.header("📁 Upload your sales data")
# uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
# knowledge_base = None

# if uploaded_file:
#     try:
#         df = pd.read_csv(uploaded_file)
#         df.columns = [col.strip().lower() for col in df.columns]
#         expected_cols = {'date', 'product', 'sales', 'region'}
#         if expected_cols.issubset(df.columns):
#             df.rename(columns={
#                 'date': 'Month',
#                 'product': 'Product',
#                 'sales': 'Sales',
#                 'region': 'Region'
#             }, inplace=True)
#             df['Month'] = pd.to_datetime(df['Month']).dt.to_period('M').astype(str)
#             knowledge_base = generate_summary_from_df(df)
#             st.sidebar.success("✅ Summary generated from uploaded data!")
#         else:
#             st.sidebar.error("❌ CSV must contain 'Date', 'Product', 'Sales', and 'Region' columns (case-insensitive).")
#     except Exception as e:
#         st.sidebar.error(f"❌ Error reading CSV: {e}")

# # Choose section
# tab = st.radio("Select interaction type:", ["📈 Ask Business Question", "📚 Ask Reference (PDF) Question", "🧪 Apply Learned Analysis"])

# if tab == "📈 Ask Business Question":
#     user_question = st.text_input("🔍 Ask a business question:", key="user_input")
#     chart_options = ["None", "Monthly Sales Trend", "Top Products", "Sales by Region"]
#     selected_chart = st.selectbox("📊 Which chart would you like to see?", chart_options)

#     if user_question:
#         if not knowledge_base:
#             st.warning("Please upload a CSV with required columns to continue.")
#         else:
#             with st.spinner("Generating insights..."):
#                 result = insightforge_chain.invoke({
#                     "data_summary": knowledge_base,
#                     "user_question": user_question
#                 })
#             st.subheader("🧑‍🎓 Insight")
#             st.success(result['insight'].strip())
#             st.subheader("💡 Recommendation")
#             st.info(result['recommendation'].strip())

#             if selected_chart != "None":
#                 st.subheader("📊 Visual Analysis")
#                 if selected_chart == "Monthly Sales Trend":
#                     monthly_sales = df.groupby('Month')['Sales'].sum()
#                     fig, ax = plt.subplots()
#                     monthly_sales.plot(ax=ax, marker='o')
#                     ax.set_title("🗓 Monthly Sales Trend")
#                     ax.set_xlabel("Month")
#                     ax.set_ylabel("Sales")
#                     plt.xticks(rotation=45)
#                     st.pyplot(fig)

#                 elif selected_chart == "Top Products":
#                     top_products = df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(10)
#                     fig, ax = plt.subplots()
#                     top_products.plot(kind='bar', ax=ax)
#                     ax.set_title("🏆 Top-Selling Products")
#                     ax.set_ylabel("Sales")
#                     st.pyplot(fig)

#                 elif selected_chart == "Sales by Region":
#                     region_sales = df.groupby('Region')['Sales'].sum()
#                     fig, ax = plt.subplots()
#                     region_sales.plot(kind='pie', ax=ax, autopct='%1.1f%%')
#                     ax.set_ylabel("")
#                     ax.set_title("🌍 Sales by Region")
#                     st.pyplot(fig)

#             if st.checkbox("🗂 Show Memory Log"):
#                 st.text(memory.buffer)

# elif tab == "📚 Ask Reference (PDF) Question":
#     if not rag_chain:
#         st.error("❌ No reference documents found to answer questions. Please check deployment.")
#     else:
#         pdf_question = st.text_input("📚 Ask a question based on uploaded reference documents:")
#         if pdf_question:
#             with st.spinner("Searching reference materials..."):
#                 response = rag_chain.run(pdf_question)
#                 st.subheader("📖 Answer from References")
#                 st.write(response)

# elif tab == "🧪 Apply Learned Analysis":
#     if not knowledge_base or not rag_chain:
#         st.warning("Please upload a CSV file and ensure PDF knowledge is available.")
#     else:
#         with st.spinner("🔍 Reading reference techniques and applying to your data..."):
#             context_prompt = f"""
# The user has uploaded a dataset summarised as follows:
# {knowledge_base}

# Based on the reference research documents available, suggest what analytical technique(s) could be applied to this data, and show what insights might emerge if we applied those methods.
# """
#             response = rag_chain.run(context_prompt)
#             st.subheader("📘 Technique-Based Insight")
#             st.write(response)



# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from langchain.chains import LLMChain, SequentialChain, RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory
# from langchain_community.llms import OpenAI
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings

# # Set up OpenAI key
# openai_api_key = st.secrets["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = openai_api_key

# # Initialize LLM
# llm = OpenAI(temperature=0.2, model_name="gpt-3.5-turbo-instruct")

# # ---------------- PDF INGESTION + EMBEDDING -------------------
# @st.cache_resource(show_spinner="🔄 Processing PDFs...")
# def load_reference_rag():
#     pdf_paths = [
#         "AI business model innovation.pdf",
#         "BI approaches.pdf",
#         "Time-Series-Data-Prediction-using-IoT-and-Machine-Le_2020_Procedia-Computer-.pdf",
#         "Walmarts sales data analysis.pdf"
#     ]

#     all_docs = []
#     for path in pdf_paths:
#         if os.path.exists(path):
#             loader = PyPDFLoader(path)
#             docs = loader.load()
#             st.sidebar.success(f"✅ Loaded: {path}")
#             all_docs.extend(docs)
#         else:
#             st.sidebar.warning(f"⚠️ File not found: {path}")

#     if not all_docs:
#         return None

#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#     chunks = splitter.split_documents(all_docs)
#     embeddings = OpenAIEmbeddings()
#     vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
#     return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# rag_chain = load_reference_rag()

# # ---------------- CSV DATA INSIGHT FLOW -------------------

# def generate_summary_from_df(df):
#     total_sales = df['Sales'].sum()
#     avg_sale = df['Sales'].mean()
#     median_sale = df['Sales'].median()
#     std_sale = df['Sales'].std()

#     best_month = df.groupby('Month')['Sales'].sum().idxmax()
#     worst_month = df.groupby('Month')['Sales'].sum().idxmin()

#     product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
#     top_products = ", ".join([f"{p} (₹{v:.2f})" for p, v in product_sales.head(3).items()])

#     region_sales = df.groupby('Region')['Sales'].sum()
#     best_region = region_sales.idxmax()

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

# 👥 Customer Demographics:
# • Best Performing Age Group: N/A
# """
#     return summary.strip()

# # Forecasting Template (Phase 2)
# forecast_prompt = PromptTemplate(
#     input_variables=["data_summary"],
#     template="""
# You are a time series forecasting expert.

# Given the following sales data summary:
# {data_summary}

# Generate a forecast for the next 3 months of total sales. Include seasonality or trends observed if any.
# Forecast:"""
# )
# forecast_chain = LLMChain(llm=llm, prompt=forecast_prompt, output_key="forecast")

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

# # Chains
# memory = ConversationBufferMemory(input_key="user_question", memory_key="chat_history")
# insight_chain = LLMChain(llm=llm, prompt=insight_prompt, output_key="insight", memory=memory)
# recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt, output_key="recommendation")
# insightforge_chain = SequentialChain(
#     chains=[insight_chain, recommendation_chain],
#     input_variables=["data_summary", "user_question"],
#     output_variables=["insight", "recommendation"],
#     verbose=False
# )

# # ---------------- STREAMLIT UI -------------------
# st.set_page_config(page_title="InsightForge", page_icon="📊")
# st.title("📊 InsightForge - Business Intelligence Assistant")

# st.markdown("Ask a question about your company’s sales performance. Get insights + strategic recommendations.")

# # Upload CSV
# st.sidebar.header("📁 Upload your sales data")
# uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
# knowledge_base = None

# if uploaded_file:
#     try:
#         df = pd.read_csv(uploaded_file)
#         df.columns = [col.strip().lower() for col in df.columns]
#         expected_cols = {'date', 'product', 'sales', 'region'}
#         if expected_cols.issubset(df.columns):
#             df.rename(columns={
#                 'date': 'Month',
#                 'product': 'Product',
#                 'sales': 'Sales',
#                 'region': 'Region'
#             }, inplace=True)
#             df['Month'] = pd.to_datetime(df['Month']).dt.to_period('M').astype(str)
#             knowledge_base = generate_summary_from_df(df)
#             st.sidebar.success("✅ Summary generated from uploaded data!")
#         else:
#             st.sidebar.error("❌ CSV must contain 'Date', 'Product', 'Sales', and 'Region' columns (case-insensitive).")
#     except Exception as e:
#         st.sidebar.error(f"❌ Error reading CSV: {e}")

# # Choose section
# tab = st.radio("Select interaction type:", ["📈 Ask Business Question", "📚 Ask Reference (PDF) Question", "🧠 Apply Learned Analysis"])

# if tab == "📈 Ask Business Question":
#     user_question = st.text_input("🔍 Ask a business question:", key="user_input")
#     chart_options = ["None", "Monthly Sales Trend", "Top Products", "Sales by Region"]
#     selected_chart = st.selectbox("📊 Which chart would you like to see?", chart_options)

#     if user_question:
#         if not knowledge_base:
#             st.warning("Please upload a CSV with required columns to continue.")
#         else:
#             with st.spinner("Generating insights..."):
#                 result = insightforge_chain.invoke({
#                     "data_summary": knowledge_base,
#                     "user_question": user_question
#                 })
#             st.subheader("🧑‍🎓 Insight")
#             st.success(result['insight'].strip())
#             st.subheader("💡 Recommendation")
#             st.info(result['recommendation'].strip())

#             if selected_chart != "None":
#                 st.subheader("📊 Visual Analysis")
#                 if selected_chart == "Monthly Sales Trend":
#                     monthly_sales = df.groupby('Month')['Sales'].sum()
#                     fig, ax = plt.subplots()
#                     monthly_sales.plot(ax=ax, marker='o')
#                     ax.set_title("🗓 Monthly Sales Trend")
#                     ax.set_xlabel("Month")
#                     ax.set_ylabel("Sales")
#                     plt.xticks(rotation=45)
#                     st.pyplot(fig)

#                 elif selected_chart == "Top Products":
#                     top_products = df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(10)
#                     fig, ax = plt.subplots()
#                     top_products.plot(kind='bar', ax=ax)
#                     ax.set_title("🏆 Top-Selling Products")
#                     ax.set_ylabel("Sales")
#                     st.pyplot(fig)

#                 elif selected_chart == "Sales by Region":
#                     region_sales = df.groupby('Region')['Sales'].sum()
#                     fig, ax = plt.subplots()
#                     region_sales.plot(kind='pie', ax=ax, autopct='%1.1f%%')
#                     ax.set_ylabel("")
#                     ax.set_title("🌍 Sales by Region")
#                     st.pyplot(fig)

#             if st.checkbox("🗂 Show Memory Log"):
#                 st.text(memory.buffer)

# elif tab == "📚 Ask Reference (PDF) Question":
#     if not rag_chain:
#         st.error("❌ No reference documents found to answer questions. Please check deployment.")
#     else:
#         pdf_question = st.text_input("📚 Ask a question based on uploaded reference documents:")
#         if pdf_question:
#             with st.spinner("Searching reference materials..."):
#                 response = rag_chain.run(pdf_question)
#                 st.subheader("📖 Answer from References")
#                 st.write(response)

# elif tab == "🧠 Apply Learned Analysis":
#     if not rag_chain or not knowledge_base:
#         st.warning("Upload both sales data and reference PDFs to proceed.")
#     else:
#         with st.spinner("Extracting techniques and applying them to your data..."):
#             learned_prompt = PromptTemplate(
#                 input_variables=["reference_techniques", "data_summary"],
#                 template="""
# You are a data analyst who has read these research insights:
# {reference_techniques}

# Based on these techniques and the following dataset summary:
# {data_summary}

# Generate a short report that demonstrates how the insights can be applied to the sales data.
# """
#             )
#             reference_text = rag_chain.run("What techniques or analytical approaches are used in these documents?")
#             chain = LLMChain(llm=llm, prompt=learned_prompt, output_key="learned_output")
#             output = chain.run({
#                 "reference_techniques": reference_text,
#                 "data_summary": knowledge_base
#             })
#         st.subheader("📘 Technique-Based Insight")
#         st.markdown(output)


import streamlit as st
import pandas as pd
import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up OpenAI API key
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
os.environ["OPENAI_API_KEY"] = openai_api_key

# App title
st.title("📊 InsightForge: AI-Powered BI Assistant")

# Load PDFs and build FAISS vector store (do this only once, cache it)
@st.cache_resource
def build_vectorstore():
    pdf_paths = [
        "AI-business-model-innovation.pdf",
        "BI-approaches.pdf",
        "Time-Series-Data-Prediction.pdf",
        "Walmart-sales-analysis.pdf"
    ]
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = build_vectorstore()

# Upload and summarize sales CSV
def generate_summary(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    total_sales = df['Sales'].sum()
    avg = df['Sales'].mean()
    best_month = df.groupby('Month')['Sales'].sum().idxmax().strftime('%B %Y')
    top_product = df.groupby('Product')['Sales'].sum().idxmax()

    return f"""
Total Sales: ₹{total_sales:,}
Average Sale: ₹{avg:.2f}
Best Month: {best_month}
Top Product: {top_product}
"""

# Upload sales data
uploaded_file = st.sidebar.file_uploader("Upload your sales data CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    summary = generate_summary(df)
    st.sidebar.success("Sales data uploaded.")
    st.subheader("📄 Data Summary")
    st.markdown(summary)

    # Memory + Retrieval-based chat
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=False
    )

    # Chat interface
    st.subheader("💬 Ask a Question")
    user_input = st.text_input("Type your question...")

    if user_input:
        full_prompt = f"{summary}\n\n{user_input}"
        response = rag_chain.run(full_prompt)
        st.markdown(f"**🧠 InsightForge:** {response}")






    

