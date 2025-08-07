# Version 3.5

# Final Insf4 — with Intent Detection, Statistical Metrics, Filters

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from io import BytesIO

from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.evaluation.qa import QAEvalChain

# Page Setup
st.set_page_config(page_title="InsightForge", page_icon="📊", layout="wide")
st.title("📊 InsightForge: AI-Powered BI Assistant")

# Sidebar: API Key and File Uploads
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload your sales data CSV", type="csv")
additional_pdfs = st.sidebar.file_uploader("Upload reference PDFs", type="pdf", accept_multiple_files=True)

# Initialize Chat History
if "chat_pairs" not in st.session_state:
    st.session_state.chat_pairs = []

# Load PDFs and Create Vectorstore
@st.cache_resource
def load_vectorstore(uploaded_files):
    all_docs = []
    if uploaded_files:
        for pdf in uploaded_files:
            with open(f"/tmp/{pdf.name}", "wb") as f:
                f.write(pdf.read())
            loader = PyPDFLoader(f"/tmp/{pdf.name}")
            all_docs.extend(loader.load())
    if not all_docs:
        return None, []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    return FAISS.from_documents(chunks, embeddings), chunks

vectorstore, all_chunks = load_vectorstore(additional_pdfs)


# Summary + Stats

def generate_advanced_summary(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    total = df['Sales'].sum()
    avg = df['Sales'].mean()
    median = df['Sales'].median()
    std_dev = df['Sales'].std()
    best_month = df.groupby('Month')['Sales'].sum().idxmax().strftime('%B %Y')
    top_product = df.groupby('Product')['Sales'].sum().idxmax()
    worst_product = df.groupby('Product')['Sales'].sum().idxmin()
    best_region = df.groupby('Region')['Sales'].sum().idxmax()
    worst_region = df.groupby('Region')['Sales'].sum().idxmin()
    return f"""
📈 **Sales Summary**
- Total Sales: ₹{total:,.0f}
- Average Sale: ₹{avg:.2f}
- Median Sale: ₹{median:.2f}
- Std Deviation: ₹{std_dev:.2f}
- Best Month: {best_month}
- Top Product: {top_product}
- Lowest Selling Product: {worst_product}
- Best Performing Region: {best_region}
- Worst Performing Region: {worst_region}
"""

# Suggestion Generator
def suggest_questions(summary):
    llm_temp = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)
    suggestion_prompt = f"""Given this sales summary:
{summary}

Suggest 3 smart business questions to ask."""
    return llm_temp.predict(suggestion_prompt)

# Chart Renderer
def render_chart(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    st.image(buf)

# Plotters
def plot_region_sales(df):
    region_sales = df.groupby('Region')['Sales'].sum().sort_values()
    fig, ax = plt.subplots(figsize=(3.1, 1.7))
    bars = ax.bar(region_sales.index, region_sales.values, color='skyblue')
    bars[list(region_sales.index).index(region_sales.idxmin())].set_color('red')
    ax.set_title('Sales by Region', fontsize=8)
    ax.set_ylabel('Total Sales', fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    render_chart(fig)

def plot_product_sales(df):
    product_sales = df.groupby('Product')['Sales'].sum().sort_values()
    fig, ax = plt.subplots(figsize=(3.1, 1.7))
    bars = ax.bar(product_sales.index, product_sales.values, color='lightgreen')
    bars[list(product_sales.index).index(product_sales.idxmin())].set_color('orange')
    ax.set_title('Sales by Product', fontsize=8)
    ax.set_ylabel('Total Sales', fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    render_chart(fig)

def plot_monthly_trend(df):
    monthly = df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum()
    fig, ax = plt.subplots(figsize=(3.1, 1.7))
    monthly.plot(ax=ax, marker='o')
    ax.set_title('Monthly Sales Trend', fontsize=8)
    ax.set_ylabel('Sales', fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    render_chart(fig)

# Main Flow
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])

    region_options = ["All"] + sorted(df['Region'].dropna().unique())
    product_options = ["All"] + sorted(df['Product'].dropna().unique())

    summary = generate_advanced_summary(df)
    st.sidebar.success("✅ Sales data uploaded")
    st.markdown(summary)

    if st.sidebar.button("💡 Suggest Questions"):
        suggestions = suggest_questions(summary)
        st.sidebar.markdown("**🤔 Suggested Questions:**")
        st.sidebar.markdown(suggestions)

    st.subheader("📊 Monthly Sales Trend")
    plot_monthly_trend(df)

    st.subheader("💬 Ask a Business Question")
    user_input = st.text_input("Type your question and press Enter")

    if user_input and vectorstore:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        docs = vectorstore.similarity_search(user_input, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Auto intent detection
        intent_prompt = f"Classify this business question: '{user_input}'\nChoose from: trend, comparison, strategy, forecast, region, product, customer, other."
        intent = llm.predict(intent_prompt).strip()
        st.caption(f"🧭 Detected intent: `{intent}`")

        full_prompt = f"""
Answer the question based on the SALES SUMMARY and PDF CONTEXT below.
Be brief. Say "I don't know" if unclear.

SALES SUMMARY:
{summary}

PDF CONTEXT:
{context}

QUESTION:
{user_input}
"""

        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            verbose=False
        )
        response = rag_chain.run(full_prompt)

        st.session_state.chat_pairs.insert(0, {"question": user_input, "answer": response})
        st.markdown(f"**🧠 InsightForge:** {response}")

        # Charts
        if intent == "region":
            plot_region_sales(df)
        elif intent == "product":
            plot_product_sales(df)

        if st.button("🧪 Evaluate Answer"):
            eval_chain = QAEvalChain.from_llm(llm)
            examples = [{"query": user_input, "answer": response, "context": context}]
            predictions = [{"result": response}]
            grade = eval_chain.evaluate(examples, predictions, prediction_key="result")
            st.markdown(f"**🎓 Evaluation Result:** {grade[0]['results']}")

    st.subheader("🗂️ Chat History")
    for pair in st.session_state.chat_pairs[::-1]:
        with st.chat_message("user"):
            st.markdown(pair['question'])
        with st.chat_message("assistant"):
            st.markdown(pair['answer'])
