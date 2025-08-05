import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.evaluation.qa import QAEvalChain

# Streamlit App Title
st.title("📊 InsightForge: AI-Powered BI Assistant")

# Sidebar: API key input
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Sidebar: Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload your sales data CSV", type="csv")

# Sidebar: Upload additional PDFs
additional_pdfs = st.sidebar.file_uploader("Upload additional reference PDFs", type="pdf", accept_multiple_files=True)

# Load and summarize CSV
def generate_advanced_summary(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')

    total = df['Sales'].sum()
    avg = df['Sales'].mean()
    best_month = df.groupby('Month')['Sales'].sum().idxmax().strftime('%B %Y')
    top_product = df.groupby('Product')['Sales'].sum().idxmax()
    worst_product = df.groupby('Product')['Sales'].sum().idxmin()
    best_region = df.groupby('Region')['Sales'].sum().idxmax()
    worst_region = df.groupby('Region')['Sales'].sum().idxmin()

    return f"""
📈 **Sales Summary**
- Total Sales: ₹{total:,.0f}
- Average Sale: ₹{avg:.2f}
- Best Month: {best_month}
- Top Product: {top_product}
- Lowest Selling Product: {worst_product}
- Best Performing Region: {best_region}
- Worst Performing Region: {worst_region}
"""

# Chart functions

def plot_region_sales(df):
    region_sales = df.groupby('Region')['Sales'].sum().sort_values()
    fig, ax = plt.subplots()
    bars = ax.bar(region_sales.index, region_sales.values, color='skyblue')
    lowest = region_sales.idxmin()
    bars[list(region_sales.index).index(lowest)].set_color('red')
    ax.set_title('Sales by Region')
    ax.set_ylabel('Total Sales')
    st.pyplot(fig)

def plot_product_sales(df):
    product_sales = df.groupby('Product')['Sales'].sum().sort_values()
    fig, ax = plt.subplots()
    bars = ax.bar(product_sales.index, product_sales.values, color='lightgreen')
    lowest = product_sales.idxmin()
    bars[list(product_sales.index).index(lowest)].set_color('orange')
    ax.set_title('Sales by Product')
    ax.set_ylabel('Total Sales')
    st.pyplot(fig)

# Load PDFs and create FAISS vectorstore
@st.cache_resource
def load_vectorstore(uploaded_files):
    default_pdfs = [
        "AI-business-model-innovation.pdf",
        "BI-approaches.pdf",
        "Time-Series-Data-Prediction.pdf",
        "Walmart-sales-analysis.pdf"
    ]
    all_docs = []

    for path in default_pdfs:
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    if uploaded_files:
        for pdf in uploaded_files:
            with open(f"/tmp/{pdf.name}", "wb") as f:
                f.write(pdf.read())
            loader = PyPDFLoader(f"/tmp/{pdf.name}")
            all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)

vectorstore = load_vectorstore(additional_pdfs)

# Show uploaded data summary
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    summary = generate_advanced_summary(df)
    st.sidebar.success("✅ Sales data uploaded")
    st.markdown(summary)

    # Initialize LLM and memory
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    st.subheader("📊 Dynamic Charts (based on your question)")

    # Ask a question
    st.subheader("💬 Ask a Business Question")
    user_input = st.text_input("Type your question...")

    if user_input:
        # Retrieve relevant PDF context
        docs = vectorstore.similarity_search(user_input, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Full prompt injection
        full_prompt = f"""
You are a business analyst.

Use the SALES SUMMARY and PDF CONTEXT below to answer the question. Prioritize insights from the sales summary. If no answer is found, say \"I don't know\".

SALES SUMMARY:
{summary}

PDF CONTEXT:
{context}

QUESTION:
{user_input}
"""
        # Run RAG conversation
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            verbose=False
        )
        response = rag_chain.run(full_prompt)
        st.markdown(f"**🧠 InsightForge:** {response}")

        # Dynamic chart trigger based on question intent
        if "region" in user_input.lower():
            plot_region_sales(df)
        elif "product" in user_input.lower() or "widget" in user_input.lower():
            plot_product_sales(df)

        # Optional: QA Evaluation
        if st.button("🧪 Evaluate Answer"):
            eval_chain = QAEvalChain.from_llm(llm)
            examples = [{"query": user_input, "answer": response, "context": context}]
            predictions = [{"result": response}]
            grade = eval_chain.evaluate(examples, predictions, prediction_key="result")
            st.markdown(f"**🎓 Evaluation Result:** {grade[0]['results']}")
