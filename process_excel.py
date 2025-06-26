
import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

EXPORT_FOLDER = "Org_folder"

def prettify_column_name(col):
    return col.replace("_", " ").replace("-", " ").replace(".", " ").title()

def build_document_content(df: pd.DataFrame) -> pd.DataFrame:
    contents = []

    for _, row in df.iterrows():
        parts = []

        for col in df.columns:
            val = row.get(col)

            if pd.isna(val):
                continue

            label = prettify_column_name(col)

            if "date" in col.lower():
                try:
                    val = pd.to_datetime(val).date()
                except Exception:
                    pass

            parts.append(f"{label}: {val}")

        summary = ". ".join(str(p) for p in parts if pd.notna(p)) + "."
        contents.append(summary)

    df["document_content"] = contents
    return df

def prepare_rag_data_for_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)

        # Remove original_* columns
        df = df.loc[:, ~df.columns.str.startswith('original_')]

        # Generate readable summaries
        df = build_document_content(df)

        loader = DataFrameLoader(df, page_content_column="document_content")
        documents = loader.load()

        # Split into LLM-friendly chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Convert to vector embeddings
        embeddings = OllamaEmbeddings(model="deepseek-llm:7b")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Save vector index by table
        index_folder = os.path.splitext(os.path.basename(csv_path))[0] + "_index"
        vectorstore.save_local(index_folder)

        print(f"✅ Indexed: {csv_path} → {index_folder}")
    except Exception as e:
        print(f"❌ Failed to process {csv_path}: {e}")

if __name__ == "__main__":
    for filename in os.listdir(EXPORT_FOLDER):
        if filename.endswith(".csv"):
            full_path = os.path.join(EXPORT_FOLDER, filename)
            print(f"🔍 Processing: {filename}")
            prepare_rag_data_for_csv(full_path)
