
# import os
# import pandas as pd
# from langchain_community.document_loaders import DataFrameLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_ollama import OllamaEmbeddings

# EXPORT_FOLDER = "Org_folder"

# def prettify_column_name(col):
#     return col.replace("_", " ").replace("-", " ").replace(".", " ").title()

# def build_document_content(df: pd.DataFrame) -> pd.DataFrame:
#     contents = []

#     for _, row in df.iterrows():
#         parts = []

#         for col in df.columns:
#             val = row.get(col)

#             if pd.isna(val):
#                 continue

#             label = prettify_column_name(col)

#             if "date" in col.lower():
#                 try:
#                     val = pd.to_datetime(val).date()
#                 except Exception:
#                     pass

#             parts.append(f"{label}: {val}")

#         summary = ". ".join(str(p) for p in parts if pd.notna(p)) + "."
#         contents.append(summary)

#     df["document_content"] = contents
#     return df

# def prepare_rag_data_for_csv(csv_path):
#     try:
#         df = pd.read_csv(csv_path)

#         # Remove original_* columns
#         df = df.loc[:, ~df.columns.str.startswith('original_')]

#         # Generate readable summaries
#         df = build_document_content(df)

#         loader = DataFrameLoader(df, page_content_column="document_content")
#         documents = loader.load()

#         # Split into LLM-friendly chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         chunks = text_splitter.split_documents(documents)

#         # Convert to vector embeddings
#         embeddings = OllamaEmbeddings(model="deepseek-llm:7b")
#         vectorstore = FAISS.from_documents(chunks, embeddings)

#         # Save vector index by table
#         index_folder = os.path.splitext(os.path.basename(csv_path))[0] + "_index"
#         vectorstore.save_local(index_folder)

#         print(f"✅ Indexed: {csv_path} → {index_folder}")
#     except Exception as e:
#         print(f"❌ Failed to process {csv_path}: {e}")

# if __name__ == "__main__":
#     for filename in os.listdir(EXPORT_FOLDER):
#         if filename.endswith(".csv"):
#             full_path = os.path.join(EXPORT_FOLDER, filename)
#             print(f"🔍 Processing: {filename}")
#             prepare_rag_data_for_csv(full_path)


import os
import pandas as pd
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from tqdm import tqdm

EXPORT_FOLDER = "Org_folder"
CHUNK_SIZE = 1000

def vectorize_documents(docs, output_folder):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(output_folder)
    print(f"✅ Indexed to: {output_folder}")

def process_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def process_excel(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    return df

def prepare_rag_data(file_path):
    try:
        print(f"🔍 Processing: {file_path}")
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        index_folder = f"{base_name}_index"
        progress_file = f"{base_name}.progress"

        if os.path.exists(index_folder):
            print(f"✅ Skipping (already indexed): {index_folder}")
            return

        ext = os.path.splitext(file_path)[-1].lower()
        df = process_csv(file_path) if ext == '.csv' else process_excel(file_path)

        start = 0
        if os.path.exists(progress_file):
            with open(progress_file) as f:
                start = int(f.read().strip() or 0)

        total = len(df)
        documents = []

        for i in tqdm(range(start, total, CHUNK_SIZE), desc="📄 Chunking"):
            chunk = df.iloc[i:i+CHUNK_SIZE].fillna("").astype(str)
            text = chunk.to_csv(index=False)
            documents.append(Document(page_content=text))

            # Save progress
            with open(progress_file, "w") as f:
                f.write(str(i + CHUNK_SIZE))

        if documents:
            vectorize_documents(documents, index_folder)

        os.remove(progress_file)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

if __name__ == "__main__":
    for file in os.listdir(EXPORT_FOLDER):
        if file.endswith(".csv") or file.endswith(".xlsx"):
            prepare_rag_data(os.path.join(EXPORT_FOLDER, file))
