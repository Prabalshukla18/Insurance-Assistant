import os
import json
from langchain_core.documents import Document

DATA_DIR = "insurance_rag_data"

def load_documents():
    all_docs = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".json"):
            file_path = os.path.join(DATA_DIR, file)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data["documents"]:
                doc = Document(
                    page_content = f"{item['title']}. {item['content']}",
                    metadata = {
                        "id": item["id"],
                        "category": item["category"],
                        "tags": item["tags"],
                        "importance": item["importance"],
                        "source_file": file
                    }
                )
                all_docs.append(doc)

    return all_docs


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} semantic documents")

    # Show one example
    print("\nSample Document:\n")
    print(docs[0].page_content)
    print(docs[0].metadata)