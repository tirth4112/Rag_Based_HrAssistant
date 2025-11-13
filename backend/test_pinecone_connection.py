from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_2ntSPi_2sr3xdfSYgJgi2hEP4MWanzT1pN29p9DSPzWPCLpKuSMzNf8jLnpFdHE591tXza")
print(pc.list_indexes())


from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_2ntSPi_2sr3xdfSYgJgi2hEP4MWanzT1pN29p9DSPzWPCLpKuSMzNf8jLnpFdHE591tXza")
index = pc.Index("hr-policies")

stats = index.describe_index_stats()
print(stats)


# import requests
# from pinecone import Pinecone

# pc = Pinecone(api_key="pcsk_2ntSPi_2sr3xdfSYgJgi2hEP4MWanzT1pN29p9DSPzWPCLpKuSMzNf8jLnpFdHE591tXza")
# index = pc.Index("hr-policies")

# # Suppose you have some HR policy texts
# documents = [
#     {"id": "1", "text": "Employees are entitled to 18 days of annual leave."},
#     {"id": "2", "text": "Office timings are from 9 AM to 6 PM, Monday to Friday."},
# ]

# for doc in documents:
#     emb_res = requests.post(
#         "http://localhost:11434/api/embeddings",
#         json={"model": "nomic-embed-text", "prompt": doc["text"]}
#     )
#     emb = emb_res.json()["embedding"]

#     index.upsert(vectors=[{
#         "id": doc["id"],
#         "values": emb,
#         "metadata": {"text": doc["text"]}
#     }])
