def search_vectordb(query,knowledgeBase,top_k):
    top_k=int(top_k)
    docs = knowledgeBase.similarity_search(query,top_k)
    print("length of search results")
    print(len(docs))
    return docs

