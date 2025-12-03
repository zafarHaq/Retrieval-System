from csv_reader import CSVReader
from inverted_index import InvertedIndex
from tf_idf_scorer import TFIDFScorer

reader = CSVReader("../documents/Articles.csv", "Article")
reader.read_csv()
docs = reader.get_documents()

inv = InvertedIndex()
inv.build(docs)

scorer = TFIDFScorer(inv)

while True:
    query = input("\nEnter query: ").strip()
    if query.lower() == "exit":
        print("Exiting...")
        break
    if not query:
        print("Empty query, try again.")
        continue
        
    print("\nGenerating results please wait...")
    results = scorer.score_query(query)
        
    if not results:
        print("\nNo documents found.")
        continue

    print("\nRanked documents:")
    for doc_id, score in results[:10]:  # show top 10 results
        print(f"\nDocument {doc_id} | Score: {score:.4f} | Content: {scorer.documents[doc_id][:100]}...")

 