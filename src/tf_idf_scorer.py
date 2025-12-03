import math
from collections import defaultdict
from inverted_index import InvertedIndex

class TFIDFScorer:
    def __init__(self, inv_index: InvertedIndex):
        self.inv_index = inv_index
        self.N = inv_index.document_count
        self.doc_freq = inv_index.doc_freq
        self.documents = inv_index.documents

    def _compute_idf(self, term):
        df = self.inv_index.get_doc_freq(term)
        return math.log((self.N + 1) / (df + 1)) + 1
    
    def _compute_tf(self, term, doc_id):
        return self.inv_index.get_postings(term).get(doc_id, 0) # count of term in doc
    
    def score_query(self, query):
        query_tokens = query.split()
        query_vec = defaultdict(float)
        for term in query_tokens:
            tf = query_tokens.count(term)
            idf = self._compute_idf(term)
            query_vec[term] = tf * idf

        # Normalize query vector
        query_norm = math.sqrt(sum(v**2 for v in query_vec.values()))

        # Score all documents
        scores = []
        for doc_id, doc in enumerate(self.documents):
            doc_vec = defaultdict(float)
            for term in query_tokens:
                tf = self.inv_index.get_postings(term).get(doc_id, 0)
                idf = self._compute_idf(term)
                doc_vec[term] = tf * idf
            doc_norm = math.sqrt(sum(v**2 for v in doc_vec.values()))
            if doc_norm == 0 or query_norm == 0:
                score = 0.0
            else: # cosine similarity
                dot = sum(doc_vec[t] * query_vec[t] for t in query_tokens)
                score = dot / (doc_norm * query_norm)
            scores.append((doc_id, score))
        # print(f"All document scores: {scores}")
        return sorted(scores, key=lambda x: -x[1])