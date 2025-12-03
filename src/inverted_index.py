from collections import defaultdict
from tokenizer import Tokenizer

class InvertedIndex:
    def __init__(self):
        # index["word"] = {doc_id: count}
        self.index = defaultdict(dict)
        self.document_count = 0
        self.doc_freq = defaultdict(int) 
        self.documents = defaultdict(str)

    def build(self, documents):
        self.document_count = len(documents)
        self.documents = documents
        tokenizer = Tokenizer()
        for doc_id, doc in enumerate(documents):
            tokens = tokenizer.preprocess(doc)

            # Count frequency of each term in this document
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            # Add terms to the index
            for term, freq in term_freq.items():
                self.index[term][doc_id] = freq
                self.doc_freq[term] += 1

    def get_postings(self, term):
        """
        Returns posting list for a term
        Example: {"oil": {0:3, 2:1}}
        """
        return self.index.get(term, {})

    def get_index(self):
        return dict(self.index)

    def get_doc_freq(self, term):
        return self.doc_freq.get(term, 0)
    
    def get_all_doc_freqs(self):
        return dict(self.doc_freq)

