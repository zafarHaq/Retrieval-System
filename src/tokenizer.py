import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # for proper lemmatization

class Tokenizer:
    """
    A class to preprocess text:
    - Lowercasing
    - Remove punctuation
    - Remove extra spaces
    - Remove stopwords
    - Lemmatization
    """

    def __init__(self, language="english"):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation (keep only letters and numbers)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return tokens
