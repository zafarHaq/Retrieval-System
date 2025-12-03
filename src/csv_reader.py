import pandas as pd

class CSVReader:
    def __init__(self, file_path, text_column="Article"):
        self.file_path = file_path
        self.text_column = text_column
        self.documents = []

    def read_csv(self):
        try:
            df = pd.read_csv(self.file_path, encoding='latin1', encoding_errors='ignore')
            if self.text_column not in df.columns:
                raise ValueError(f"Column '{self.text_column}' not found in CSV")
            
            self.documents = df[self.text_column].fillna("").astype(str).tolist()
            print(f"Loaded {len(self.documents)} documents from {self.file_path}")
        
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
        except Exception as e:
            print(f"Error reading CSV: {e}")

    def get_documents(self):
        return self.documents

