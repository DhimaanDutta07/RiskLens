import pandas as pd

class DataIngestor():
    def __init__(self,path):
        self.path="data/raw/transactions.csv"

    def ingest(self):
        df=pd.read_csv(self.path)
        return df