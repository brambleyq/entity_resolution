import pandas as pd
import numpy as np

class Tmdb:
    def __init__(self):
        self.df:pd.DataFrame = None
        self.popular:pd.DataFrame = None
        self.train = None
        self.test = None
        self.columns = None
        self.safe = None
        self.adult = None

    def read(self, path:str):
        self.df = pd.read_csv(path)
        self.safe = self.df.loc[ ~ self.df['adult']]
        self.adult = self.df.loc[self.df['adult']]
        self.columns = self.df.columns
        self.popular = self.df.loc[self.df['popularity'] > 1]

    def training(self,test_frac: float = 0.1):
        """ Return ONLY the training data.
        Idendify the training data if it does not yet exist
        """
        if self.train is None:
            self._train_test_split(test_frac)    
        return self.train
    
    def testing(self,test_frac: float = 0.1):
        """ Return ONLY the training data.
        Idendify the training data if it does not yet exist
        """
        if self.test is None:
            self._train_test_split(test_frac)
        return self.test

    # if it starts with an underscore you are not allowed
    # to reference it from another file
    def _train_test_split(self, test_frac: float = 0.1):
        all_rows = np.arange(len(self.df))
        np.random.shuffle(all_rows)
        test_n_rows = round(len(self.df)*test_frac)
        test_rows = all_rows[:test_n_rows]
        train_rows = all_rows[test_n_rows:]
        
        self.test = self.df.loc[test_rows].reset_index(drop=True)
        self.train = self.df.loc[train_rows].reset_index(drop=True)