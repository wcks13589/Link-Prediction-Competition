import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

class Data():
    def __init__(self, dataset):
        
        self.data_path = f'./{dataset}'
        self.train_file = 'train.csv'
        self.test_file = 'test.csv'
        self.feature_file = 'content.csv'
        
        self.load_edges()
        
        self.num_features = self.train_x.size(1)
        
    def split_edges(self, test_size=0.1):
        '''
        split edges into training edges and validation edges by the test_size
        '''
        
        train_pos_edge_index, val_pos_edge_index = map(lambda x: x.T, 
                                                       train_test_split(self.train_pos_edges, test_size=test_size))
        train_neg_edge_index, val_neg_edge_index = map(lambda x: x.T, 
                                                       train_test_split(self.train_neg_edges, test_size=test_size))
        
        return train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index
        
        
    def load_edges(self):
        
        train_data = pd.read_csv(os.path.join(self.data_path, self.train_file)).sort_values('label', ascending=False)
        test_data = pd.read_csv(os.path.join(self.data_path, self.test_file))
        train_x = pd.read_csv(os.path.join(self.data_path, self.feature_file), delimiter='\t', header=None).sort_values(0)
        
        train_edges = torch.LongTensor(train_data[['from', 'to']].to_numpy())
        self.train_x = torch.Tensor(train_x.to_numpy()[:,1:]) # first column is node index
        
        self.test_edges = torch.LongTensor(test_data[['from', 'to']].to_numpy()).T

        self.train_pos_edges = train_edges[train_data.label==1]
        self.train_neg_edges = train_edges[train_data.label==0]
