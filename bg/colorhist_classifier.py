import pandas as pd
import numpy as np

csv_data = pd.read_csv('train_data.csv').values
np.random.shuffle(csv_data)
genres = np.array(csv_data[:,-1]).astype(int)

test_data_csv = pd.read_csv('test_data.csv').values

