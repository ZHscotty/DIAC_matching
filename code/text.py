import pandas as pd
import numpy as np

a = [1,2,3]
b = [3,4,5]
l = [a, b]
l = np.array(l)
l = np.transpose(l)
print(l)
print(l.shape)
p = pd.DataFrame(l)
p.to_csv('../data/text.csv', index=False, header=False)