import pandas as pd
col_list = ['Name','Age','Club']
df = pd.read_csv("original_clubagedata.csv", usecols=col_list)
df.dropna(inplace=True)
df.to_csv('clubagedata.csv', sep=',',header=None, index=False)
