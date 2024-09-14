import pandas as pd

csv_path = "D:/Users/bob/project/ai/covid-19-demo/covid-19-dataset-3/dataset-meta-data.csv"

df = pd.read_csv(csv_path)
print(df)
var = (df[df["set-type"] != "train"])
print(var)

