import shutil
import pandas as pd

shutil.copy("dataset.csv", "dataset_backup.csv")  

df = pd.read_csv("dataset.csv", header=None)
print("Before:", len(df), "rows,", (df.iloc[:, -1] == "V").sum(), "are U")

df = df[df.iloc[:, -1] != "V"]
df.to_csv("dataset.csv", header=False, index=False)

print("After:", len(df), "rows")