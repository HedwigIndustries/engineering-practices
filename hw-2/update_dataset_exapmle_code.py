import pandas as pd

df = pd.read_csv("dataset.csv", encoding="windows-1251", sep=";")
print(df)
df.drop(
    [
        "TDP:(W)",
        "Technical process:(nm)",
        "L1 cache:(byte)",
        "L2 cache:(byte)",
        "L3 cache:(byte)",
        "Type of integrated GPU",
        "PCIe(version)",
        "PCIe(lines)",
    ],
    axis=1,
    inplace=True,
)
print(df)
# df.to_csv('dataset.csv', index=False)
df.to_csv("update_dataset.csv", index=False)
