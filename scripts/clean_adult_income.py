import pandas as pd

# Load raw Adult Income CSV
df = pd.read_csv("data/data.csv")

# Normalize label column (handle both formats)
df["income"] = (
    df["income"]
    .astype(str)
    .str.strip()
    .replace({
        "<=50K": 0,
        "<=50K.": 0,
        ">50K": 1,
        ">50K.": 1
    })
    .astype(int)
)

# Optional: drop rows with missing labels
df = df.dropna(subset=["income"])

# Save cleaned file
df.to_csv("data/data.csv", index=False)

print("Saved cleaned dataset to data/data.csv")
