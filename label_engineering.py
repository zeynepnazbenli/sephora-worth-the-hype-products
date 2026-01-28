import pandas as pd

df = pd.read_csv("data/product_info.csv")
df = df.dropna(subset= ["rating","reviews","loves_count"])

rating_high = df["rating"].quantile(0.75)
loves_high = df["loves_count"].quantile(0.75)
loves_low = df["loves_count"].quantile(0.25)

def assign_label(row):
    if row["rating"] >= rating_high and row["loves_count"] >= loves_high:
        return "worth_it"
    elif row["rating"] < rating_high and row["loves_count"] >= loves_high:
        return "overrated"
    elif row["rating"] >= rating_high and row["loves_count"] <= loves_low:
        return "underrated"
    else:
        return "neutral"
    
df["hype_label"] = df.apply(assign_label, axis=1)
df = df[df["hype_label"] != "neutral"]
print("Class distribution:")
print(df["hype_label"].value_counts())
df.to_csv("data/labeled_products.csv", index=False)
print("\nSaved: data/labeled_products.csv")