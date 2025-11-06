import pandas as pd
import numpy as np

df = pd.read_csv(
    "../tech-as-topology/tech-as-topology.edges",
    sep=" ", header=None, comment="%",
    names=["node1", "node2", "weight", "timestamp", "value"]
)

df.drop(columns=["weight", "timestamp"], inplace=True)

df["value"] = (np.random.randint(1, 6, size=len(df)))/10

print(df["value"])

df.to_csv("../tech-as-topology/tech-as-topology.edges", sep=" ", index=False, header=False)
