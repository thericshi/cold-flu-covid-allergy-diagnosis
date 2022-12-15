import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("symptom_data.csv")

df = df.drop(range(0, 15357))
df = df.drop(range(18429, 43429))

df.to_csv('under-sampled.csv', index=False)


