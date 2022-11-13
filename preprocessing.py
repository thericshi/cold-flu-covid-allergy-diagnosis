import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("symptom_data.csv")

print(train["TYPE"].value_counts())

train = train.drop(range(0, 15357))
print(train)
train = train.drop(range(18429, 43429))

print(train["TYPE"].value_counts())
