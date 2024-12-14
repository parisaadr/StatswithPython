import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

heart = pd.read_csv('processed.cleveland.data.csv')
# print(heart.head())
# print(heart.info())
# print(heart.ca.unique())
# print(heart.thal.unique())

heart = heart.replace("?", np.nan)

# print(heart.ca.unique())
# print(heart.thal.unique())

heart.ca = heart.ca.astype("float")
# print(heart.info())

# - cp: chest pain type
#  - Value 1: typical angina
#  - Value 2: atypical angina
#  - Value 3: non-anginal pain
#  - Value 4: asymptomatic
heart.cp = heart.cp.replace({1.0: "typical anagina",
                  2.0: "atypical angina",
                  3.0: "non-anginal pain",
                  4.0: "asymptomatic"})

# print(heart.info())
# print(heart.describe(include= "all"))

# slope: the slope of the peak exercise ST segment
# - Value 1: upsloping
# - Value 2: flat
# - Value 3: downsloping
heart.slope = heart.slope.replace({
    1.0: "upsloping",
    2.0: "flat",
    3.0: "downsloping"})
heart.slope = pd.Categorical(heart.slope, ["upsloping", "flat", "downsloping"], ordered=True)

# print(heart.slope.cat.codes)
# print(heart.describe(include="all"))
# print(heart.slope.cat.codes.median())
# thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

heart.thal = heart.thal.replace({"3.0": "normal", "6.0": "fixed defect", "7.0": "reversable defect"})

heart.heart_disease = np.where(heart.heart_disease == 0, "absence", "presence")
# print(heart.heart_disease.head())

print(heart.describe(include= "all"))
