## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df
```
![1](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/4f69a620-77f0-486c-ab1d-e55954228b55)

# OrdinalEncoder
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![2](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/d8e7206d-bd6f-48ce-bd61-8fa4ca8ff7ec)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![3](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/6c8922fc-801b-42fe-8590-0ba94d357967)

# LabelEncoder
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df[["ord_2"]])
dfc
```
![4](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/b1575617-b2ba-4664-8e9b-0af9050016a1)

# OneHotEncoder
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![5](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/c02d3525-457f-4ac4-a63c-aea309283c13)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![6](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/ff5979c7-8702-422b-92e2-cb956bb378b2)

# BinaryEncoder
```
pip install --upgrade category_encoders
```
![7](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/20034cac-babf-494f-84f2-52926830aa43)

```
from category_encoders import BinaryEncoder
df=pd.read_csv('/content/data.csv')
df
```
![8](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/289a8765-f575-493b-9eb8-dbe4e992a117)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
![9](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/38dba045-f997-457b-9827-7efd6ec1e196)

# Target Encoder
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![10](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/5ba2f1eb-9d43-4383-a092-ae23f7b74b2b)

# Data Transformation
```
import pandas as pd
import numpy as np
from scipy import stats
df=pd.read_csv('/content/Data_to_Transform.csv')
df
```
![11](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/a2c5b9f0-42ef-4fae-8fec-d566fc1adb82)

```
df.skew()
```
![12](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/0514be50-f859-4cca-814a-eef99f6b7ca4)

```
np.log(df["Highly Positive Skew"])
```
![13](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/4d36e867-0d23-4a51-9c78-ec7cadced91a)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![14](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/e8efad38-f8e0-4490-9374-80fe839eae5e)

```
np.sqrt(df["Highly Positive Skew"])
```
![15](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/6cadb0cb-fdf6-464b-ad7c-6757cb1692bb)
```
np.square(df["Highly Positive Skew"])
```
![16](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/17f6e1d4-78e9-490c-9f3a-b64ec777a131)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![17](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/85b04f64-6d3c-413b-83f1-1c2145ed56d0)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![18](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/81cc30aa-b23b-4e1f-9c38-1682780d3eab)
```
df.skew()
```
![19](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/6b536695-88b6-4e1e-bd12-6ac8ab696757)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
```
![20](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/ee2ba310-88ba-45d0-9a76-c08f418aad21)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![21](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/c1cfb2b0-5e5e-4c78-8255-b9dbeb6b7c36)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![22](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/656a03d6-efc6-488a-9ac6-4957bfe29566)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![23](https://github.com/Kirthi-Niharika/EXNO-3-DS/assets/114135005/e55c2596-f11b-4c81-a8b5-7306882e5ae4)

# RESULT:
      Finally,perform Feature Encoding and Transformation process is executed successfully.

