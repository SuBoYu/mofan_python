import pandas as pd
import numpy as np

# Pandas是基于Numpy构建的，让Numpy为中心的应用变得更加简单

### 1. Series, DataFrame

# Series

s = pd.Series([1, 3, 6, np.nan, 44, 1])

print(s)

# DataFrame

dates =  pd.date_range("20160101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=["a", "b", "c", "d"])
print(df)

print(df["b"])

df1 = pd.DataFrame(np.arange(12).reshape((3, 4)))
print(df1)

df2 = pd.DataFrame({"A" : 1.,
                    "B" : pd.Timestamp("20130102"),
                    "C" : pd.Series(1, index=list(range(4)), dtype="float32"),
                    "D" : np.array([3] * 4, dtype="int32"),
                    "E" : pd.Categorical(["test", "train", "test", "train"]),
                    "F" : "foo"})

print(df2)
print(df2.dtypes)
print(df2.index)
print(df2.columns)
print(df2.values) # pandas -> numpy array
df2.describe() # 數據總結
print(df2.T)
print(df2.sort_index(axis=1, ascending=False))
print(df2.sort_values(by="B"))

### 2. Pandas選擇數據

dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.arange(24).rshape((6, 4)), index=dates, columns=["A", "B", "C", "D"])

print(df["A"])
print(df[0:3])
print(df["20130102":"20130104"])

# loc
print(df.loc["20130102"])
print(df.loc[:, ["A", "B"]])
print(df.loc["20130102", ["A", "B"]])

# iloc
print(df.iloc[3, 1])
print(df.iloc[3:5, 1:3])
print(df.iloc[[1, 3, 5], 1:3])

# 判斷篩選
print(df[df["A"] > 8])

### 3.pandas設置值

dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=["A", "B", "C", "D"])

# 根據位置設置
df.iloc[2, 2] = 1111
df.loc["20130101", "B"] = 2222

# 根據條件設置
df["B"][df["A"]>4] = 0

# 根據行或列設置
df["F"] = np.nan

# 添加數據
df["E"] = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130101", periods=6))

### 4.pandas處理丟失數據

# 創建含 Nan 的矩陣
dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=["A", "B", "C", "D"])
df.iloc[0, 1] = np.nan
df.iloc[1, 2] = np.nan

# pd.dropna()
df.dropna(
    axis=0,  # 0: 对行进行操作; 1: 对列进行操作
    how="any"  # 'any': 只要存在 NaN 就 drop 掉; 'all': 必须全部是 NaN 才 drop
    )

# pd.fillna()
df.fillna(value=0)

# pd.isnull()
df.isnull()
# 檢測數據中是否存在nan
np.any(df.isnull()) == True

### 5.pandas導入導出

# pandas可以读取与存取的资料格式有很多种，像csv、excel、json、html与pickle等…

data = pd.read_csv("student-mat.csv", sep=";")
print(data)

# 將資料存成pickle

data.to_pickle("student.pickle")

### 6.pandas合併 concat

df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=["a", "b", "c", 'd'])
df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=["a", "b", "c", 'd'])
df3 = pd.DataFrame(np.ones((3, 4)) * 2, columns=["a", "b", "c", 'd'])

# concat縱向合併
res = pd.concat([df1, df2, df3], axis=0)
print(res)

# index 重置
res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
print(res)

# join合併方式
df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=["a", "b", "c", 'd'], index=[1, 2, 3])
df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=["b", "c", "d", 'e'], index=[2, 3, 4])

# 縱向“外”合併df1和df2
res = pd.concat([df1, df2], axis=0, join="outer")
print(res)

# 縱向“內”合併df1和df2
res = pd.concat([df1, df2], axis=0, join="inner")
print(res)

# 重置index並打印結果
res = pd.concat([df1, df2], axis=0, join="inner", ignore_index=True)
print(res)

# append(添加數據) ps. append只有縱向合併, 沒有橫向合併
df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=["a", "b", "c", 'd'])
df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=["a", "b", "c", 'd'])
df3 = pd.DataFrame(np.ones((3, 4)) * 1, columns=["a", "b", "c", 'd'])
s1 = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])

# 将df2合并到df1的下面，以及重置index，并打印出结果
res = df1.append(df2, ignore_index=True)
print(res)

# 合并多个df，将df2与df3合并至df1的下面，以及重置index，并打印出结果
res = df1.append([df2, df3], ignore_index=True)
print(res)

# 合并series，将s1合并至df1，以及重置index，并打印出结果
res = df1.append(s1, ignore_index=True)
print(res)

### 7.pandas合併merge

# pandas中的merge和concat类似,但主要是用于两组有key column的数据,统一索引的数据. 通常也被用在Database的处理当中.

# 依據一組key合併
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                             'A': ['A0', 'A1', 'A2', 'A3'],
                             'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                              'C': ['C0', 'C1', 'C2', 'C3'],
                              'D': ['D0', 'D1', 'D2', 'D3']})

print(left)
print(right)

res = pd.merge(left, right, on="key")
print(res)

# 依據2組key合併
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                      'key2': ['K0', 'K1', 'K0', 'K1'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                       'key2': ['K0', 'K0', 'K0', 'K0'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']})
print(left)
print(right)

res = pd.merge(left, right, on=["key1", "key2"], how="inner")
print(res)

res = pd.merge(left, right, on=["key1", "key2"], how="outer")
print(res)

res = pd.merge(left, right, on=["key1", "key2"], how="left")
print(res)

res = pd.merge(left, right, on=["key1", "key2"], how="right")
print(res)

# indicator
df1 = pd.DataFrame({"col1":[0, 1], "col_left":["a", "b"]})
df2 = pd.DataFrame({"col1":[1, 2, 2], "col_right":[2, 2, 2]})
print(df1)
print(df2)

# 依据col1进行合并，并启用indicator=True，最后打印出
res = pd.merge(df1, df2, on="col1", how="outer", indicator=True)
print(res)

# 自定indicator column的名称，并打印出
res = pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')
print(res)

# 依據index合併
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                     index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])
print(left)
print(right)

# 依据左右资料集的index进行合并，how='outer',并打印出
res = pd.merge(left, right, left_index=True, right_index=True, how="outer")
print(res)

# 依据左右资料集的index进行合并，how='inner',并打印出
res = pd.merge(left, right, left_index=True, right_index=True, how="inner")
print(res)

# 解決overlapping問題
boys = pd.DataFrame({"k": ["K0", "K1", "k2"], "age": [1, 2, 3]})
girls = pd.DataFrame({"k": ["K0", "K0", "k3"], "age": [4, 5, 6]})

# 使用suffixes解決overlapping問題
res = pd.merge(boys, girls, on="k", suffixes=["_boy", "_girl"], how="inner")
print(res)

### 8.pandas plot

import matplotlib.pyplot as plt

# Series
data = pd.Series(np.random.randn(1000), index=np.arange(1000))
data.cumsum()
data.plot()
plt.show()

# DataFrame
data = pd.DataFrame(
    np.random.randn(1000, 4),
    index=np.arange(1000),
    columns=list("ABCD")
    )
data.cumsum()
data.plot()
plt.show()

# scatter
ax = data.plot.scatter(x="A", y="B", color="DarkBlue", label="Class1")
data.plot.scatter(x="A", y="C", color="LightGreen", label="Class2", ax=ax)
plt.show()






