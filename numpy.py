import numpy as np

### 1. numpy屬性
# numpy 3種屬性
array = np.array([[1, 2, 3], [2, 3, 4]])
print("number of dim:", array.ndim)
print("shape:", array.shape)
print("size:", array.size)

### 2. numpy創建
# array：创建数组
a = np.array([2, 23, 4])
print(a)

# dtype：指定数据类型
a = np.array([2, 23, 4], dtype=np.int)
print(a.dtype)

# int64
a = np.array([2, 23, 4], dtype=np.int32)
print(a.dtype)

a = np.array([2, 23, 4], dtype=np.float)
print(a.dtype)

a = np.array([2, 23, 4], dtype=np.float32)
print(a.dtype)

# zeros：创建数据全为0
a = np.zeros((3, 4))
print(a)

# ones：创建数据全为1
a = np.ones((3, 4), dtype=np.int)

# empty：创建数据接近0
a = np.empty((3, 4))

# arange：按指定范围创建数据
a = np.arange(10, 20, 2)

# reshape: 改變數據形狀
a = np.arange(12).reshape((3, 4))

# linspace：创建线段
a = np.linspace(1, 10, 20)

a = np.linspace(1, 10, 20).reshape((5, 4))


### 3.基礎運算1

a = np.array([10, 20, 30, 40])
b = np.arange(4)

# +, -, *, **
c = a - b

# sin
c = 10*np.sin(a)

print(b < 3)

# dim 2
a = np.array([[1, 1], [0, 1]])
b = np.arange(4).reshape((2, 2))

# 內積
c_dot = np.dot(a, b)

# sum, min, max
a = np.random.random((2, 4))  # 0~1隨機數
print(a)

np.sum(a)
np.min(a)
np.max(a)

print("a = ", a)
print("sum = ", np.sum(a, axis=1))
print("min = ", np.min(a, axis=0))
print("max = ", np.max(a, axis=1))

### 4. 基礎運算2

A = np.arange(2, 14).reshape((3, 4))

# 針對index
print(np.argmin(A))
print(np.argmax(A))
print(np.mean(A))
print(np.average(A))
print(np.median(A))
print(np.cumsum(A))
print(np.diff(A))
print(np.nonzero(A))

A = np.arange(14, 2, -1).reshape((3, 4))
print(np.sort(A))
print(np.sort(A, axis=0))

print(np.transpose(A))

print(np.clip(A, 5, 9))


### 5. 索引

A = np.arange(3, 15)
print(A[3])

A = np.arange(3, 15).reshape((3, 4))
print(A[2])

print(A[1][1])
print(A[1, 1:3])

for row in A:
    print(row)

for column in A.T:
    print(column)

A = np.arange(3, 15).reshape((3, 4))
print(A.flatten())

for item in A.flat:
    print(item)

### 6. numpy array合併

# np.vstack(): vertical stack

A = np.array([1, 1, 1])
B = np.array([2, 2, 2])

C = np.vstack((A, B))
print(C)
print(A.shape, C.shape)

# np.hstack(): horizontal stack

D = np.hstack((A, B))
print(D)
print(A.shape, D.shape)

# np.newaxis()

print(A[np.newaxis, :])  # 1*3
print(A[:, np.newaxis])  # 3*1
print(A[:, np.newaxis].shape)

A = np.array([1, 1, 1])[:, np.newaxis]
B = np.array([2, 2, 2])[:, np.newaxis]

C = np.vstack((A, B))
D = np.hstack((A, B))

# np.concatenate()

C = np.concatenate((A, B, B, A), axis=0)
print(C)
D = np.concatenate((A, B, B, A), axis=1)
print(D)

### 7. numpy array分割

A = np.arange(12).reshape((3, 4))
print(A)

# 縱向分割
print(np.split(A, 2, axis=1))

# 橫向分割
print(np.split(A, 3, axis=0))

# 不等量的分割
print(np.array_split(A, 3, axis=1))

# 其他分割
print(np.vsplit(A, 3))

print(np.hsplit(A, 2))

### 8. numpy copy & deep copy

a = np.arange(4)

b = a
c = a
d = b

a[0] = 11
print(a)
d[1:3] = [22, 33]

b = a.copy()
print(b)
a[3] = 44
print(a)
print(b)
