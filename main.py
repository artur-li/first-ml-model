import numpy as np
from sklearn.linear_model import LinearRegression

# data to train model on
x = np.array([[i] for i in range(1,11)])
y = np.array([[i*3+2] for i in range(1,11)])

# create and train model
model = LinearRegression()
model.fit(x, y)

# use model to predict y based on these x inputs
prediction = model.predict([[0], [1], [12], [100]])
# print the results in a more appealing fashion
flattened_results = prediction.flatten()
print([int(value) for value in flattened_results])
