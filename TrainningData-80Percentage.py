

import numpy
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100)/x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

myline = numpy.linspace(0, 6, 100)

# Display the Training Set
# Display the same scatter plot with the training set:
plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()

# Note: The result 0.799 shows that there is a OK relationship.
print(r2_score(train_y, mymodel(train_x)))

# Display the Testing Set
# To make sure the testing set is not completely different, we will take a look at the testing set as well.

plt.scatter(test_x, test_y)
plt.plot(myline, mymodel(myline))
plt.show()

#Note: The result 0.809 shows that the model fits the testing set as well, and we are confident that we can use the model to predict future values.

print(r2_score(test_y, mymodel(test_x)))

# Now that we have established that our model is OK, we can start predicting new values.
print(mymodel(5))