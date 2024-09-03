import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pandas.read_csv("D:\ML\Final-Regression\Machine-Learning2\data-multiple.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)

regr = linear_model.LogisticRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

# predictedCO2 = regr.predict([scaled[0]])
print(regr.predict([scaled[0]]))