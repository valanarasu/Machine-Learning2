import pandas
from sklearn import linear_model

df = pandas.read_csv("D:\ML\Final-Regression\Machine-Learning2\data-multiple.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)


#predict the CO2 emission of a car where the weight is 2300g, and the volume is 1300ccm:
predictedCO2 = regr.predict([[2300, 1300]])



# The result array represents the coefficient values of weight and volume.

# Weight: 0.00755095
# Volume: 0.00780526

# These values tell us that if the weight increase by 1kg, the CO2 emission increases by 0.00755095g.

# And if the engine size (Volume) increases by 1 cm3, the CO2 emission increases by 0.00780526 g.
print(regr.coef_)

print("We have predicted that a car with 1.3 liter engine, and a weight of 2300 kg, will release approximately 107 grams of CO2 for every kilometer it drives.")
print(predictedCO2)