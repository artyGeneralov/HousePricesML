import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from helperFuncs import *

mainDataPath = "dbs/house_prices_canada.csv"
pd.set_option("display.max_rows", 3)
pd.set_option("display.max_columns", 60)
mainData = None
dummifiedData = None


def main():
    print("start")
    global mainData, dummifiedData

    mainData = pd.read_csv(mainDataPath, encoding="ISO-8859-1")

    columns_to_dummify = ["Province"]
    dummifiedData = createDummies(mainData, columns_to_dummify)
    dummifiedData = dropObjectColumns(dummifiedData) # if we decide not to dummify something
    dummifiedData.dropna(inplace=True)
    # add new column to data:

    dummifiedData = addNewColumn(dummifiedData, "beds:baths", "dummifiedData[\"Number_Beds\"] / dummifiedData[\"Number_Baths\"]")

    # filtered_rows = dummifiedData[dummifiedData['Price'] <= 0]
    # # Displaying the filtered rows
    # print("FILTERED:::\n\n", filtered_rows)


    #decide which values to log
    values_to_log = ["Price", "Number_Beds", "Number_Baths", "Population", "Median_Family_Income"]
    dummifiedData = logFields(dummifiedData, values_to_log)

    #Data Scaling - Standartization:
    scaler = StandardScaler()
    scaledData = scaler.fit_transform(dummifiedData)
    dummifiedData = pd.DataFrame(scaledData, columns=dummifiedData.columns)
    dummifiedData.dropna(inplace=True)

## At this point dummified data is Scaled => loged

    # create the test and train pair:
    x_train, x_test, y_train, y_test, trainingData, testData = createTestTrainPair(dummifiedData, "Price")

    # train linear regression model
    reg = LinearRegression()
    reg.fit(x_train,y_train)
    print("Linear Regression Score: ", reg.score(x_test, y_test))

    # train forest regression model
    forest = RandomForestRegressor()
    forest.fit(x_train, y_train)
    print("Forest Regression Score: ", forest.score(x_test, y_test))   

    # test with test data
    for _ in range(10):
        input("Press Enter for next house\n\n")
        testnum = random.randint(1, 500)  # Change the range as needed

        fullTestRow = pd.concat([getRow(y_test, testnum), getRow(x_test, testnum)], axis=1)
        test_row_x = fullTestRow.drop(["Price"], axis = 1)
        test_row_y = fullTestRow["Price"]


        unscaledTestRow = scaler.inverse_transform(fullTestRow)
        unscaledTestRow = pd.DataFrame(unscaledTestRow, columns=fullTestRow.columns)


        expedTestRow = expFields(unscaledTestRow, values_to_log)


        #Linear prediction
        prediction = reg.predict(test_row_x)
        prediction_df = pd.DataFrame({"Price":prediction}, index=test_row_x.index)
        rowWithPrediction = pd.concat([prediction_df, test_row_x], axis = 1)
        rowWithPrediction_unscaled = scaler.inverse_transform(rowWithPrediction)
        rowWithPrediction_unscaled = pd.DataFrame(rowWithPrediction_unscaled, columns=fullTestRow.columns)
        rowWithPrediction_exped = expFields(rowWithPrediction_unscaled, values_to_log)


        
        print("\nLR Model prediction Price:\n", int(rowWithPrediction_exped["Price"].iloc[0]))
        #Forest prediction
        prediction = forest.predict(test_row_x)
        prediction_df = pd.DataFrame({"Price":prediction}, index=test_row_x.index)
        rowWithPrediction = pd.concat([prediction_df, test_row_x], axis = 1)
        rowWithPrediction_unscaled = scaler.inverse_transform(rowWithPrediction)
        rowWithPrediction_unscaled = pd.DataFrame(rowWithPrediction_unscaled, columns=fullTestRow.columns)
        rowWithPrediction_exped = expFields(rowWithPrediction_unscaled, values_to_log)

        print("\nForest Model prediction Price:\n", int(rowWithPrediction_exped["Price"].iloc[0]))
        print("\nActual Price:\n", int(expedTestRow["Price"].iloc[0]))
        
        # print("**\nActual Price: ", np.exp(y_test.iloc[testnum] - 1))
        # print("TestRow:: ", testRow)
        # scaledPrediction = np.exp(reg.predict(testRow) - 1)
        # testRow["Price"] = scaledPrediction
        # print("Scaledf:",testRow)
        # print("Prediction: ", scaler.inverse_transform(testRow))
    

if __name__ == "__main__":
    main()