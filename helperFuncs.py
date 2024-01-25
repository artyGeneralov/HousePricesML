import random
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plot 
import seaborn as sb
from sklearn.model_selection import train_test_split

def createTestTrainPair(data, testElem):
    dt = data.copy()
    x = data.drop(testElem, axis = 1)
    y = data[testElem]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
    trainingData = x_train.join(y_train)
    testData = x_test.join(y_test)
    return (x_train, x_test, y_train, y_test, trainingData, testData)

def addNewColumn(dataframe, newColumn, operation):
    dt = dataframe.copy()
    try:
        dt[newColumn] = eval(operation)
        dt[newColumn] = np.where(np.isfinite(dt[newColumn]), dt[newColumn], np.nan)
    except Exception as e:
        print(f"Error{e}")
    return dt

def showGraphs(data, x = None, y= None, hue = None):
    showHist(data)
    showHeat(data)
    if x is not None and y is not None and hue is not None:
        showScatter(data, x, y, hue)



def showHist(data):
    data.hist(figsize = (20,13))
    plot.show()

def showHeat(data):
    corrMat = data.corr()
    plot.figure(figsize=(150, 135))
    sb.heatmap(corrMat, annot=True, cmap = 'coolwarm', annot_kws={"size": 6})
    plot.show()


def showScatter(data, x, y, hue):
    sb.scatterplot(x = x, y = y, data = data, hue = hue)
    plot.show()

def dropObjectColumns(data):
    object_cols = data.select_dtypes(include=['object']).columns
    if len(object_cols) == 0:
        return data
    dt = data.drop(columns=object_cols)
    return dt

def createDummies(dataframe, columns):
    temp_dfs = [dataframe]
    for column in columns:
        dummies = pd.get_dummies(dataframe[column]).astype(int)
        temp_dfs.append(dummies)
    
    dataframe_with_dummies = pd.concat(temp_dfs, axis = 1)
    dataframe_with_dummies = dataframe_with_dummies.drop(columns=columns)
    return dataframe_with_dummies


def getAttributesList(data):
    return data.columns.tolist()

def getRow(data, row):
    if(row > len(data)):
        return
    theRow = data.iloc[row:row+1]
    return theRow

def logFields(data, fields):
    dt = data.copy()
    for field in fields:
        if field not in dt.columns:
            print("Field \"",field,"\" Not in table: ", dt.columns)
            continue
        try:
            dt[field] = np.log(dt[field] + 1)
        except Exception as e:
            print(f"Error: {e} for field {field} value {dt[field]}")
    return dt

def expFields(data, fields):
    dt = data.copy()
    for field in fields:
        if field not in dt.columns:
            print("Field \"",field,"\" Not in table: ", dt.columns)
            continue
        dt[field] = np.exp(dt[field]) - 1
    return dt
