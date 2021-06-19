#predict whether the pokemon is a legendary Pokemon (rare - only one of its species)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import csv

df = pd.read_csv('pokemon_data.csv')

#see column headers
#print(df.columns)

#replace dataframe with only relevant columns needed
df = df[['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style']]

#Convert false/true into 0/1
df['isLegendary'] = df['isLegendary'].astype(int)

#create dummy variables to change multiple choices into yes/no data
#creates dummy dataframe of a category - which is concatenated to the original df
#then you drop the original column
def dummy_creation(df, dummy_categories):
    for i in dummy_categories:
        df_dummy = pd.get_dummies(df[i])
        df = pd.concat([df, df_dummy], axis=1)
        df = df.drop(i, axis=1)
    return(df)

#apply data to dummy_creation function
df = dummy_creation(df, ['Egg_Group_1', 'Body_Style', 'Color', 'Type_1', 'Type_2'])

#print(df.columns)


#split data into training and test
#test data is where column chosen (generation) = 1, training data is all other generation numbers
#then drop Generation column from both training and test datasets
def train_test_split(DataFrame, column):
    df_train =  DataFrame.loc[df[column] != 1]
    df_test =  DataFrame.loc[df[column] == 1]

    df_train = df_train.drop(column, axis=1)
    df_test = df_test.drop(column, axis=1)

    return(df_train, df_test)



df_train, df_test = train_test_split(df, 'Generation')

#seperate labels from data
def label_delineator(df_train, df_test, label):
    train_data = df_train.drop(label, axis = 1).values
    train_labels = df_train[label].values
    test_data = df_test.drop(label, axis=1).values
    test_labels = df_test[label].values
    return(train_data, train_labels, test_data, test_labels)

train_data, train_labels, test_data, test_labels = label_delineator(df_train, df_test, 'isLegendary')
#print(train_data, train_labels, test_data, test_labels)


#normalise the data (so everything is on the same scale)
def data_normaliser(train_data, test_data):
    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    return(train_data, test_data)

train_data, test_data = data_normaliser(train_data, test_data)

#create keras model (layer1 = ReLU, layer2 = softmax - log reg)
length = train_data.shape[1]
model = keras.Sequential()
model.add(keras.layers.Dense(500, activation='relu', input_shape=[length,]))
model.add(keras.layers.Dense(2, activation='softmax'))

#compile model 
#optimiser - how model is updated as it gains info
#loss - loss function (measures accuracy as model trains)
#metrics - specifies which info it provides (so we can analyse the model)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#fit data
#epochs = iterations
model.fit(train_data, train_labels, epochs=400)

loss_value, accuracy_value = model.evaluate(test_data, test_labels)
print(f'Our test accuracy was {accuracy_value}')



#test prediction using test data
def predictor(test_data, test_labels, index):
    prediction = model.predict(test_data)
    if np.argmax(prediction[index]) == test_labels[index]:
        type = "not be a Legendary Pokemon"
        if test_labels[index] == 1:
            type = "be a Legendary Pokemon"
        print('This was correctly predicted to', type)

    else:
        print('This was incorrectly predicted by the model.')
        return(prediction)

#userinput to make this more understandable
pokemon_num=(0 - 1)
print("This model is used to predict whether a Pokemon is legendary (only one of its species), or not")
pokemon_name=input("Enter the name of your Pokemon of choice (lowercase): ")
with open('pokemon_data.csv', 'r') as f:
    reader = csv.reader(f)
    for line_num, content in enumerate(reader):
        if content[1] == pokemon_name:
            pokemon_num += line_num

if pokemon_num == 0:
    print("Pokemon not found. Make sure it is written lowercase")
else:
    #print (pokemon_num)
    #choose pokemon to test (mewtwo is a legendary pokemon)
    predictor(test_data, test_labels, pokemon_num)
    
