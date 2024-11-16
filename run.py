# Solution code for the Iris Dataset Homework (run.py)

import pandas as pd
from scipy.stats import zscore

# Question 1: Pre-process the data
def preprocess_data(input_filename):
    data = pd.read_csv(input_filename)
    data.columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]

    data["length_z"] = abs(zscore(data["SepalLengthCm"]))
    data["width_z"] = abs(zscore(data["SepalWidthCm"]))

    data = data[(data["length_z"] <= 2) & (data["width_z"] <= 2)]
    data = data.drop(columns=["length_z", "width_z"])
    data["ID"] = range(1, len(data)+1)

    return data


# Question 2: Descriptive Statistics Functions
def species_count(data):
    data_obj = preprocess_data(data)
    value_counts = data_obj["Species"].value_counts()
    return value_counts.to_dict()

def average_sepal_length(data):
    data_obj = preprocess_data(data)
    return round(data_obj["SepalLengthCm"].mean(), 1)

def max_petal_width(data):
    data_obj = preprocess_data(data)
    return round(data_obj["PetalWidthCm"].max(), 1)

def min_petal_length(data):
    data_obj = preprocess_data(data)
    return round(data_obj["PetalLengthCm"].min(), 1)

def count_sepal_length_above_5(data):
    data_obj = preprocess_data(data)
    return len(data_obj[data_obj["SepalLengthCm"] > 5])

# Question 3: Analysis Functions
def count_petal_length_below_2(data):
    data_obj = preprocess_data(data)
    return len(data_obj[data_obj["PetalLengthCm"] < 2])

def get_sepal_width_above_3_5(data):
    data_obj = preprocess_data(data)
    return list(data_obj[data_obj["SepalWidthCm"] > 3.5]["ID"])

def species_count_petal_width_above_1_5(data):
    data_obj = preprocess_data(data)
    value_counts = data_obj["Species"][data_obj["PetalWidthCm"] > 1.5].value_counts()
    return value_counts.to_dict()

def get_virginica_petal_length_above_6(data):
    data_obj = preprocess_data(data)
    result = data_obj[(data_obj["PetalLengthCm"] > 6) & (data_obj["Species"] == "Iris-virginica")]["ID"]
    
    return list(result)

def get_largest_sepal_width(data):
    data_obj = preprocess_data(data)
    
    max_sepal_width = data_obj["SepalWidthCm"].max()
    result = data_obj[data_obj["SepalWidthCm"] == max_sepal_width]["ID"]
    
    return list(result)[0]