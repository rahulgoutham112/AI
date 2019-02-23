import csv
import random
import nltk
from sklearn.model_selection import train_test_split
#import random

names_temp=[]
names = []

def features(name):
    if name == "": #put this check to avoid String index error when name is empty
        return {
            "Last-Letter": ''
        }
    else:    
        return {
            "Last-Letter": name[0][-1]
        }

# Loading female name csv
with open('/home/rahul/Python Practice/GenderDetection/Indian-Female-Names.csv', 'rt') as f:
    reader = csv.reader(f)
    for row in reader:
        names_temp.append([row[0], 'female'])
names = names_temp[1:]

names_temp = []
# Loading male name csv
with open('/home/rahul/Python Practice/GenderDetection/Indian-Male-Names.csv', 'rt') as m:
    reader = csv.reader(m)
    for row in reader:
        names_temp.append([row[0], 'male'])
print(names_temp[:10])
names =  names + names_temp[1:]

random.shuffle(names)

Train_data_features = [(features(n), gender) for n, gender in names]

#Created Train and Test sets
Train_set, Test_set = Train_data_features[:15000], Train_data_features[15001:]

clf = nltk.NaiveBayesClassifier.train(Train_set)
#Test some names
print(clf.classify(features('Kushal')))
print(clf.classify(features('amit')))

#Check the accuracy
print(nltk.classify.accuracy(clf, Test_set))

#Check the ratio of correctness
print(clf.show_most_informative_features(10))


