import load_data
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict
import utils

headers, data1 = load_data.get_POLLEN_2016()
_, data2 = load_data.get_POLLEN_2017()

headers, data1, data2 = np.asarray(headers), np.asarray(data1), np.asarray(data2)

n_attributes = headers.shape[0]

total_data = np.concatenate((data1, data2))

n_samples = total_data.shape[0]


print 'Number of attributes: ' + str(n_attributes) +'\n'

print 'Number of samples: ' + str(n_samples) +'\n'


print 'Headers: ' 
print headers

print '\n'

print 'Example data before removing the first 2 columns: ' 
print total_data[150:155,:]

total_data_no_dates = total_data[:,2:]
dates = total_data[:,1]

months_nums=[1,2,3,4,5,6,7,8,9,10,11,12]
months_text=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

months_labels = []

for date in dates:

    for i in range(12):
        if months_text[i] in date:
            months_labels.append(months_nums[i])

months_labels = np.asarray(months_labels)

months_labels = months_labels.reshape((months_labels.shape[0],1))

print dates[25:35]
print months_labels[25:35]

data_to_shuffle = np.concatenate((months_labels, total_data_no_dates), axis = 1)

np.random.shuffle(data_to_shuffle)

labels = data_to_shuffle[:,0]
data = data_to_shuffle[:,1:]

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if ',' in data[i,j]:
            data[i,j] = data[i,j].replace(',','.')
        else:
            data[i,j] = data[i,j]+'.'
#We also observed during debugging that some fields are null, so we set a value of zero for these fields
        if data[i,j]=='.':
            data[i,j]='0.'

data = data.astype(np.float)

clf = tree.DecisionTreeClassifier()
clf.fit(data[1:40,:],labels[1:40])

y_pred = clf.predict(data[1:40])


scores = accuracy_score(labels[1:40],y_pred)
conf_mat = confusion_matrix(labels[1:40],y_pred)

print 'Scores for the 10 models that were created by the CV:'
print scores

utils.plot_confusion_matrix(conf_mat, classes=months_text,
                      title='Confusion matrix')

print 'Recall for each label: '
print recall_score(labels[1:40], y_pred, average=None)  

scores = cross_val_score(clf, data, labels, cv=10)
y_pred = cross_val_predict(clf, data, labels, cv=10)
conf_mat = confusion_matrix(labels,y_pred)

print 'Scores for the 10 models that were created by the CV:'
print scores
print '\n'
print 'The mean score: '
print scores.mean()

utils.plot_confusion_matrix(conf_mat, classes=months_text,
                      title='Confusion matrix')

print 'Recall for each label: '
print recall_score(labels, y_pred, average=None)  

seasons_numbers = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
seasons_text = ['Winter', 'Spring', 'Summer', 'Autumn']
#1 -> Winter, 2 -> Spring, 3 -> Summer, 4 -> Autumn (for  every month)

season_labels = []

for date in dates:

    for i in range(12):
        if months_text[i] in date:
            season_labels.append(seasons_numbers[i])
            
season_labels = np.asarray(season_labels)

season_labels = season_labels.reshape((season_labels.shape[0],1))

data_to_shuffle = np.concatenate((season_labels, total_data_no_dates), axis = 1)

np.random.shuffle(data_to_shuffle)

labels = data_to_shuffle[:,0]
data = data_to_shuffle[:,1:]

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if ',' in data[i,j]:
            data[i,j] = data[i,j].replace(',','.')
        else:
            data[i,j] = data[i,j]+'.'
#We also observed during debugging that some fields are null, so we set a value of zero for these fields
        if data[i,j]=='.':
            data[i,j]='0.'

data = data.astype(np.float)


scores = cross_val_score(clf, data, labels, cv=10)
y_pred = cross_val_predict(clf, data, labels, cv=10)
conf_mat = confusion_matrix(labels,y_pred)

print 'Scores for the 10 models that were created by the CV:'
print scores
print '\n'
print 'The mean score: '
print scores.mean()

utils.plot_confusion_matrix(conf_mat, classes=seasons_text,
                      title='Confusion matrix')

from sklearn.metrics import recall_score

print 'Recall for each label: '
print recall_score(labels, y_pred, average=None)  


seasons_numbers = [1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 1]
seasons_text = ['Winter-Autumn', 'Spring', 'Summer']
#1 -> Winter-Autumn, 2 -> Spring, 3 -> Summer

season_labels = []

for date in dates:

    for i in range(12):
        if months_text[i] in date:
            season_labels.append(seasons_numbers[i])
            
season_labels = np.asarray(season_labels)

season_labels = season_labels.reshape((season_labels.shape[0],1))

data_to_shuffle = np.concatenate((season_labels, total_data_no_dates), axis = 1)

np.random.shuffle(data_to_shuffle)

labels = data_to_shuffle[:,0]
data = data_to_shuffle[:,1:]

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if ',' in data[i,j]:
            data[i,j] = data[i,j].replace(',','.')
        else:
            data[i,j] = data[i,j]+'.'
#We also observed during debugging that some fields are null, so we set a value of zero for these fields
        if data[i,j]=='.':
            data[i,j]='0.'

data = data.astype(np.float)


scores = cross_val_score(clf, data, labels, cv=10)
y_pred = cross_val_predict(clf, data, labels, cv=10)
conf_mat = confusion_matrix(labels,y_pred)

print 'Scores for the 10 models that were created by the CV:'
print scores
print '\n'
print 'The mean score: '
print scores.mean()

utils.plot_confusion_matrix(conf_mat, classes=seasons_text,
                      title='Confusion matrix')

from sklearn.metrics import recall_score

print 'Recall for each label: '
print recall_score(labels, y_pred, average=None)  