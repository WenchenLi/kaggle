#!/usr/bin/python
import numpy as np
import csv
import random
from sklearn import preprocessing


import smtplib
from email.mime.text import MIMEText


def send_me_email(st,threshold):
    # Open a plain text file for reading.  For this example, assume that
    # the text file contains only ASCII characters.
    with open('textfile.txt', 'rw') as fp:
    # Create a text/plain message
        msg = MIMEText(fp.read())
    


    # me == the sender's email address
    # you == the recipient's email address
    msg['Subject'] = ' I found one good result at'+str(st)+"with threshold "+str(threshold)
    #msg['From'] = me
    #msg['To'] = [you]

    # Send the message via our own SMTP server, but don't include the
    # envelope header.
    s = smtplib.SMTP(host='smtp.gmail.com', port=587)
    s.set_debuglevel(1) 
    s.ehlo() 
    s.starttls() 
    s.ehlo() 
    s.login('sender email', 'pw') 
    s.sendmail('sender email', ['recipient email'], msg.as_string())
    s.quit()


def write_to_file(pred,param,eval_result,fname=None):
    path = '/home/wenchen/projects/xgboost/wrapper/result/'
    if fname is not None:
        str(fname)
    with open(path+'predict'+fname+'.csv', 'w') as fp:
        a = csv.writer(fp)
        a.writerow(["id","Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"])
        for i in range(pred.shape[0]):
            #summation=sum(pred[i,:])
            #a.writerow([i+1,pred[i,0]/summation,pred[i,1]/summation,pred[i,2]/summation,pred[i,3]/summation,pred[i,4]/summation,pred[i,5]/summation,pred[i,6]/summation,pred[i,7]/summation,pred[i,8]/summation])
            a.writerow([i+1,pred[i,0],pred[i,1],pred[i,2],pred[i,3],pred[i,4],pred[i,5],pred[i,6],pred[i,7],pred[i,8]])
    with open(path+'predict'+fname+'_param.csv', 'w') as fp:
        a = csv.writer(fp)
        for item in param:
            a.writerow([item,param[item]])
    with open(path+'predict'+fname+'_eval.csv', 'w') as fp:
        a = csv.writer(fp)
        for item in eval_result:
            a.writerow([item,eval_result[item]])


def make_dataset(train_data_array,train_label_array):
    dataset=np.zeros((train_data_array.shape[0],train_data_array.shape[1]+1))
    dataset[:,0:-1]=train_data_array
    dataset[:,-1]=train_label_array
    return dataset

def make_cross_validation(dataset,k=0.9):
    np.random.shuffle(dataset)
    row=int(dataset.shape[0]*k)
    X_train=dataset[0:row,0:-1]
    Y_train=dataset[0:row,-1]
    X_cross=dataset[row:-1,0:-1]
    Y_cross=dataset[row:-1,-1]
    return (X_train,Y_train,X_cross,Y_cross)

def read_data():
    train_data=[]
    train_label=[]
    test_data=[]
    with open("train.csv", "rb") as infile:
        reader = csv.reader(infile)
        next(reader, None)  # skip the headers
        for row in reader:
       # process each row
            train_label.append(int(row[-1][-1]))
            tem1=row[1:-1]
            tem2=[float(item) for item in tem1]
            train_data.append(tem2)

    with open("test.csv", "rb") as infile:
        reader = csv.reader(infile)
        next(reader, None)  # skip the headers
        for row in reader:
       # process each row
            tem1=row[1:len(row)]
            tem2=[float(item) for item in tem1]
            test_data.append(tem2)

    train_label_array=np.array(train_label)
    train_data_array=np.array(train_data)
    dataset=make_dataset(train_data_array,train_label_array)
    np.random.shuffle(dataset)
    train_label_array=dataset[:,-1]
    train_data_array=dataset[:,0:-1]
    test_data_array=np.array(test_data)
    scaler = preprocessing.MinMaxScaler().fit(train_data_array)
    train_data_array=scaler.transform(train_data_array)
    test_data_array=scaler.transform(test_data_array)
    return (train_data_array,train_label_array,test_data_array)
