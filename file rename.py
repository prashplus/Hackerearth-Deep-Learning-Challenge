import csv
import os

with open('train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    linecount = 0
    for row in readCSV:
        if linecount == 0:
            linecount +=1
            continue
        new_name = row[1] + "." + row[0]
        old_name = row[0]
        try:
            os.rename(old_name,new_name)
        except:
            print("File Not found")
        linecount+=1
    print("Files modified:" + str(linecount))