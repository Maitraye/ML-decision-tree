import csv

def parse(filename):
  # initialize variables
    out = []  
    csvfile = open(filename,'r')
    fileToRead = csv.reader(csvfile)
    headers = next(fileToRead)

  # iterate through rows of actual data
    for row in fileToRead:
        out.append(dict(zip(headers, row)))
    return out