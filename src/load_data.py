import csv


def get_POLLEN_2016():
	with open('../data/data2016-POLLEN.csv', 'rb') as csvfile:
	    data_from_csv = csv.reader(csvfile, delimiter=',')
	    headers = next(data_from_csv)
	    actual_data = [row[:] for row in data_from_csv]

	    return headers,actual_data

def get_POLLEN_2017():
	with open('../data/data2017-POLLEN.csv', 'rb') as csvfile:
	    data_from_csv = csv.reader(csvfile, delimiter=',')
	    headers = next(data_from_csv)
	    actual_data = [row[:] for row in data_from_csv]

	    return headers,actual_data


	    