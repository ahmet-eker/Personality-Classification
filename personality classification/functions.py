import pandas as pd
import numpy as np

def decode(personality):  # this function is for decoding personalities for the information part at the end
	if personality == 0:
		return "ESTJ"
	elif personality == 1:
		return "ENTJ"
	elif personality == 2:
		return "ESFJ"
	elif personality == 3:
		return "ENFJ"
	elif personality == 4:
		return "ISTJ"
	elif personality == 5:
		return "ISFJ"
	elif personality == 6:
		return "INTJ"
	elif personality == 7:
		return "INFJ"
	elif personality == 8:
		return "ESTP"
	elif personality == 9:
		return "ESFP"
	elif personality == 10:
		return "ENTP"
	elif personality == 11:
		return "ENFP"
	elif personality == 12:
		return "ISTP"
	elif personality == 13:
		return "ISFP"
	elif personality == 14:
		return "INTP"
	elif personality == 15:
		return "INFP"

def encode(personality):  # this function is for encoding the personalities to integer
	if personality == "ESTJ":
		return 0
	elif personality == "ENTJ":
		return 1
	elif personality == "ESFJ":
		return 2
	elif personality == "ENFJ":
		return 3
	elif personality == "ISTJ":
		return 4
	elif personality == "ISFJ":
		return 5
	elif personality == "INTJ":
		return 6
	elif personality == "INFJ":
		return 7
	elif personality == "ESTP":
		return 8
	elif personality == "ESFP":
		return 9
	elif personality == "ENTP":
		return 10
	elif personality == "ENFP":
		return 11
	elif personality == "ISTP":
		return 12
	elif personality == "ISFP":
		return 13
	elif personality == "INTP":
		return 14
	elif personality == "INFP":
		return 15

def scale():  # scaling dataframe and returning a data array and a personality array

	#region           reads the csv file as df, deletes Response Id and personality, turns to array, encodes personality -> ------df_arr------  ------personality_arr------
	df_raw = pd.read_csv("16P.csv", encoding='mac_roman')
	personality_list = list(df_raw["Personality"])
	df = df_raw.drop(columns=["Response Id","Personality"])
	df_arr = np.array(df)
	for t in range(len(personality_list)):
		personality_list[t] = encode(personality_list[t])
	personality_arr = np.array(personality_list)
	#endregion
	
	return df_arr,personality_arr

def determine_max_and_min(data_array):  # [[1,2,3,4] get array as inputs an returns returns min and max -> ------max,min--------
										#  [2,3,4,5]
	max = data_array[0,0]				#  [3,4,5,6]
	min = data_array[0,0]               #  [4,5,6,7]
	for row in data_array:              #  [5,6,7,8]
		for member in row:              #  [6,7,8,9]]
			if max > member:            
				max = member
			if min < member:
				min = member
	return max,min                      # returns 1,9   min,max

def feature_normalization(data_array):  # normalizes every value of given array, returns normalized array, reassignment needed afterwards
	data_array = np.asfarray(data_array)
	min , max = determine_max_and_min(data_array)
	for i,row in enumerate(data_array):
		for t,member in enumerate(row):
			data_array[i,t] = (member - min)/(max - min)
	return data_array

def slicing_train_and_test(data_array,test_index,personality_arr):  # returns test(k) and train(4k) part in this order
	length = len(data_array)
	part_len = (length//5)
	if test_index ==1:
		personality_test_arr = personality_arr[:part_len]
		personality_train_arr = personality_arr[part_len:]
		return data_array[:part_len], data_array[part_len:], personality_test_arr, personality_train_arr
	elif test_index ==2:
		personality_test_arr = personality_arr[part_len:part_len*2]
		personality_train_arr = np.concatenate((personality_arr[:part_len], personality_arr[part_len*2:]))
		return data_array[part_len:part_len*2], np.vstack((data_array[:part_len],data_array[part_len*2:])), personality_test_arr, personality_train_arr
	elif test_index ==3:
		personality_test_arr = personality_arr[part_len*2:part_len*3]
		personality_train_arr = np.concatenate((personality_arr[:part_len*2], personality_arr[part_len*3:]))
		return data_array[part_len*2:part_len*3], np.vstack((data_array[:part_len*2],data_array[part_len*3:])), personality_test_arr, personality_train_arr
	elif test_index ==4:
		personality_test_arr = personality_arr[part_len*3:part_len*4]
		personality_train_arr = np.concatenate((personality_arr[:part_len*3], personality_arr[part_len*4:]))
		return data_array[part_len*3:part_len*4], np.vstack((data_array[:part_len*3],data_array[part_len*4:])), personality_test_arr, personality_train_arr
	elif test_index ==5:
		personality_test_arr = personality_arr[part_len*4:]
		personality_train_arr = personality_arr[:part_len*4]
		return data_array[part_len*4:], data_array[:part_len*4], personality_test_arr, personality_train_arr

def calculate_accuracy(conf_matrix):  # returns accuracy
	true = sum([conf_matrix[i,i] for i in range(len(conf_matrix))])
	all = np.sum((conf_matrix))
	return true/all

def knn(slice_point,k,is_feature_normalization): # return confussion_matrix and misclassified_samples
	if slice_point==5:  # determining parts length for matrix multiplication
		test_len , train_len = 12003 , 47996
	else:
		test_len , train_len = 11999 , 48000
	df_arr , personality_arr = scale()
	if is_feature_normalization==True:
		df_arr = feature_normalization(df_arr)
	test_arr, train_arr, per_test_arr, per_train_arr = slicing_train_and_test(df_arr,slice_point,personality_arr)
	confussion_matrix = np.zeros((16,16))  # creating a confussion matrix
	test_part = (test_arr*test_arr).sum(axis=1)
	test_part = test_part.reshape((test_len,1))*np.ones(shape=(1,train_len))
	train_part = (train_arr*train_arr).sum(axis=1)
	train_part = train_part*np.ones(shape=(test_len,1))
	sum =  test_part + train_part -2*test_arr.dot(train_arr.T)  #  matrix multiplication
	nearest_personalities = per_train_arr[np.argsort(sum)[:,:k]]  #  nearest points indexs
	misclassified_points_indexs = [person for person in range(len(nearest_personalities)) if list(np.bincount(nearest_personalities[person])).count(np.max(np.bincount(nearest_personalities[person]))) > 1]  # misclassified samples
	predict = np.array([np.bincount(row).argmax() for row in nearest_personalities])  # predict value
	for m in range(len(per_test_arr)):
		confussion_matrix[per_test_arr[m],predict[m]] +=1  # adding informations to confussing matrix
	def calculate_precision_recall_and_print(conf_matrix):  # calculating and printing the precision and recall
		def precision_recall_calculate(per):  # returns precision and recall values
			tp = conf_matrix[per, per]  # true positive
			fn = np.sum(np.concatenate((conf_matrix[per,:per],conf_matrix[per,per+1:])))  # false negative
			fp = np.sum(np.concatenate((conf_matrix[:per,per],conf_matrix[per+1:,per])))  # false positive
			#tn = np.sum(conf_matrix)-(tp+fn+fp)  true negatives
			def precision():
				return tp/(tp+fp)
			def recall():
				return tp/(tp+fn)
			return precision(),recall()
		def print_calculations():
			sum_precision = 0
			sum_recall = 0
			for row in range(len(conf_matrix)):
				precision , recall = precision_recall_calculate(row)
				sum_precision+=precision
				sum_recall+=recall
				personality = decode(row)
				print(personality)
				print("Precision =", precision)
				print("Recall =", recall,end="\n\n")
			print("Acuracy = {}".format(calculate_accuracy(conf_matrix)))
			print("Average Precision = {}".format(sum_precision/16))
			print("Average Recall = {}\n".format(sum_recall/16))
			print("Total Misclassified Samples :", len(misclassified_points_indexs))
			if len(misclassified_points_indexs) != 0:
				print("Misclassified Samples Index :", misclassified_points_indexs)
		print_calculations()
	calculate_precision_recall_and_print(confussion_matrix)

#  example run  --->     knn(3, 7, False)    -->    k-fold  ,  k nearest  ,  feature normalization