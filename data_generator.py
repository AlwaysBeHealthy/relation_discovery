import numpy as np
import random as random


class Data:
	def __init__(self, nid, hot_threshold=10, is_cold=False):
		self.__network_id = nid
		self.__directory = "./new_data/"
		self.__edge_file = nid+".edges"
		self.__feature_field_file = nid+".featnames"
		self.__feature_file = nid+".feat"
		self.__hot_threshold = hot_threshold
		self.__is_cold = is_cold

		self.__raw_friends_mat = {}
		self.__feature_field = {}
		self.__friends_mat = {}
		self.__user_id_list = []
		self.__id_feature = {}
		self.__feature = {}
		self.__test_mat = []

		# read in features' dimension
		with open(self.__directory+self.__feature_field_file) as fin:
			for line in fin:
				feature_tuple = line.strip().split(" ")
				field_idx = feature_tuple[0]
				feature_name = feature_tuple[1]
				if feature_name in self.__feature_field:
					self.__feature_field[feature_name][1] = int(field_idx)
				else:
					self.__feature_field[feature_name] = [int(field_idx), int(field_idx)]

		# build the friend matrix, row: uid, col: friend_id
		with open(self.__directory+self.__edge_file) as fin:
			for line in fin:
				id_tuple = line.strip().split(" ")
				if id_tuple[0] in self.__friends_mat:
					self.__friends_mat[id_tuple[0]].append(id_tuple[1])
					self.__raw_friends_mat[id_tuple[0]].append(id_tuple[1])
				else:
					self.__friends_mat[id_tuple[0]] = [id_tuple[1]]
					self.__raw_friends_mat[id_tuple[0]] = [id_tuple[1]]

		# read in users' features
		with open(self.__directory+self.__feature_file) as fin:
			for line in fin:
				feature_tuple = line.strip().split(" ")
				self.__feature[feature_tuple[0]] = []
				self.__user_id_list.append(feature_tuple[0])
				for one_hot in feature_tuple[1:]:
					self.__feature[feature_tuple[0]].append(int(one_hot))

		# generate the id feature
		for idx, user_id in enumerate(self.__user_id_list):
			one_hot = [0]*len(self.__user_id_list)
			one_hot[idx] = 1
			self.__id_feature[user_id] = one_hot

		print("Generating Test Set by Leave One Out")
		# leave one out as the test set
		# avoid isolated nodes, so some users will be skipped
		for user_id in self.__friends_mat:
			if not self.__is_cold:
				if len(self.__friends_mat[user_id]) >= self.__hot_threshold:
					# sample a friend id of this user
					rand_fid = self.__sample_friend_id(user_id)
					# move this link from the original friend matrix to the test friend matrix
					self.__test_mat.append([user_id, rand_fid])
					self.__friends_mat[user_id].remove(rand_fid)
					self.__friends_mat[rand_fid].remove(user_id)
			else:
				if 2 <= len(self.__friends_mat[user_id]) < self.__hot_threshold:
					# sample a friend id of this user
					rand_fid = self.__sample_friend_id(user_id)
					# move this link from the original friend matrix to the test friend matrix
					self.__test_mat.append([user_id, rand_fid])
					self.__friends_mat[user_id].remove(rand_fid)
					self.__friends_mat[rand_fid].remove(user_id)

	def __sample_friend_id(self, user_id):
		num_friend = len(self.__friends_mat[user_id])
		random_id = self.__friends_mat[user_id][random.randint(0, num_friend-1)]
		# remove this link only when this removal won't cause an isolate node
		while len(self.__friends_mat[random_id]) <= 1:
			random_id = self.__friends_mat[user_id][random.randint(0, num_friend - 1)]
		return random_id

	def __sample_stranger_id(self, user_id):
		random_id = self.__user_id_list[random.randint(0, len(self.__user_id_list)-1)]
		# if random_id is a friend of user, sample again
		while random_id in self.__friends_mat[user_id] or random_id == user_id:
			random_id = self.__user_id_list[random.randint(0, len(self.__user_id_list) - 1)]
		return random_id

	def __fill_X(self, X, data):
		X.append(data["id"])
		for feature_name in self.__feature_field:
			X.append(data[feature_name])

	def sample(self):
		user_data = {}
		friend_data = {}
		stranger_data = {}

		for feature_name in self.__feature_field:
			user_data[feature_name] = []
			friend_data[feature_name] = []
			stranger_data[feature_name] = []
		user_data["id"] = []
		friend_data["id"] = []
		stranger_data["id"] = []

		train_num = 0
		for user_id in self.__friends_mat:
			for friend_id in self.__friends_mat[user_id]:
				# sample a stranger_id
				stranger_id = self.__sample_stranger_id(user_id)

				for feature_name in self.__feature_field:
					start_idx = self.__feature_field[feature_name][0]
					end_idx = self.__feature_field[feature_name][1]
					user_data[feature_name].append(self.__feature[user_id][start_idx:end_idx+1])
					friend_data[feature_name].append(self.__feature[friend_id][start_idx:end_idx+1])
					stranger_data[feature_name].append(self.__feature[stranger_id][start_idx:end_idx+1])

				user_data["id"].append(self.__id_feature[user_id])
				friend_data["id"].append(self.__id_feature[friend_id])
				stranger_data["id"].append(self.__id_feature[stranger_id])
				train_num += 1

		X = []
		self.__fill_X(X, user_data)
		self.__fill_X(X, friend_data)
		self.__fill_X(X, stranger_data)
		return [X, train_num]

	def get_input_dimension(self):
		input_dimension = {}
		for feature_name in self.__feature_field:
			start_idx = self.__feature_field[feature_name][0]
			end_idx = self.__feature_field[feature_name][1]
			input_dimension[feature_name] = end_idx - start_idx + 1
		input_dimension["id"] = len(self.__user_id_list)
		return input_dimension

	def get_testset(self):
		return self.__test_mat

	def get_test_feature(self, user_id, friend_id):

		# one positive case

		X_u = [self.__id_feature[user_id]]
		X_f = [self.__id_feature[friend_id]]
		for feature_name in self.__feature_field:
			start_idx = self.__feature_field[feature_name][0]
			end_idx = self.__feature_field[feature_name][1]
			X_u.append(self.__feature[user_id][start_idx:end_idx + 1])
			X_f.append(self.__feature[friend_id][start_idx:end_idx + 1])

		positive = [X_u+X_f]

		# all negative cases
		negative = []
		for stranger_id in self.__user_id_list:
			if stranger_id != friend_id and stranger_id != user_id and (stranger_id not in self.__raw_friends_mat[user_id]):
				X_s = [self.__id_feature[stranger_id]]
				for feature_name in self.__feature_field:
					start_idx = self.__feature_field[feature_name][0]
					end_idx = self.__feature_field[feature_name][1]
					X_s.append(self.__feature[stranger_id][start_idx:end_idx + 1])
				negative.append(X_u+X_s)

		res = []
		for idx in positive[0]:
			res.append([idx])
		for row in negative:
			for idx, field in enumerate(row):
				res[idx].append(field)

		return res
