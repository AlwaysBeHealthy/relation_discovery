import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import data_generator
import keras.backend as K
from sklearn.metrics import roc_auc_score

layers = keras.layers
embedding_dim = 10
epoch_num = 1000


def wide_part(input_a, input_b, embedding_map_list, first_order_weight, side_info_balance):
	embedding_a_list = []
	embedding_b_list = []
	input_list = []
	for idx, i in enumerate(input_a):
		input_list.append(i)
		embedding_a_list.append(embedding_map_list[idx](i))
	for idx, i in enumerate(input_b):
		input_list.append(i)
		embedding_b_list.append(embedding_map_list[idx](i))

	merged_inputs_first_order = layers.concatenate(input_list, axis=-1)
	first_order_output = first_order_weight(merged_inputs_first_order)

	# Generate the inner product layer
	inner_product_list = [layers.dot([embedding_a_list[0], embedding_b_list[0]], axes=1)]

	for i in range(1, len(embedding_a_list)):
		for j in range(i+1, len(embedding_a_list)):
			inner_product_list.append(layers.dot([embedding_a_list[i], embedding_a_list[j]], axes=1))
	for i in range(1, len(embedding_b_list)):
		for j in range(i+1, len(embedding_b_list)):
			inner_product_list.append(layers.dot([embedding_b_list[i], embedding_b_list[j]], axes=1))

	for i in range(1, len(embedding_a_list)):
		for j in range(1, len(embedding_b_list)):
			inner_product_list.append(layers.dot([embedding_a_list[i], embedding_b_list[j]], axes=1))

	inner_product_list.append(first_order_output)
	fm_output = layers.concatenate(inner_product_list, axis=-1)
	fm_output = side_info_balance(fm_output)
	return fm_output


def deep_part(input_a, input_b, embedding_map_list, deep_layers):
	embedding_list = []
	for idx, i in enumerate(input_a):
		embedding_list.append(embedding_map_list[idx](i))
	for idx, i in enumerate(input_b):
		embedding_list.append(embedding_map_list[idx](i))

	output = layers.concatenate(embedding_list)

	for layer in deep_layers:
		output = layers.Dropout(0.5)(layer(output))
	return output


def get_auc(model, test_set, data):
	auc = 0.0
	for idx, single_sample in enumerate(test_set):
		LOO_set = data.get_test_feature(single_sample[0], single_sample[1])
		score_list = model.predict(LOO_set)
		target = [0 for _ in score_list]
		target[0] = 1
		# unfold score_list
		score_list = [number for _number_ in score_list for number in _number_]
		if len(score_list) > 1:
			auc += roc_auc_score(target, score_list)
	auc /= len(test_set)
	return auc


def main():
	debug_switch = False

	print("Reading Data ...")
	network_id = "2"
	data = data_generator.Data(network_id)

	# get the test set and the size of it
	test_set = data.get_testset()

	print("Building Model ...")
	input_dimension = data.get_input_dimension()

	user_inputs = []
	friend_inputs = []
	stranger_inputs = []

	# the input layer and shared embedding layers
	id_dim = input_dimension["id"]
	user_inputs.append(layers.Input(shape=(id_dim,)))
	friend_inputs.append(layers.Input(shape=(id_dim,)))
	stranger_inputs.append(layers.Input(shape=(id_dim,)))
	embedding_map_list = [layers.Dense(embedding_dim)]

	for feature_name in input_dimension:
		if feature_name != "id":
			dim = input_dimension[feature_name]
			user_inputs.append(layers.Input(shape=(dim, )))
			friend_inputs.append(layers.Input(shape=(dim, )))
			stranger_inputs.append(layers.Input(shape=(dim, )))
			embedding_map_list.append(layers.Dense(embedding_dim))

	first_order_weight = layers.Dense(1, name="first_order")
	side_info_balance = layers.Dense(1)

	# the deep layer stack
	layer_size = [128, 64, 32]
	layer_stack = []

	for tensor_size in layer_size:
		layer_stack.append(layers.Dense(tensor_size, activation="relu"))
	layer_stack.append(layers.Dense(1))

	balance_layer = layers.Dense(1, name="wd_balance", kernel_constraint=keras.constraints.non_neg())

	# friend part
	friend_wide_score = wide_part(user_inputs, friend_inputs, embedding_map_list, first_order_weight, side_info_balance)
	friend_deep_score = deep_part(user_inputs, friend_inputs, embedding_map_list, layer_stack)
	friend_score = balance_layer(layers.concatenate([friend_wide_score, friend_deep_score]))

	# stranger part
	stranger_wide_score = wide_part(user_inputs, stranger_inputs, embedding_map_list, first_order_weight, side_info_balance)
	stranger_deep_score = deep_part(user_inputs, stranger_inputs, embedding_map_list, layer_stack)
	stranger_score = balance_layer(layers.concatenate([stranger_wide_score, stranger_deep_score]))

	# friend_score - stranger_score
	deepfm_out = layers.subtract([friend_score, stranger_score], name="diff_layer")
	# deepfm_out = layers.subtract([friend_wide_score, stranger_wide_score], name="diff_layer")
	# deepfm_out = layers.subtract([friend_deep_score, stranger_deep_score], name="diff_layer")
	deepfm_out = layers.Dense(1, activation="sigmoid", trainable=False, kernel_initializer="ones", name="sigmoid_output")(deepfm_out)

	score_model = keras.Model(inputs=user_inputs+friend_inputs, outputs=friend_wide_score)
	deepfm_model = keras.Model(inputs=user_inputs+friend_inputs+stranger_inputs, outputs=deepfm_out)
	deepfm_model.compile(loss='binary_crossentropy', optimizer='adam')

	print("AUC before training:", get_auc(score_model, test_set, data))
	for epoch_idx in range(epoch_num):
		X, train_num = data.sample()
		deepfm_model.fit(X, np.ones(train_num), batch_size=32, epochs=1, shuffle=True, verbose=2)
		if epoch_idx % 10 == 0:
			print("AUC:", get_auc(score_model, test_set, data))
			print(deepfm_model.get_layer("wd_balance").get_weights())


if __name__ == '__main__':
	main()

