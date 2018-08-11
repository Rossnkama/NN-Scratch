# Adding hidden layers for deeper pattern extraction

import numpy as np

def nonlin(x, deriv=False):
	if deriv == True:
		return x * (1 - x)
	return 1 / (1 + np.exp(-x))

X = np.array([
		[0, 0, 1],
		[1, 1, 1],
		[1, 0, 1],
		[0, 1, 1]
	])

y = ([
		[0],
		[1],
		[1],
		[0]
	])

np.random.seed(1)

# Random init of weights: mean(0), variance(1)
synapse0 = 2 * np.random.random((3, 4)) - 1
synapse1 = 2 * np.random.random((4, 1)) - 1

for j in xrange(60000):

	# Forward pass from input -> hidden -> output
	layer0 = X
	layer1 = nonlin(np.dot(layer0, synapse0))
	layer2 = nonlin(np.dot(layer1, synapse1))

	# Error
	l2_error = y - layer2

	if (j % 10000) == 0:
		print("Error: " + str(np.mean(np.abs(l2_error))))

	layer2_delta = l2_error * nonlin(layer2, deriv=True)

	# How much did l1 val contribute to l2
	# errer (according to weights)?
	l1_error = layer2_delta.dot(synapse1.T)
	layer1_delta = l1_error * nonlin(layer1, deriv=True)

	synapse1 += layer1.T.dot(layer2_delta)
	synapse0 += layer0.T.dot(layer1_delta)

print("Output after training:")
print(layer2)
