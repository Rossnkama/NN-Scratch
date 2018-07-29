import numpy as np

# Sigmoid function
# Gets us probability distribution \propto{weights}
def nonlin(x, deriv=False):
	if deriv == True:
		return x * (1 - x)
	return 1/(1 + np.exp(-x))

# Our dataset
X = np.array([
		[0, 0, 1],
		[1, 1, 1],
		[1, 0, 1],
		[0, 1, 1]
	])

# Transposition of the matrix below for appropriate shape
y = np.array([[0, 1, 1, 0]]).T

"""
	As we can see, the leftmost column is perfectly correlated to the
	output, we're going to see if our NN can work this out.
"""

# Seeding random numbers to make calculations deterministic
np.random.seed(1)

# Initialising weight with a mean of 0 and variance of 1
# Shape of weight tensor: 3 x 1
weight_matrix = 2 * np.random.random((3, 1)) - 1

for epoch in xrange(10000):

	# Feed forward
	input_layer = X
	# probability_dist_of(weighted_input)
	output_layer_hyp = nonlin(np.dot(input_layer, weight_matrix))

	# Calculating loss
	loss = y - output_layer_hyp
	print(loss)
	print('\n')

	# Multiplying error by slope of
	# sigmoid at output layer
	# A.k.a Error weighted derivative
	output_delta = loss * nonlin(output_layer_hyp, True)

	# Updating weights
	# Weight update step \ispropto{loss}
	weight_matrix += np.dot(input_layer.T, output_delta)

print("Output after training:")
print(output_layer_hyp)