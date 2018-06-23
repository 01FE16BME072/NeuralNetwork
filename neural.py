import numpy as np

print("Hi Welcome for finding out the Grade Values")

datasets = [
				[38,96,10],
                [46,36,10],
                [49,25,10],
                [39,54,10],
				[30,76,10],
				[32,43,10],
				[41,76,10],
				[22,84,9],
				[24,70,9],
				[14,47,9],
				[38,67,9],
			]

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

# print(w1,w2,b)

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_p(x):
	return sigmoid(x)*(1-sigmoid(x))

learning_rate = 0.5

for i in range(50000):
	ri  = np.random.randint(len(datasets))
	point = datasets[ri]
	# print(point)

	x = point[0]*w1 + point[1]*w2 + b
	y = sigmoid(x)

	target = point[2]

	cost = (y - target)**2

	dcost_dy = 2*(y - target)

	dy_dx = sigmoid_p(x)

	dx_dw1 = point[0]
	dx_dw2 = point[1]
	dx_db = 1

	dcost_dw1 = dcost_dy * dy_dx * dx_dw1
	dcost_dw2 = dcost_dy * dy_dx * dx_dw2
	dcost_db = dcost_dy * dy_dx * dx_db

	w1 = w1 - learning_rate*dcost_dw1
	## print(w1)
	w2 = w2 - learning_rate*dcost_dw2
	b = b - learning_rate*dcost_db

def predict_data(length,width):
	x = point[0]*w1 + point[1]*w2 + b
	y = sigmoid(x)

	print(y)
	if y < 0.5:
		print("The Grade value found out through my brain/neural analysis is 9")

	else:
		print("The Grade value found out through my brain/neural analysis is 10")



a = 50
c = 100
predict_data(a,c)
