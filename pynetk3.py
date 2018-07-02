import numpy as np
import time
#Sigmoidal Activation Function
def sgm(x,Derivative = False):
    if not Derivative:
        return 1/(1 + np.exp(-x))
    else:
        derivative_out = sgm(x)
        return derivative_out*(1 - derivative_out)

#Linear Activation Function
def linear(x,Derivative = False):
    if not Derivative:
        return x
    else:
        return 1.0

#Hyperbolic Activation Function
def tanh(x,Derivative = False):
    if not derivative:
        return np.tanh(x)
    else:
        return 1.0 - np.tanh(x)**2

#NeuralNetwork class where we write different methods
class NeuralNetwork:
    layercount = 0
    size = None
    weights = []
    activation_function = []

    #init method of our class
    def __init__(self,layer_size,layer_activation_function = None):

        if layer_activation_function is None:
            layerfun = []
            for i in range(self.layercount):
                if i == self.layercount - 1:
                    layerfun.append(sgm)
                else:
                    layerfun.append(linear)
        else:
            if len(layer_size) != len(layer_activation_function):
                raise ValueError('Length of the activation functions and layersize arguments does not match')
            elif layer_activation_function[0] is not None:
                raise ValueError('Input Layer wont have any activation functions')
            else:
                layerfun = layer_activation_function[1:]

        self.activation_function = layerfun
        self.layercount = len(layer_size) - 1
        self.size = layer_size
        self.LayerInput = []
        self.LayerOutput = []

        for (layer1,layer2) in zip(layer_size[:-1],layer_size[1:]):
            self.weights.append(np.random.normal(scale = 0.1,size = (layer2 ,layer1 + 1)))
    
    #Building our NeuralNetwork
    def build(self,data):
        self.data = data
        inputsizenumber = data.shape[0]
        self.LayerInput = []
        self.LayerOutput = []

        for index in range(self.layercount):
            if index == 0:
                lateinput = self.weights[0].dot(np.vstack([data.T,np.ones([1,inputsizenumber])]))
            else:
                lateinput = self.weights[index].dot(np.vstack([self.LayerOutput[-1],np.ones([1,inputsizenumber])]))

            self.LayerInput.append(lateinput)
            self.LayerOutput.append(self.activation_function[index](lateinput))

        return self.LayerOutput[-1].T

    #Compiling our NeuralNetwork that is BackPropagation
    def compile(self,data,target,learning_rate = 0.2):
        self.build(data)

        delta_measure = []
        input_size_num = data.shape[0]

        for index in reversed(range(self.layercount)):
            if index == self.layercount - 1:
                output_error = self.LayerOutput[index] - target.T
                toterror = np.sum(output_error**2)
                delta_measure.append(output_error * self.activation_function[index](self.LayerInput[index],Derivative = True))
            else:
                delta_measure_prev = self.weights[index + 1].T.dot(delta_measure[-1])
                delta_measure.append(delta_measure_prev[:-1,:] * self.activation_function[index](self.LayerInput[index],Derivative = True))

        for index in range(self.layercount):
            delta_measure_index = self.layercount - 1 - index

            if index == 0:
                layeroutput = np.vstack([data.T,np.ones([1,input_size_num])])
            else:
                layeroutput = np.vstack([self.LayerOutput[index -1],np.ones([1,self.LayerOutput[index - 1].shape[1]])])

            weight_delta = np.sum((layeroutput[None,:,:].transpose(2,0,1) * delta_measure[delta_measure_index][None,:,:].transpose(2,1,0)),axis = 0)

            self.weights[index] -= learning_rate * weight_delta

        return toterror

    def predict(self,data):
        pass

nn = NeuralNetwork(layer_size = (2,5,6,1),layer_activation_function = [None,sgm,sgm,sgm])
da = np.array([[1,1],[0,0],[1,0],[0,1]])
da_out = np.array([[0.05],[0.05],[0.95],[0.95]])

for i in range(1000001):
    error = nn.compile(da,da_out)
    if i % 10000 == 0:
        print error
    if error <= 1e-5:
        break
time.sleep(20)
pred_out = nn.build(da)
print pred_out
