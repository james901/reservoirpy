import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from reservoirpy import ESN, mat_gen

####################################################################################################
#                This is the first example of the ReservoirPy librairy, you'll learn how to create and
#                train your first Echo State Network to predict the chaotic behaviour of the MackeyGlass serie
#
#                This example script is also available in a Notbook version.
   #################################################################################################

#import the data
data = np.loadtxt('MackeyGlass_t17.txt')

# plot some of it to see what it looks like
plt.figure(0).clear()
plt.plot(data[0:1000])
plt.title('A sample of input data')
plt.show()

N = 200 #the number of neurons in the reservoir
sr = 1.25 #the spectral radius
sparsity = 0.2 #the sparisity of the reservoir (the number of non zero weights in the link matrix)
nb_features = 1 # We only have one number in input : the previous term of the sequence
regularization_coef =  1e-8 # a regularization coefficient
input_scale_factor = 1.0 # the scaling of the input values : here no rescaling of the input
leak_rate = 0.3

W = mat_gen.fast_spectra_initialization(N, spectral_radius=sr, proba = sparsity) #the weights of the reservoir

Win = mat_gen.generate_input_weights(nbr_neuron=N, dim_input=nb_features, #the input weights
                                    input_bias=True, input_scaling=input_scale_factor)


reservoir = ESN(lr=leak_rate, W=W, Win=Win, input_bias=True, ridge=regularization_coef)

print("the ESN is created !")


wash_initLen = 100 # number of time steps during which internal activations are washed-out during training
# we consider trainLen including the warming-up period (i.e. internal activations that are washed-out when training)
trainLen = wash_initLen + 1900 # number of time steps during which we train the network
testLen = 2000 # number of time steps during which we test the network

train_in = data[None,0:trainLen] # the train sequence at time t
train_out = data[None,0+1:trainLen+1] # the train sequence at time t+1 to be predicted : the teacher

test_in = data[None,trainLen:trainLen+testLen] # the test sequence at time t
test_out = data[None,trainLen+1:trainLen+testLen+1] # the test sequence at time t+1 to be predicted

# We reshape them to be compatible with the shape expected for the ESN
train_in, train_out = train_in.T, train_out.T
test_in, test_out = test_in.T, test_out.T

print( "train_in, train_out dimensions", train_in.shape, train_out.shape)
print( "test_in, test_out dimensions", test_in.shape, test_out.shape)


internal_states_train = reservoir.train(inputs=[train_in,], teachers=[train_out,], wash_nr_time_step=wash_initLen, verbose=False)

print("the ESN is trained !")


#We keep the last state of the reservoir because the test sequence follow the train sequence.
# You can use the `init_state` option to `run` to specify another initial state

output_pred, internal_states_sequence = reservoir.run(inputs=[test_in,])


#we only take into account the results after the `wash_initLen` first steps

mse = np.mean((test_out[wash_initLen:] - output_pred[0][wash_initLen:])**2) # Mean Squared Error: see https://en.wikipedia.org/wiki/Mean_squared_error for more details
rmse = np.sqrt(mse) # Root Mean Squared Error

print("Mean Squared error : ", mse)
print("Root Mean Squared error : ", rmse)


plt.figure()
plt.plot( internal_states_sequence[0][:200,:12])
plt.ylim([-1.1,1.1])
plt.title('Activations $\mathbf{x}(n)$ from Reservoir Neurons ID 0 to 11 for 200 time steps')


plt.figure(figsize=(12,4))
plt.plot(output_pred[0][wash_initLen:], color='red', lw=1.5, label="output predictions")
plt.plot(test_out[wash_initLen:], lw=0.75, label="real timeseries")
plt.title("Output predictions against real timeseries")
plt.legend()

plt.figure(figsize=(12,4))
plt.plot(output_pred[0][wash_initLen:wash_initLen + 50], color='red', lw=1.5, label="output predictions")
plt.plot(test_out[wash_initLen:wash_initLen + 50], lw=0.75, label="real timeseries")
plt.title("Output predictions against real timeseries (zoom)")
plt.legend()

plt.show()



