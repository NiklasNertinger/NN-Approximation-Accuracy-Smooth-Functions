import torch
from function_training import run_experiment, calculate_weights
from architectures import standard_architecture, increasing_architecture, decreasing_architecture, diamond_architecture, sandglass_architecture



#examplary parameters; For real application you would want epochs and size to be in the scope of 100 times the size
epochs = 2500
train_size = 1000
test_size = 1000
lr = 0.1
optimizer_type = "adam" #"SGD" is the alternative
device = "cuda" if torch.cuda.is_available() else "cpu"
function = "f1" #f2 to f6 also possible
num_experiments = 3 #number of experiments per architecture
network_type = "FF" #"MLP" for a Multilayer Perceptron, "FF" for a Feedforward Neural Network
early_stopping = (10000, 5) #Early Stopping if loss improves by less than 5% in 10000 epochs
animation = False #Creates an animation of the neural network as a function. Requires installation of ffmpeg

#generation of examplary architectures
architectures = []
for depth in [2, 3, 4]:
    for width in [8, 16, 24]:
        architectures.append(standard_architecture(depth, width)) #standard_architecture gibt Liste der Länge 2 mit Netzwerkarchitektur und Name zurück

#run experiments for all architectures
for hidden_layers, architecture in architectures:
    run_experiment(hidden_layers, epochs, train_size, test_size, lr, optimizer_type, device, function, num_experiments, network_type, architecture, early_stopping, animation)