Currently, when impoting the resnet model to torch-mlir (using torch.export()) 
many params of the model are expected to be passed as input. The number of input tensors 
increases from 1 to over 500. My solution by now isn't pretty, but it works. 
Basically I extract the params from the model and manually insert them into the mlir model.
