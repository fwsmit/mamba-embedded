In this todo list the tasks are described for testing the ONNX models from the hyperparameter optimization on the microcontroller. The models at the
pareto front of accuracy and latency and models that are close to the pareto fron are tested. They are quantized, the quantized model is tested on the laptop and
then the model is tested on the microcontroller for a latency test. The top 10% of models are tested.

Pick the first item that is not finished and write code to do this task.

- [x] Find the top 10% of models. Greedily add items one at a time, always choosing the one that increases hypervolume most
- [ ] Quantize the model and measure the accuracy of the quantized model on the PC.
- [ ] Store the unquantized accuracy and the quantized accuracy of the model in a json file (all results of one study should be in one json file)
- [ ] Run the model on the microcontroller (in a loop 10 times) and find the average latency.
- [ ] Store the PC latency and the microcontroller latency in the same json file
- [ ] Record the memory summary stats on the microcontroller and store in the json file also
