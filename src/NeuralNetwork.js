import { Matrix } from 'tinymatrix';
import { sigmoid } from './activation';

class NeuralNetwork {
  constructor({
    inputNodes = 1,
    hiddenNodes = 1,
    outputNodes = 1,
    activation = sigmoid,
    learningRate = 0.1
  }) {
    this.inputNodes = inputNodes;
    activation = sigmoid;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;
    this.activation = activation;
    this.learningRate = learningRate;

    // Create weights from input -> hidden
    // And from hidden -> output
    this.weightsIH = new Matrix({
      rows: hiddenNodes,
      columns: inputNodes
    });
    this.weightsHO = new Matrix({
      rows: outputNodes,
      columns: hiddenNodes
    });

    // Generate random weights
    this.weightsIH = Matrix.map(this.weightsIH, () => Math.random() * 2 - 1);
    this.weightsHO = Matrix.map(this.weightsHO, () => Math.random() * 2 - 1);

    // Generate bias nodes
    this.biasH = new Matrix({ rows: hiddenNodes, columns: 1 });
    this.biasO = new Matrix({ rows: outputNodes, columns: 1 });
    this.biasH = Matrix.randomize(this.biasH, 2);
    this.biasO = Matrix.randomize(this.biasO, 2);
  }

  // get outputs from hidden layer
  getHiddenOutputs(inputs) {
    if (!inputs instanceof Matrix) {
      throw new Error('inputs to getHiddenOutputs must be instance of Array');
    }

    // multiply input weights by inputs and add hidden node bias
    let hiddenOutputs = Matrix.multiply(this.weightsIH, inputs);
    hiddenOutputs = Matrix.add(hiddenOutputs, this.biasH);
    hiddenOutputs = Matrix.map(hiddenOutputs, ([i, j]) =>
      this.activation(hiddenOutputs.values[i][j])
    );

    return hiddenOutputs;
  }

  // get final outputs
  getOutputs(hiddenOutputs) {
    // multiply hidden output weights by hidden outputs and add output bias
    let outputs = Matrix.multiply(this.weightsHO, hiddenOutputs);
    outputs = Matrix.add(outputs, this.biasO);

    // call activatoin function on outputs
    outputs = Matrix.map(outputs, ([i, j]) =>
      this.activation(outputs.values[i][j])
    );

    return outputs;
  }

  // FeedForward algorithm -
  // get outputs from hidden layer, feed them into output layer
  // then obtain results
  feedForward(input) {
    if (!input instanceof Array) {
      throw new Error('feedForward input must be an array');
    }

    // get inputs vector
    const inputs = Matrix.fromArray(input);

    // calculate hidden outputs
    const hiddenOutputs = this.getHiddenOutputs(inputs);

    // calculate outputs
    const outputs = this.getOutputs(hiddenOutputs);

    // send back outputs as array
    return Matrix.toArray(outputs);
  }

  // get gradient for generic outputs and errors
  getGradient(outputs, errors) {
    // calculate output gradients
    // first, use derivate of sigmoid function
    //    - not exactly - outputs have already been mapped to sigmoid
    let gradient = Matrix.map(
      outputs,
      ([i, j]) => outputs.values[i][j] * (1 - outputs.values[i][j])
    );
    // entrywise multiply gradient with errors
    gradient = Matrix.entrywiseProduct(gradient, errors);
    // multiply by learning rate
    gradient = Matrix.scale(gradient, this.learningRate);

    return gradient;
  }

  // get weights delta between generic input and output nodes
  getDeltaWeights(inputs, outputGradient) {
    // calculate deltas for inputs -> outputs
    const inputsT = Matrix.transpose(inputs);
    const deltaWeights = Matrix.multiply(outputGradient, inputsT);
    return deltaWeights;
  }

  train(inputsArray, targetsArray) {
    const inputs = Matrix.fromArray(inputsArray);
    // replicate feed forward - get hidden outputs from inputs
    const hiddenOutputs = this.getHiddenOutputs(inputs);
    // replicate feed forward - get outputs from hidddenOutputs
    const outputs = this.getOutputs(hiddenOutputs);
    // backpropagation!!!
    // calculate errors between target values and outputs
    const outputErrors = Matrix.subtract(
      Matrix.fromArray(targetsArray),
      outputs
    );
    // calculate deltas for hidden -> output using gradient
    const outputGradient = this.getGradient(outputs, outputErrors);
    // const hiddenT = Matrix.transpose(hiddenOutputs);
    // const deltaWeightsHO = Matrix.multiply(outputGradient, hiddenT);
    const deltaWeightsHO = this.getDeltaWeights(hiddenOutputs, outputGradient);
    // adjust hidden -> output weights by delta and bias by gradient
    this.weightsHO = Matrix.add(this.weightsHO, deltaWeightsHO);
    this.biasO = Matrix.add(this.biasO, outputGradient);
    // calculate hidden layer errors
    const weightsHOT = Matrix.transpose(this.weightsHO);
    const hiddenErrors = Matrix.multiply(weightsHOT, outputErrors);
    // calculate deltas for input -> hidden using gradient
    const hiddenGradient = this.getGradient(hiddenOutputs, hiddenErrors);
    // const inputsT = Matrix.transpose(inputs);
    // const deltaWeightsIH = Matrix.multiply(hiddenGradient, inputsT);
    const deltaWeightsIH = this.getDeltaWeights(inputs, hiddenGradient);
    // adjust input -> hidden weights by delta
    this.weightsIH = Matrix.add(this.weightsIH, deltaWeightsIH);
    this.biasH = Matrix.add(this.biasH, hiddenGradient);
  }
}

export default NeuralNetwork;
