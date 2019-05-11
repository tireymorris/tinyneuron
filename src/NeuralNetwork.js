import { Matrix } from 'tinymatrix';
import { sigmoid } from './activation';

class NeuralNetwork {
  constructor({
    inputNodes = 1,
    hiddenNodes = 1,
    outputNodes = 1,
    activation = sigmoid
  }) {
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;
    this.activation = activation;

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

  feedForward(input) {
    if (!input instanceof Array) {
      throw new Error('feedForward input must be an array');
    }

    // Get inputs vector
    const inputs = Matrix.fromArray(input);

    // Multiply input weights by inputs and add hidden node bias
    let hiddenOutputs = Matrix.multiply(this.weightsIH, inputs);
    hiddenOutputs = Matrix.add(hiddenOutputs, this.biasH);

    // Multiply hidden output weights by hidden outputs and add output bias
    let output = Matrix.multiply(this.weightsHO, hiddenOutputs);
    output = Matrix.add(output, this.biasO);

    // call activatoin function on outputs
    output = Matrix.map(
      output,
      ([i, j], result) =>
        (output.values[i][j] = this.activation(output.values[i][j]))
    );

    // send back outputs as array
    return Matrix.toArray(output);
  }
}

export default NeuralNetwork;
