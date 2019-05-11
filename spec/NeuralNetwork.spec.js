import NeuralNetwork from '../src/NeuralNetwork';

describe('NeuralNetwork', () => {
  test('can create a neural network', () => {
    const network = new NeuralNetwork({
      inputNodes: 1,
      hiddenNodes: 1,
      outputNodes: 1
    });

    expect(network).toBeTruthy();
  });

  describe('feedForward', () => {
    test('', () => {
      const network = new NeuralNetwork({
        inputNodes: 2,
        hiddenNodes: 2,
        outputNodes: 1
      });
      const input = [1, 0];
      const output = network.feedForward(input);
    });
  });
});
