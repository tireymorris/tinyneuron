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
    test('can feed data forward through the network', () => {
      const network = new NeuralNetwork({
        inputNodes: 2,
        hiddenNodes: 2,
        outputNodes: 1
      });
      const inputs = [1, 0];
      const output = network.feedForward(inputs);
      expect(output[0]).toBeGreaterThanOrEqual(0);
      expect(output[0]).toBeLessThanOrEqual(1);
    });
  });

  describe('train', () => {
    test('can solve XOR', () => {
      const network = new NeuralNetwork({
        inputNodes: 2,
        hiddenNodes: 5,
        outputNodes: 1,
        learningRate: 0.25
      });
      const inputs = [[1, 0], [0, 1], [1, 1], [0, 0]];
      const targets = [1, 1, 0, 0];

      for (let i = 0; i < 50000; i++) {
        const idx = Math.floor(Math.random() * 4);
        network.train(inputs[idx], [targets[idx]]);
      }

      expect(network.feedForward([1, 0])[0]).toBeLessThanOrEqual(1);
      expect(network.feedForward([1, 0])[0]).toBeGreaterThanOrEqual(0.95);

      expect(network.feedForward([0, 1])[0]).toBeLessThanOrEqual(1);
      expect(network.feedForward([0, 1])[0]).toBeGreaterThanOrEqual(0.95);

      expect(network.feedForward([0, 0])[0]).toBeLessThanOrEqual(0.05);
      expect(network.feedForward([0, 0])[0]).toBeGreaterThanOrEqual(0.0);

      expect(network.feedForward([1, 1])[0]).toBeLessThanOrEqual(0.05);
      expect(network.feedForward([1, 1])[0]).toBeGreaterThanOrEqual(0.0);
    });

    test('can learn sin for small inputs', () => {
      const network = new NeuralNetwork({
        inputNodes: 1,
        hiddenNodes: 16,
        outputNodes: 1,
        learningRate: 0.25
      });

      for (let i = 0; i < 100000; i++) {
        const input = Math.floor(Math.random() * 10);
        network.train([input], [Math.sin(input)]);
      }

      expect(network.feedForward([0])[0]).toBeLessThanOrEqual(0.04);
      expect(network.feedForward([0])[0]).toBeGreaterThanOrEqual(-0.4);

      expect(network.feedForward([1])[0]).toBeLessThanOrEqual(0.88);
      expect(network.feedForward([1])[0]).toBeGreaterThanOrEqual(0.76);
    });
  });
});
