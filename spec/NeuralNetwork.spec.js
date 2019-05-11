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
    it('can solve XOR', () => {
      const network = new NeuralNetwork({
        inputNodes: 2,
        hiddenNodes: 10, // probably overkill
        outputNodes: 1,
        learningRate: 0.25
      });
      const inputs = [[1, 0], [0, 1], [1, 1], [0, 0]];
      const targets = [1, 1, 0, 0];

      // probably overkill
      for (let i = 0; i < 100000; i++) {
        const idx = Math.floor(Math.random() * 4);
        network.train(inputs[idx], [targets[idx]]);
      }

      console.log('[1, 0] => ', network.feedForward([1, 0]));
      console.log('[0, 1] => ', network.feedForward([0, 1]));
      console.log('[0, 0] => ', network.feedForward([0, 0]));
      console.log('[1, 1] => ', network.feedForward([1, 1]));

      expect(network.feedForward([1, 0])[0]).toBeLessThanOrEqual(1);
      expect(network.feedForward([1, 0])[0]).toBeGreaterThanOrEqual(0.95);

      expect(network.feedForward([0, 1])[0]).toBeLessThanOrEqual(1);
      expect(network.feedForward([0, 1])[0]).toBeGreaterThanOrEqual(0.95);

      expect(network.feedForward([0, 0])[0]).toBeLessThanOrEqual(0.05);
      expect(network.feedForward([0, 0])[0]).toBeGreaterThanOrEqual(0.0);

      expect(network.feedForward([1, 1])[0]).toBeLessThanOrEqual(0.05);
      expect(network.feedForward([1, 1])[0]).toBeGreaterThanOrEqual(0.0);
    });
  });
});
