const dsigmoid = input => sigmoid(input) * 1 - sigmoid(x);

const sigmoid = input => 1 / (1 + Math.exp(-input));

const sign = input => (input >= 0 ? 1 : -1);

export { dsigmoid, sigmoid, sign };
