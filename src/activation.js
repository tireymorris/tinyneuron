const dsigmoid = x => x * (1 - x);
const sigmoid = x => 1 / (1 + Math.exp(-x));

const tanh = x => Math.tanh(x);
const dtanh = x => 1 - x * x;

const sign = x => (x >= 0 ? 1 : -1);

export { dsigmoid, sigmoid, sign, tanh, dtanh };
