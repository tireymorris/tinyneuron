const sigmoid = input => 1 / (1 + Math.exp(-input));

const sign = input => (input >= 0 ? 1 : -1);

export { sigmoid, sign };
