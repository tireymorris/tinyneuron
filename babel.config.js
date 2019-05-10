module.exports = function(api) {
  api.cache(false);

  const presets = ['@babel/env'];

  return {
    presets
  };
};
