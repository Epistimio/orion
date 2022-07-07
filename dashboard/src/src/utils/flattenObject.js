function _flattenObject(inObj, outObj, prefix = '') {
  if (prefix.length) {
    prefix += '.';
  }
  for (let key of Object.keys(inObj)) {
    const value = inObj[key];
    if (value.constructor === Object) {
      _flattenObject(value, outObj, prefix + key);
    } else {
      outObj[prefix + key] = value;
    }
  }
}

export function flattenObject(obj, prefix = '') {
  const output = {};
  _flattenObject(obj, output, prefix);
  return output;
}
