/* eslint-disable max-len */
/* eslint-disable no-restricted-globals */
/* IBM_SOURCE_PROLOG_BEGIN_TAG                                            */
/* *****************************************************************      */
/*                                                                        */
/* IBM Confidential                                                       */
/* OCO Source Materials                                                   */
/*                                                                        */
/* (C) Copyright IBM Corp. 2020                                           */
/*                                                                        */
/* The source code for this program is not published or otherwise         */
/* divested of its trade secrets, irrespective of what has been           */
/* deposited with the U.S. Copyright Office.                              */
/*                                                                        */
/* *****************************************************************      */
/* IBM_SOURCE_PROLOG_END_TAG                                              */

/* Validation functions to be used in form Field validate lists, such as
 * validate={(validate.required, validate.number)}
 * Will return undefined if the value is valid, and an intl message ID if the value is invalid.
 * The component is in charge of showing the corresponding error message using intl.
 * Ex. invalidText={meta.error && this.props.intl.formatMessage({ id: meta.error })}
 *  or invalidText={meta.error && this.props.intl.formatMessage(meta.error)} for max/min errors
 */

// returns an error message ID if this value is empty
export const required = value => {
  return (value || value === 0) && (typeof value === 'string' ? value.trim() !== '' : true) ? undefined : 'global.requiredValue';
};

// returns an error message ID if this value is not a number
export const number = value => {
  return (value || value === 0) && isNaN(Number(value)) ? 'global.numberValue' : undefined;
};

// returns an error message ID if this value is not an integer
export const positiveNum = value => {
  return (value || value === 0) && (isNaN(Number(value)) || value < 0) ? 'global.positiveValue' : undefined;
};

// returns an error message ID if this value is not a whole number (non-decimal except for .0's)
export const wholeNum = value => {
  if (!value && value !== 0) {
    return 'global.wholeNumValue';
  }
  return (value || value === 0) && (isNaN(Number(value)) || !(value + '').match(/^[0-9]+\.?[0]*$/g))
    ? 'global.wholeNumValue'
    : undefined;
};

export const minValue = (min, isExclusive) => value => {
  if (isExclusive) {
    // the value must be greater than the min value
    return (value || value === 0) && (min || min === 0) && value <= min ? { id: 'global.minValue', values: { min } } : undefined;
  }
  // the min value is a possible value
  return (value || value === 0) && (min || min === 0) && value < min ? { id: 'global.minValueInclusive', values: { min } } : undefined;
};

export const maxValue = (max, isExclusive) => value => {
  if (isExclusive) {
    // the value must be less than the max value
    return (value || value === 0) && (max || max === 0) && value >= max ? { id: 'global.maxValue', values: { max } } : undefined;
  }
  // the max value is a possible value
  return (value || value === 0) && (max || max === 0) && value > max ? { id: 'global.maxValueInclusive', values: { max } } : undefined;
};

// returns an error message ID if this value is not a time
export const time = value => {
  const parts = (value || '').split(':');
  let isValid = false;
  if (parts.length && parts.length <= 3) {
    // check that each part is a number and only the last have a decimal
    isValid = parts.every((num, i) => {
      if (i === parts.length - 1) {
        const subparts = num.split('.');
        if (subparts.length > 0 && subparts.length <= 2) {
          return subparts.every((num2, i2) => (i2 === subparts.length - 1 ? (num2 + '').match(/^\d{1,3}$/g) : (num2 + '').match(/^\d*$/g)));
        }
        return false;
      }
      return (num + '').match(/^\d+$/g);
    });
  } else {
    // check for ms
    isValid = (value + '').match(/^\d+$/g);
  }

  // TODO give a better error message depending on what part is incorrect
  return isValid ? undefined : 'global.timeValue';
};

/* Used when you need to pass specific non-field value information to a validate function,
 * such as min vlues and max values.
 * Ex. on Field: validate={composeValidators(validate.required, validate.number, validate.minValue(18))}
 * If no extra information is needed, you can ignore this function and just pass the validate functions directly.
 */
export const composeValidators = (...validators) => value => validators.reduce((error, validator) => error || validator(value), undefined);
