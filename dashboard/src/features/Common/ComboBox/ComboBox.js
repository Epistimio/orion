/* IBM_SOURCE_PROLOG_BEGIN_TAG                                            */
/* *****************************************************************      */
/*                                                                        */
/* IBM Confidential                                                       */
/* OCO Source Materials                                                   */
/*                                                                        */
/* (C) Copyright IBM Corp. 2018, 2020                                     */
/*                                                                        */
/* The source code for this program is not published or otherwise         */
/* divested of its trade secrets, irrespective of what has been           */
/* deposited with the U.S. Copyright Office.                              */
/*                                                                        */
/* *****************************************************************      */
/* IBM_SOURCE_PROLOG_END_TAG                                              */
import cx from 'classnames';
import Downshift from 'downshift';
import PropTypes from 'prop-types';
import React from 'react';
import ListBox, { PropTypes as ListBoxPropTypes } from 'carbon-components-react/lib/components/ListBox';

// Define constants to use in the ComboBox
const defaultItemToString = item => item && item.label;

const getInputValue = (props, state) => {
  if (props.initialSelectedItem) {
    return props.itemToString(props.initialSelectedItem);
  }
  return state.inputValue || '';
};


class ComboBox extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      inputValue: getInputValue(props, {}),
    };

    this.filterItems = this.filterItems.bind(this);
    this.handleOnInputKeyDown = this.handleOnInputKeyDown.bind(this);
  }

  UNSAFE_componentWillReceiveProps(nextProps) {
    this.setState(state => ({
      inputValue: getInputValue(nextProps, state),
    }));
  }

  filterItems(items, itemToString, inputValue) {
    return items.filter(
      item => !inputValue || itemToString(item).toLowerCase().includes(inputValue.toLowerCase())
    );
  }

  handleOnChange(selectedItem) {
    if (this.props.onChange) {
      this.props.onChange({ selectedItem });
    }
  }

  handleOnInputKeyDown(event) {
    event.stopPropagation();
  }

  handleOnInputValueChange(inputValue) {
    if (this.props.onInputValueChange) {
      this.props.onInputValueChange({ inputValue });
    }
    this.setState(() => ({
      inputValue: inputValue,
    }));
  }

  render() {
    const {
      className: containerClassName,
      disabled,
      id,
      items,
      itemToString,
      placeholder,
      initialSelectedItem,
      ariaLabel,
    } = this.props;
    const className = cx('bx--combo-box', containerClassName);
    return (
      <Downshift
        onChange={(selectedItem) => this.handleOnChange(selectedItem)}
        onInputValueChange={(inputValue) => this.handleOnInputValueChange(inputValue)}
        inputValue={this.state.inputValue || ''}
        itemToString={itemToString}
        onOuterClick={this.props.filterOnly ? (() => {}) : ((stateAndHelpers) => {
          this.setState({inputValue: stateAndHelpers.inputValue});
          this.handleOnInputValueChange(stateAndHelpers.inputValue);
        })}
        defaultSelectedItem={initialSelectedItem}>
        {({
          getToggleButtonProps,
          getInputProps,
          getItemProps,
          getRootProps,
          isOpen,
          inputValue,
          selectedItem,
          highlightedIndex,
          clearSelection,
        }) => (
            <ListBox
              className={className}
              disabled={disabled}
              {...getRootProps({ refKey: 'innerRef' })}>
              <ListBox.Field id="combo-box__field" {...getToggleButtonProps({ disabled })}>
                <input
                  className="bx--text-input"
                  aria-label={ariaLabel}
                  {...getInputProps({
                    disabled,
                    id,
                    placeholder,
                    onKeyDown: this.handleOnInputKeyDown,
                    onBlur: (e) => !this.props.filterOnly && e.preventDefault()
                  })}
                />
                {inputValue &&
                  isOpen && <ListBox.Selection clearSelection={clearSelection} />}
                <ListBox.MenuIcon isOpen={isOpen} id="combo-box__menu-icon" />
              </ListBox.Field>
              {isOpen && (
                <ListBox.Menu id="combo-box__menu">
                  {this.filterItems(items, itemToString, inputValue).map(
                    (item, index) => (
                      <ListBox.MenuItem
                        key={item.id}
                        id="combo-box__menu-item"
                        isActive={selectedItem === item}
                        isHighlighted={highlightedIndex === index}
                        {...getItemProps({ item, index })}>
                        {itemToString(item)}
                      </ListBox.MenuItem>
                    )
                  )}
                </ListBox.Menu>
              )}
            </ListBox>
          )}
      </Downshift>
    );
  }
}

ComboBox.propTypes = {
  /**
   * An optional className to add to the container node
   */
  className: PropTypes.string,

  /**
   * Specify if the control should be disabled, or not
   */
  disabled: PropTypes.bool,

  /**
   * Specify a custom `id` for the input
   */
  id: PropTypes.string,

  /**
   * Allow users to pass in arbitrary items from their collection that are
   * pre-selected
   */
  initialSelectedItem: PropTypes.object,

  /**
   * We try to stay as generic as possible here to allow individuals to pass
   * in a collection of whatever kind of data structure they prefer
   */
  items: PropTypes.array,

  /**
   * Helper function passed to downshift that allows the library to render a
   * given item to a string label. By default, it extracts the `label` field
   * from a given item to serve as the item label in the list
   */
  itemToString: PropTypes.func,

  /**
   * `onChange` is a utility for this controlled component to communicate to a
   * consuming component what kind of internal state changes are occuring
   */
  onChange: PropTypes.func,

  /**
   * Used to provide a placeholder text node before a user enters any input.
   * This is only present if the control has no items selected
   */
  placeholder: PropTypes.string.isRequired,

  /**
   * Currently supports either the default type, or an inline variant
   */
  type: ListBoxPropTypes.ListBoxType,

  /**
   * `onInputValueChange` is a utility for this component to communicate
   * the changing input value
   */
  onInputValueChange: PropTypes.func,

  /**
   * If set to `true`, the ComboBox shouldn't be treated like a textbox,
   * only as a type-ahead to filter out dropdown options
   */
  filterOnly: PropTypes.bool
};


// The default prop values of the component
ComboBox.defaultProps = {
  disabled: false,
  itemToString: defaultItemToString,
  type: 'default',
  ariaLabel: 'ListBox input field',
  filterOnly: false,
};

export default ComboBox;
