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
import './styles.scss';

import React, { Component } from 'react';
import classnames from 'classnames';
import { injectIntl, FormattedMessage } from 'react-intl';

import { Checkbox, Dropdown, Toggle, Button, Search } from 'carbon-components-react';

import {
  Grid16, List16, Unknown16, Search16, Restart16, Edit16,
  TrashCan16, Close16, Export16, NewTab16, Keyboard16, Model16,
  Help16, View16, ViewOff16, CopyFile16, SubtractAlt20
} from '@carbon/icons-react';
import augmentSvg from 'images/AugmentIcon.svg';
import autoLabelIcon from 'images/auto-label_16.svg';

/* The Toolbar component takes in a prop of 'includeWidgets', that has a list
 * of widgets to be added to the toolbar. Example:
 * [
 *  { type: 'search', onChange: this.onSearch, size: "sm" },
 *  { type: 'buttons', includeButtons: [
 *    { type: 'duplicate', disabled: true, onClick: this.onDuplicate },
 *    { type: 'deleteBtn', disabled: false, onClick: this.onDelete }
 *  ]},
 *  { type: 'view', defaultSelectedView: view }
 * ]
 * The only required value in each widget object is 'type' which is used to load
 * the default data for the supported widget. The currently supported widgets
 * can be seen in the widgetMap below.
 *
 * NOTE: The order is maintained when passing in the includeWidgets array,
 * so widgets should be listed in the order they will appear in the toolbar
 */

const widgetMap = {
  search: {
    labelId: "toolbar.search",
    placeHolderId: "toolbar.search",
    disabled: false,
    size: "sm",
    iconTag: Search16
  },
  view: {
    descListId: "toolbar.view.descList",
    descCardId: "toolbar.view.descCard",
    disabled: false
  },
  select: {
    labelId: "toolbar.select",
    disabled: false
  },
  sort: {
    labelId: "toolbar.sort",
    placeholderText: "toolbar.select",
    items: [],
    disabled: false
  },
  advancedView: {
    labelId: "toolbar.advancedView",
    toggled: false,
    disabled: false
  },
  buttons: {
    duplicate: {
      labelId: "btn.duplicate",
      iconTag: CopyFile16,
      disabled: false,
      hideLabel: true
    },
    refresh: {
      labelId: "toolbar.refresh",
      iconTag: Restart16,
      disabled: false,
      hideLabel: true
    },
    edit: {
      labelId: "btn.edit",
      iconTag: Edit16,
      disabled: false,
      hideLabel: true
    },
    deleteBtn: {
      labelId: "btn.delete",
      iconTag: TrashCan16,
      disabled: false,
      hideLabel: true
    },
    close: {
      labelId: "btn.close",
      iconTag: Close16,
      disabled: false,
      hideLabel: true
    },
    remove: {
      labelId: "btn.remove",
      iconTag: SubtractAlt20,
      disabled: false,
      hideLabel: true
    },
    augmentData: {
      labelId: "toolbar.augment",
      svgIcon: augmentSvg,
      disabled: false,
      className: "toolbar__action-icon--augment"
    },
    autoLabel: {
      labelId: "toolbar.autoLabel",
      svgIcon: autoLabelIcon,
      disabled: false,
      className: "toolbar__action-icon--autoLabel"
    },
    export: {
      labelId: "toolbar.export",
      iconTag: Export16,
      disabled: false
    },
    autoCapture: {
      labelId: "btn.autoCapture",
      iconTag: NewTab16,
      disabled: false
    },
    keyboardShortcuts: {
      labelId: "toolbar.keyboardShortcuts",
      iconTag: Keyboard16,
      disabled: false,
      className: "toolbar__action-icon--keyboard"
    },
    howTo: {
      labelId: "toolbar.howTo",
      iconTag: Help16,
      disabled: false
    },
    show: {
      labelId: "toolbar.show",
      iconTag: View16,
      disabled: false
    },
    hide: {
      labelId: "toolbar.hide",
      iconTag: ViewOff16,
      disabled: false
    },
    train: {
      labelId: "toolbar.train",
      id: "train",
      buttonType: "primary",
      iconTag: Model16,
      disabled: false
    },
    primary: {
      labelId: "toolbar.train",
      id: "train",
      buttonType: "primary",
      disabled: false
    },
    secondary: {
      labelId: "toolbar.assignCategory",
      id: "assignCategory",
      buttonType: "secondary",
      disabled: false
    }
  }
};

class Toolbar extends Component {
  state = {};
  searchInput = React.createRef();

  static getDerivedStateFromProps(nextProps, prevState) {
    let updatedState = {};

    // handle when the user chooses an indeterminate state by selecting / unselecting items.
    // the select checkbox shows this indeterminate state with a "-"
    if (!prevState.selectCheckbox) {
      updatedState.selectCheckbox = nextProps.selectCheckbox || "none";
    } else if (nextProps.selectCheckbox && prevState.selectCheckbox !== nextProps.selectCheckbox) {
      updatedState.selectCheckbox = nextProps.selectCheckbox;
    }

    // normally, the View value is set on startup and changed by the user, but there
    // might be a use case where the parent wants to manually set the View value after startup
    // so handle any possible state updates here
    if (!prevState.selectedView) {
      updatedState.selectedView = nextProps.defaultSelectedView || "card";
    } else if (nextProps.defaultSelectedView && prevState.selectedView !== nextProps.defaultSelectedView) {
      updatedState.selectedView = nextProps.defaultSelectedView;
    }

    // handle when the user inputs a search text and then leaves the current page.
    // when the user returns to the page, the page will still retain the filtered items based on the search text,
    // so if there is any text in the search box, we should leave the search box open.
    if (prevState.showSearch === undefined) {
      updatedState.showSearch = nextProps.showSearch || false;
    }

    return (updatedState.selectedView || updatedState.selectCheckbox || updatedState.showSearch) ? updatedState : null;
  }

  createWidget(type, widget) {
    // used to create the first-level object list in widgetMap;
    // the buttons sub-types are created in a separate createButton method

    if (type.startsWith("buttons")) {
      // create each button individually and wrap it in a container so they can be separated from other widgets
      return (
        <div key={type} className="action-container">
          {widget.includeButtons.map((button) => button && this.createButton(button))}
        </div>
      );
    } else if (type === "search") {
      return this.createSearchBar(widget);
    } else if (type === "view") {
      return this.createViewChanger(widget);
    } else if (type === "select") {
      return this.createSelectCheckbox(widget);
    } else if (type === "sort") {
      return this.createSortDropdown(widget);
    } else if (type === "advancedView") {
      return this.createAdvancedViewToggle(widget);
    }
    // if none of the if/else if statements were reached, this button isn't supported
  }

  createButton(btnData) {
    if (btnData.type === "html") {
      // used to include a node that doesn't fit the button conventions
      return (
        <div className={btnData.className || ""} key={btnData.key || "html"}>
          {btnData.child}
        </div>
      );
    }

    // merge button default values with passed in values
    const btn = {
      ...widgetMap.buttons[btnData.type],
      ...btnData,
      onClick: () => {
        this.defaultOnClick(btnData.type);
        if (btnData.onClick) {
          btnData.onClick();
        }
      }
    };
    const iconClassNames = classnames({
      //'bx--toolbar-action__icon': true,
      'toolbar__action-icon--blue': true,
      [btn.className]: btn.className
    });


    if (btn.buttonType) {
      // Using a primary or secondary button 
      // Usually these do not contain an SVG or Carbon icon, but if there is one we will render it
      return (
        <Button
          key={btn.type + "-" + btn.id}
          id={`${btn.type}-btn`}
          className="toolbar__action-container toolbar__btn-spacing"
          kind={btn.buttonType}
          size="field"
          disabled={btn.disabled}
          {...(btn.iconTag && { renderIcon: btn.iconTag })}
          onClick={btn.onClick}>
          <FormattedMessage id={btn.labelId} className="toolbar__action-label" />
        </Button>
      );
    } else if (btn.svgIcon) {
      // Using a locally stored SVG for the button
      return (
        <Button
          key={btn.type}
          id={`${btn.type}-btn`}
          className="toolbar__action-container"
          kind="ghost"
          size="field"
          renderIcon={React.forwardRef((props, ref) => (
            <svg ref={ref} id={btn.type} className={iconClassNames} viewBox={btn.svgIcon.viewBox}>
              <use xlinkHref={`#${btn.svgIcon.id}`}></use>
            </svg>
          ))}
          iconDescription={btn.type}
          disabled={btn.disabled}
          onClick={btn.onClick}>
          {/*<svg id={btn.type} className={iconClassNames} viewBox={btn.svgIcon.viewBox}>
            <use xlinkHref={`#${btn.svgIcon.id}`}></use>
          </svg>*/}
          {btn.hideLabel ? "" : <FormattedMessage id={btn.labelId} className="toolbar__action-label" /> }
        </Button>
      );
    } else {
      // Carbon icon where we aren't using a locally stored SVG
      const IconTag = btn.iconTag || Unknown16;

      //iconClassNames
      return (
        <Button
          key={btn.type}
          id={`${btn.type}-btn`}
          className="toolbar__action-container"
          kind="ghost"
          size="field"
          hasIconOnly={!!btn.hideLabel}
          tooltipAlignment="center"
          tooltipPosition="bottom"
          renderIcon={IconTag}
          iconDescription={btn.description || this.props.intl.formatMessage({ id: btn.labelId })}
          disabled={btn.disabled}
          onClick={btn.onClick}>
          {btn.hideLabel ? "" :  <FormattedMessage id={btn.labelId} className="toolbar__action-label" /> }
        </Button>
      );
    }
  }

  createSearchBar(searchData) {
    // merge search bar default values with passed in values
    const search = {
      ...widgetMap.search,
      onChange: () => this.defaultOnChange('search'),
      ...searchData,
      onClick: () => {
        this.defaultOnClick(searchData.type);
        this.searchAction();
      }
    };

    return (
      <div className="search-container" key={search.type}>
        {this.state.showSearch &&
          <div className="bx--toolbar-search-container">
            <Search
              ref={this.searchInput}
              labelText={this.props.intl.formatMessage({ id: search.labelId })}
              placeHolderText={this.props.intl.formatMessage({ id: search.placeHolderId })}
              onChange={search.onChange}
              size={search.size}
              value={search.defaultValue}
              closeButtonLabelText={this.props.intl.formatMessage({ id: "toolbar.search.iconDescription" })}
            />
          </div>
        }
         <Button
          key={search.type}
          id={`${search.type}-btn`}
          className="toolbar__action-container"
          kind="ghost"
          size="small"
          renderIcon={search.iconTag}
          iconDescription={search.description || this.props.intl.formatMessage({ id: search.labelId })}
          disabled={search.disabled}
          onClick={search.onClick} />
      </div>
    );
  }

  createViewChanger(viewData) {
    // merge view changer default values with passed in values
    const selectedViewClasses = "toolbar__view-icon toolbar__view-icon--selected";
    const unselectedViewClasses = "toolbar__view-icon toolbar__view-icon--unselected";

    const view = {
      ...widgetMap.view,
      ...viewData,
      onClick: (type) => {
        this.viewAction(type);
        if (viewData.onClick) {
          viewData.onClick(type);
        }
      }
    };

    return (
      <div className="view-container" key={view.type}>
        <div className="toolbar__action-container">
          <div className="toolbar__view-text">
            <FormattedMessage id="toolbar.view" />
          </div>
          <Grid16
            tabIndex="0"
            onKeyPress={(e) => e.which === 13 && view.onClick("card")}
            onClick={() => view.onClick("card")}
            description={this.props.intl.formatMessage({ id: view.descCardId })}
            className={this.state.selectedView !== "list" ? selectedViewClasses : unselectedViewClasses}
          />
          <List16
            tabIndex="0"
            onKeyPress={(e) => e.which === 13 && view.onClick("list")}
            onClick={() => view.onClick("list")}
            description={this.props.intl.formatMessage({ id: view.descListId })}
            className={this.state.selectedView === "list" ? selectedViewClasses : unselectedViewClasses}
          />
        </div>
      </div>
    );
  }

  createSelectCheckbox(selectData) {
    // Check if this page has pagination
    const hasPagination = this.props.hasPagination || false; 

    // merge select checkbox default values with passed in values
    const select = {
      ...widgetMap.select,
      ...selectData,
      onChange: (value) => {
        this.selectAction(value);
        if (selectData.onChange) {
          selectData.onChange(value);
        }
      }
    };

    return (
      <div className="select-container" key={select.type}>
        <Checkbox
          className="toolbar__select-checkbox"
          id="selectCheckbox"
          labelText=""
          indeterminate={this.state.selectCheckbox === "current" || this.state.selectCheckbox === "indeterminate"}
          checked={this.state.selectCheckbox !== "none"}
          disabled={select.disabled}
          onChange={select.onChange} />
        <Dropdown
          label={this.props.intl.formatMessage({ id: select.labelId })}
          items={[
            ...([ { id: "all", label: this.props.intl.formatMessage({ id: "dropdown.select.all" })} ]),
            ...(hasPagination  ? [{ id: "current", label: this.props.intl.formatMessage({ id: "dropdown.select.current" })}] : []),
            ...([ { id: "none", label: this.props.intl.formatMessage({ id: "dropdown.select.none" })}])
          ]}
          selectedItem={{ id: "select", label: this.props.intl.formatMessage({ id: select.labelId }) }}
          id="selectDropdown"
          disabled={select.disabled}
          onChange={select.onChange} />
      </div>
    );
  }

  createSortDropdown(sortData) {
    // merge sort dropdown default values with passed in values
    const sort = {
      ...widgetMap.sort,
      onChange: () => this.defaultOnChange('sort'),
      ...sortData
    };

    return (
      <div className="sort-container" key={sort.type}>
        <label htmlFor={sort.type + "Dropdown"} className="toolbar__view-text">
          <FormattedMessage id={sort.labelId} />
        </label>
        <Dropdown
          className="toolbar__sort-dropdown"
          id={sort.type + "Dropdown"}
          label={this.props.intl.formatMessage({ id: sort.placeholderText })}
          items={sort.items}
          onChange={sort.onChange}
          {...(sort.initialSelectedItem ? { selectedItem: sort.initialSelectedItem } : { selectedItem: null })}
        />
      </div>
    );
  }

  createAdvancedViewToggle(advancedViewData) {
    // merge advanced view toggle default values with passed in values
    const advancedView = {
      ...widgetMap.advancedView,
      onChange: () => this.defaultOnChange('advanced view toggle'),
      ...advancedViewData
    };

    return (
      <div className="advanced-view-container" key={advancedView.type}>
        <label htmlFor={advancedView.type + "Toggle"} className="toolbar__advanced-view-text">
          <FormattedMessage id={advancedView.labelId} />
        </label>
        <Toggle
          className="toolbar__advanced-view-toggle"
          id={advancedView.type + "Toggle"}
          onToggle={advancedView.onChange}
          toggled={advancedView.toggled}
          disabled={advancedView.disabled}
        />
      </div>
    );
  }

  defaultOnClick(type) {
    // after a button is clicked, blur the button so the focus does not remain
    document.getElementById(`${type}-btn`).blur();
  }

  defaultOnChange(type) {
    console.log("No onChange function specified for toolbar button " + type);
  }

  viewAction(type) {
    // type will either be "card" or "list"
    this.setState({
      selectedView: type
    });
  }

  searchAction() {
    // the search box will be shown or hidden
    // state change occurs when the search button is clicked
    this.setState(prevState => ({
      showSearch: !prevState.showSearch
    }), () => {
      if (this.state.showSearch && this.searchInput) {
        this.searchInput.current.input.focus();
      }
    });
  }

  selectAction(value) {
    // type will either be "all", "none", or "indeterminate" or "current"
    if (value && value.selectedItem) {
      this.setState({
        selectCheckbox: value.selectedItem.id
      });
    } else {
      if (value) {
        this.setState({
          selectCheckbox: "all"
        });
      } else {
        this.setState({
          selectCheckbox: "none"
        });
      }
    }
  }

  render() {
    let widgets = {};
    let mainWidgetList = [];
    let secondaryWidgetList = [];
   
    if (this.props.includeWidgets) {
      this.props.includeWidgets.forEach((widget) => {
        widgets[widget.type] = widget;
        if (widget.secondary) {
          secondaryWidgetList.push(widget.type);
        } else {
          mainWidgetList.push(widget.type);
        }
      });
    }

    return (
      <div className={this.props.toolbarClass || ""}>
        <section className="bx--toolbar">
          {mainWidgetList.map((type) => this.createWidget(type, widgets[type]))}
        </section>
        {secondaryWidgetList.length > 0 &&
          <section className="bx--toolbar secondary-toolbar">
            {secondaryWidgetList.map((type) => this.createWidget(type, widgets[type]))}
          </section>
        }
      </div>
    );
  }
}

export default injectIntl(Toolbar);
