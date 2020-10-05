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
import PropTypes from 'prop-types';

import { DataTable as BaseDataTable } from 'carbon-components-react';
import TableSelectRow from './TableSelectRow';

import { sortStates, initialSortState, getNextSortDirection } from 'utils/sorting';

/*
 * NOTE: This DataTable should only be used if the data table rows are being controlled completely
 * by another source (ex. Redux). This data table doesn't provide sorting, toolbar, or filtering
 * capabilities. Instead, those features need to be handled somewhere else, and the `rows` property
 * should be updated with the results. If you don't need to modify the row values on your own,
 * Carbons's DataTable component should be used:
 * https://github.com/IBM/carbon-components-react/blob/1eb620d9d0cbb155aa86aa6b0ae1144f449a8e2a/src/components/DataTable/README.md
 */
class ControlledDataTable extends Component {
  // Used to get a table prefix for elements that require an `id` attribute that is unique.
  getTablePrefix = () => `data-table-${this.props.tableId}`;

  // Handler for transitioning to the next sort state of the table
  handleSortBy = headerKey => () => {
    const nextSortDirection = getNextSortDirection(
      headerKey,
      this.props.sortHeaderKey,
      this.props.sortDirection
    );
    this.props.onSortBy(nextSortDirection === "NONE" ? "NONE" : (`${headerKey}--${nextSortDirection}`))
  };

  render() {
    const { headers, selectedRows, rowIds, rowsById } = this.props;

    // TODO we currently don't need to handle expanded rows, but if it needs to be added in the future,
    // a TableExpandHeader, TableExpandRow, and TableExpandedRow need to be included in the render function

    return (
      <BaseDataTable.TableContainer className="controlled-data-table__container">
        <BaseDataTable.Table useZebraStyles isSortable={true}>
          <BaseDataTable.TableHead>
            <BaseDataTable.TableRow>
              {this.props.isSelectable && <BaseDataTable.TableHeader />}
              {headers.map(header => (
                <BaseDataTable.TableHeader
                  key={header.key}
                  sortDirection={this.props.sortDirection}
                  isSortable={true}
                  isSortHeader={this.props.sortHeaderKey === header.key}
                  onClick={this.handleSortBy(header.key)}>
                  {header.header}
                </BaseDataTable.TableHeader>
              ))}
            </BaseDataTable.TableRow>
          </BaseDataTable.TableHead>
          <BaseDataTable.TableBody>
            {rowIds.map(rowId => (
              <BaseDataTable.TableRow key={rowId}>
                {this.props.isSelectable &&
                  <TableSelectRow
                    id={`${this.getTablePrefix()}__select-row-${rowId}`}
                    key={`select-row-${rowId}`}
                    name={`select-row-${rowId}`}
                    resourceId={rowId}
                    checked={selectedRows.includes(rowId)}
                    showThumbnails={this.props.showThumbnails}
                    bgImgPath={rowsById[rowId].thumbnailPath}
                    onChange={this.props.checkboxOnChange}
                    onClick={this.props.imageCardOnClick}
                  />
                }
                {rowsById[rowId].cells.map((cell) => (
                  <BaseDataTable.TableCell
                    key={cell.id}
                    onClick={() => this.props.onRowClick(rowId)}>
                    {cell.value}
                  </BaseDataTable.TableCell>
                ))}
              </BaseDataTable.TableRow>
            ))}
          </BaseDataTable.TableBody>
        </BaseDataTable.Table>
      </BaseDataTable.TableContainer>
    )
  }
}

ControlledDataTable.propTypes = {
  tableId: PropTypes.string.isRequired,
  /**
   * The `headers` prop represents the order in which the headers should
   * appear in the table. We expect an array of objects to be passed in, where
   * `key` is the name of the key in a row object, and `header` is the name of
   * the header.
   */
  headers: PropTypes.arrayOf(
    PropTypes.shape({
      key: PropTypes.string.isRequired, // name of field on row object itself
      header: PropTypes.object.isRequired, // name (as a FormattedMessage object) you want rendered in the table header
    })
  ).isRequired,
  /**
   * The `rowIds` prop represents the order in which the rows should appear in
   * the table. It should be an array of row IDs that are present in the rowsById prop.
   */
  rowIds: PropTypes.arrayOf(PropTypes.string).isRequired,
  /**
   * The `rowsById` prop is where you provide a list of objects that contain the
   * information to be rendered in the row. It should be an object with key names
   * matching the row IDs. The value of each key should contain an object that contains
   * a thumbnailPath (if showThumbnails is true) and a cells property. The cells
   * property should contain an array of objects with a cell id and cell value.
   */
  rowsById: PropTypes.objectOf(PropTypes.object).isRequired,
  // The `selectedRows` prop contains a list of row IDs that should be marked "selected"
  selectedRows: PropTypes.arrayOf(PropTypes.string),
  // The `onRowClick` prop is used to handle row clicks outside of the TableSelectRow cell
  onRowClick: PropTypes.func,
  showThumbnails: PropTypes.bool,
  // The `isSelectable` prop is used to handle tables that are not selectable
  isSelectable: PropTypes.bool,
  sortHeaderKey: PropTypes.string,
  sortDirection: PropTypes.oneOf(Object.values(sortStates)),
  onSortBy: PropTypes.func.isRequired,
  // If image cards are being used to show thumbnails,
  // imageCardOnClick is required to handle the card/checkbox click events
  imageCardOnClick: function (props, propName) {
    if (props.showThumbnails && (!props[propName] || typeof props[propName] !== "function")) {
      return new Error('A imageCardOnClick function is required when displaying thumbnails.');
    }
  },
  // checkboxOnChange is only required if image cards are NOT being used.
  // If will trigger when the row select checkbox is toggled.
  checkboxOnChange: function (props, propName) {
    if (!props.showThumbnails && (!props[propName] || typeof props[propName] !== "function")) {
      return new Error('An checkboxOnChange function is required for the row selection checkbox.');
    }
  },
};

ControlledDataTable.defaultProps = {
  headers: [],
  rowIds: [],
  rowsById: {},
  selectedRows: [],
  showThumbnails: false,
  isSelectable: true,
  sortDirection: initialSortState,
  sortHeaderKey: null,
  onSortBy: (() => { }),
  onRowClick: (() => { }),
  imageCardOnClick: (() => { }),
  checkboxOnChange: (() => { })
};

export default ControlledDataTable;
