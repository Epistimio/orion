import { ArrowDown20, ArrowsVertical20, ArrowUp20 } from '@carbon/icons-react';
import React from 'react';
import {
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';
import {
  Column,
  Grid,
  MultiSelect,
  Pagination,
  Row,
} from 'carbon-components-react';

const sortingIcons = {
  asc: <ArrowUp20 className="bx--table-sort__icon" />,
  desc: <ArrowDown20 className="bx--table-sort__icon" />,
};

export function FeaturedTable({ columns, data, experiment }) {
  const [sorting, setSorting] = React.useState([]);
  const [columnVisibility, setColumnVisibility] = React.useState({});
  const [{ pageIndex, pageSize }, setPagination] = React.useState({
    pageIndex: 0,
    pageSize: 10,
  });
  const pagination = React.useMemo(() => ({ pageIndex, pageSize }), [
    pageIndex,
    pageSize,
  ]);
  const pageCount =
    Math.round(data.length / pageSize) + (data.length % pageSize);
  const table = useReactTable({
    columns,
    data,
    pageCount,
    state: { sorting, columnVisibility, pagination },
    getCoreRowModel: getCoreRowModel(),
    onSortingChange: setSorting,
    getSortedRowModel: getSortedRowModel(),
    onColumnVisibilityChange: setColumnVisibility,
    // getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    onPaginationChange: setPagination,
    // manualPagination: true
  });
  const selectableColumns = table.getAllLeafColumns().map((col, index) => ({
    id: col.id,
    label: col.columnDef.header,
  }));
  const columnVisibilitySetter = selectedColumns => {
    const colSet = new Set(selectedColumns.selectedItems.map(item => item.id));
    const def = {};
    table
      .getAllLeafColumns()
      .forEach(column => (def[column.id] = colSet.has(column.id)));
    table.setColumnVisibility(def);
  };
  const setCarbonPagination = ({ page, pageSize }) => {
    // table.setPageIndex(page - 1);
    // table.setPageSize(pageSize);
    setPagination({ pageIndex: page - 1, pageSize: pageSize });
  };
  return (
    <div className="bx--data-table-container">
      <div className="bx--data-table-header">
        <Grid>
          <Row>
            <Column>
              <h4 className="bx--data-table-header__title">
                Experiment Trials for "{experiment}"
              </h4>
              <p className="bx--data-table-header__description">
                {data.length} trial(s) for experiment "{experiment}"
              </p>
            </Column>
            <Column>
              <Pagination
                page={pageIndex + 1}
                pageSize={pageSize}
                pageSizes={[5, 10, 20, 50, 100]}
                totalItems={data.length}
                onChange={setCarbonPagination}
              />
            </Column>
            <Column>
              <MultiSelect
                id="multiselect-columns"
                label="Columns to display"
                items={selectableColumns}
                initialSelectedItems={selectableColumns}
                onChange={columnVisibilitySetter}
                sortItems={items => items}
              />
            </Column>
          </Row>
        </Grid>
      </div>
      <div className="bx--data-table-content">
        <table className="bx--data-table bx--data-table--normal bx--data-table--no-border">
          <thead>
            {table.getHeaderGroups().map(headerGroup => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map(header => (
                  <th
                    key={header.id}
                    colSpan={header.colSpan}
                    {...(header.isPlaceholder
                      ? { className: 'placeholder' }
                      : {})}
                    {...(header.column.getCanSort()
                      ? { 'aria-sort': false }
                      : {})}>
                    {header.isPlaceholder ? null : header.column.getCanSort() ? (
                      <button
                        className={
                          'bx--table-sort' +
                          (header.column.getIsSorted()
                            ? ' bx--table-sort--active'
                            : '')
                        }
                        onClick={header.column.getToggleSortingHandler()}>
                        <span className="bx--table-sort__flex">
                          <div className="bx--table-header-label">
                            {flexRender(
                              header.column.columnDef.header,
                              header.getContext()
                            )}
                          </div>
                          {header.column.getIsSorted()
                            ? sortingIcons[header.column.getIsSorted()]
                            : null}
                          <ArrowsVertical20 className="bx--table-sort__icon-unsorted" />
                        </span>
                      </button>
                    ) : (
                      flexRender(
                        header.column.columnDef.header,
                        header.getContext()
                      )
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map(row => (
              <tr key={row.id}>
                {row.getVisibleCells().map(cell => (
                  <td key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
