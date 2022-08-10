import {
  ArrowDown20,
  ArrowsVertical20,
  ArrowUp20,
  ArrowsHorizontal20,
} from '@carbon/icons-react';
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
import { useDrag, useDrop } from 'react-dnd';

function collectLeafColumnIndices(columDefinitions, output) {
  columDefinitions.forEach(columnDefinition => {
    if (columnDefinition.hasOwnProperty('columns')) {
      collectLeafColumnIndices(columnDefinition.columns, output);
    } else {
      output.push(columnDefinition.id);
    }
  });
}

function reorderColumn(draggedColumnId, targetColumnId, columnOrder) {
  columnOrder.splice(
    columnOrder.indexOf(targetColumnId),
    0,
    columnOrder.splice(columnOrder.indexOf(draggedColumnId), 1)[0]
  );
  return [...columnOrder];
}

function DraggableColumnHeader({ header, table }) {
  const { getState, setColumnOrder } = table;
  const { columnOrder } = getState();
  const { column } = header;

  const [{ isOver }, dropRef] = useDrop({
    accept: 'column',
    drop: draggedColumn => {
      console.log('column', column.id, 'dragged', draggedColumn.id, 'before');
      console.log(columnOrder);
      const newColumnOrder = reorderColumn(
        draggedColumn.id,
        column.id,
        columnOrder
      );
      console.log('after');
      console.log(newColumnOrder);
      setColumnOrder(newColumnOrder);
    },
    collect: monitor => ({
      isOver: monitor.isOver(),
    }),
  });

  const [{ isDragging }, dragRef, previewRef] = useDrag({
    collect: monitor => ({
      isDragging: monitor.isDragging(),
    }),
    item: () => column,
    type: 'column',
  });

  const thClassNames = [];
  if (!header.column.getCanSort()) thClassNames.push('header-unsortable');
  if (header.isPlaceholder) thClassNames.push('placeholder');
  else thClassNames.push('no-placeholder');
  if (isOver) thClassNames.push('is-over');

  return (
    <th
      aria-sort={'none'}
      ref={dropRef}
      style={{ opacity: isDragging ? 0.5 : 1 }}
      colSpan={header.colSpan}
      className={thClassNames.join(' ')}>
      <div ref={previewRef}>
        {header.isPlaceholder ? null : header.column.getCanSort() ? (
          <button
            className={
              'bx--table-sort' +
              (header.column.getIsSorted() ? ' bx--table-sort--active' : '')
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
              {header.column.columns.length ? null : (
                <span className="header-dnd" ref={dragRef}>
                  <ArrowsHorizontal20 className="bx--table-sort__icon-unsorted" />
                </span>
              )}
            </span>
          </button>
        ) : (
          <button className="bx--table-sort">
            <span className="bx--table-sort__flex">
              <div className="bx--table-header-label">
                {flexRender(
                  header.column.columnDef.header,
                  header.getContext()
                )}
              </div>
              {header.column.columns.length ? null : (
                <span className="header-dnd" ref={dragRef}>
                  <ArrowsHorizontal20 className="bx--table-sort__icon-unsorted" />
                </span>
              )}
            </span>
          </button>
        )}
      </div>
    </th>
  );
}

const sortingIcons = {
  asc: <ArrowUp20 className="bx--table-sort__icon" />,
  desc: <ArrowDown20 className="bx--table-sort__icon" />,
};

export function FeaturedTable({ columns, data, experiment }) {
  const defaultColumnOrder = [];
  collectLeafColumnIndices(columns, defaultColumnOrder);
  const [columnOrder, setColumnOrder] = React.useState(defaultColumnOrder);
  console.log('Default');
  console.log(columnOrder);
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
    state: { sorting, columnVisibility, pagination, columnOrder },
    getCoreRowModel: getCoreRowModel(),
    onSortingChange: setSorting,
    getSortedRowModel: getSortedRowModel(),
    onColumnVisibilityChange: setColumnVisibility,
    // getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    onPaginationChange: setPagination,
    // manualPagination: true
    onColumnOrderChange: setColumnOrder,
  });
  const itemSelectAll = { id: '(select all)', label: '(select all)' };
  const selectableColumns = [
    itemSelectAll,
    ...table.getAllLeafColumns().map((col, index) => ({
      id: col.id,
      label: col.columnDef.header,
    })),
  ];
  const [selectedColumns, setSelectedColumns] = React.useState(
    selectableColumns
  );
  const columnVisibilitySetter = multiSelect => {
    const previousColSet = new Set(selectedColumns.map(item => item.id));
    const colSet = new Set(multiSelect.selectedItems.map(item => item.id));
    if (!previousColSet.has(itemSelectAll.id) && colSet.has(itemSelectAll.id)) {
      table.getAllLeafColumns().forEach(column => colSet.add(column.id));
    } else if (
      previousColSet.has(itemSelectAll.id) &&
      colSet.has(itemSelectAll.id) &&
      previousColSet.size > colSet.size
    ) {
      colSet.delete(itemSelectAll.id);
    }
    const def = {};
    table
      .getAllLeafColumns()
      .forEach(column => (def[column.id] = colSet.has(column.id)));
    table.setColumnVisibility(def);
    setSelectedColumns([
      ...(colSet.has(itemSelectAll.id) ? [itemSelectAll] : []),
      ...selectableColumns.filter(item => !!def[item.id]),
    ]);
  };
  const setCarbonPagination = ({ page, pageSize }) => {
    setPagination({ pageIndex: page - 1, pageSize: pageSize });
  };
  const experimentWords = experiment.split(/(\W|_)/).map((p, i) => (
    <span className="experiment-word" key={i}>
      {p}
    </span>
  ));
  return (
    <div className="bx--data-table-container">
      <div className="bx--data-table-header">
        <Grid>
          <Row>
            <Column>
              <div>
                <h4 className="bx--data-table-header__title">
                  Experiment Trials for "{experimentWords}"
                </h4>
                <p className="bx--data-table-header__description">
                  {data.length} trial(s) for experiment "{experimentWords}"
                </p>
              </div>
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
                selectedItems={selectedColumns}
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
                  <DraggableColumnHeader
                    key={header.id}
                    header={header}
                    table={table}
                  />
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
