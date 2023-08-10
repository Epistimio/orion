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
  Modal,
  MultiSelect,
  Pagination,
  Row,
} from 'carbon-components-react';
import { useDrag, useDrop } from 'react-dnd';
import ReactDOM from 'react-dom';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

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
      const newColumnOrder = reorderColumn(
        draggedColumn.id,
        column.id,
        columnOrder
      );
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

/**
 * Simple state manager for modals.
 * Reference: https://react.carbondesignsystem.com/?path=/story/components-modal--with-state-manager
 */
const ModalStateManager = ({
  renderLauncher: LauncherContent,
  children: ModalContent,
}) => {
  const [open, setOpen] = React.useState(false);
  return (
    <>
      {!ModalContent || typeof document === 'undefined'
        ? null
        : ReactDOM.createPortal(
            <ModalContent open={open} setOpen={setOpen} />,
            document.body
          )}
      {LauncherContent && <LauncherContent open={open} setOpen={setOpen} />}
    </>
  );
};

export function FeaturedTable({
  columns,
  data,
  experiment,
  trialStatus,
  nbTrials,
}) {
  const defaultColumnOrder = [];
  collectLeafColumnIndices(columns, defaultColumnOrder);
  const [columnOrder, setColumnOrder] = React.useState(defaultColumnOrder);
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
    <DndProvider backend={HTML5Backend}>
      <div className="bx--data-table-container">
        <div className="bx--data-table-header">
          <Grid>
            <Row>
              <Column>
                <div>
                  <h4
                    className="bx--data-table-header__title"
                    title={`Experiment Trials for "${experiment}"`}>
                    Experiment Trials for "{experimentWords}"
                  </h4>
                  <p className="bx--data-table-header__description">
                    {nbTrials} trial(s) for experiment "{experimentWords}"
                    {trialStatus && nbTrials !== data.length
                      ? `, ${data.length} displayed for status "${trialStatus}"`
                      : ''}
                  </p>
                </div>
              </Column>
              <Column>
                <Pagination
                  id="trials-pagination"
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
                <ModalStateManager
                  key={row.id}
                  renderLauncher={({ setOpen }) => (
                    <tr className="trial-row" onClick={() => setOpen(true)}>
                      {row.getVisibleCells().map(cell => (
                        <td key={cell.id}>
                          {flexRender(
                            cell.column.columnDef.cell,
                            cell.getContext()
                          )}
                        </td>
                      ))}
                    </tr>
                  )}>
                  {({ open, setOpen }) => (
                    <Modal
                      modalLabel={`Trial info`}
                      modalHeading={`${experiment} / ${row.original.id}`}
                      passiveModal={true}
                      // primaryButtonText="Add"
                      // secondaryButtonText="Cancel"
                      open={open}
                      onRequestClose={() => setOpen(false)}>
                      <Grid>
                        {defaultColumnOrder.map((columnID, index) => (
                          <Row key={columnID}>
                            <Column className="modal-trial-key">
                              <strong>
                                {columnID.startsWith('params.')
                                  ? 'Parameter '
                                  : null}
                                {table.getColumn(columnID).columnDef.header}
                              </strong>
                            </Column>
                            <Column>{row.getValue(columnID)}</Column>
                          </Row>
                        ))}
                      </Grid>
                    </Modal>
                  )}
                </ModalStateManager>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </DndProvider>
  );
}
