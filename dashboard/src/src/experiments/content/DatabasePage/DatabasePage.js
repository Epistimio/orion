import React from 'react';
import { Backend, DEFAULT_BACKEND } from '../../../utils/queryServer';
import { BackendContext } from '../../BackendContext';
import { Grid, Row, Column, MultiSelect } from 'carbon-components-react';
import { flattenObject } from '../../../utils/flattenObject';
import { ArrowUp20, ArrowDown20, ArrowsVertical20 } from '@carbon/icons-react';
import {
  useReactTable,
  flexRender,
  getSortedRowModel,
  getCoreRowModel,
} from '@tanstack/react-table';

/**
 * Component to pretty display an object (JSON dictionary) into data table.
 * Used to render trial parameters and statistics.
 */
class ObjectToGrid extends React.Component {
  render() {
    const object = this.props.object;
    const keys = Object.keys(object);
    if (!keys.length) return '';
    keys.sort();
    return (
      <Grid condensed fullWidth className="object-to-grid">
        {keys.map(key => (
          <Row key={key}>
            <Column className="object-to-grid-key">
              <strong>
                <em>{key}</em>
              </strong>
            </Column>
            <Column>
              {Array.isArray(object[key])
                ? object[key].map((value, i) => (
                    <div key={i}>{value.toString()}</div>
                  ))
                : object[key].toString()}
            </Column>
          </Row>
        ))}
      </Grid>
    );
  }
}

/**
 * Utility class to provide and cache experiment trials.
 */
class TrialsProvider {
  constructor(address) {
    this.backend = new Backend(address);
    this.trials = {};
  }
  async get(experiment) {
    if (!this.trials.hasOwnProperty(experiment)) {
      const queryTrials = await this.backend.query(`trials/${experiment}`);
      const trialIndices = queryTrials.map(trial => trial.id);
      trialIndices.sort();
      const trials = [];
      /**
       * Map to check whether each param column is sortable.
       * Array params are not considered sortable,
       * except if they contains only 1 element.
       */
      const sortableParamCols = {};
      for (let trialID of trialIndices) {
        const rawTrial = await this.backend.query(
          `trials/${experiment}/${trialID}`
        );
        // Flatten parameters
        const flattenedParameters = flattenObject(
          rawTrial.parameters,
          // Add prefix `params`
          // to prevent collision with existing keys in trial object
          'params'
        );
        // Prepare rendering for array parameters
        for (let key of Object.keys(flattenedParameters)) {
          let sortableCell = true;
          if (Array.isArray(flattenedParameters[key])) {
            if (flattenedParameters[key].length === 1) {
              // Array contains only 1 element.
              // Flatten it and assume element is displayable as-is.
              flattenedParameters[key] = flattenedParameters[key][0];
            } else {
              // Real array with many values.
              // Render it immediately and mark cell as not sortable.
              flattenedParameters[key] = flattenedParameters[
                key
              ].map((value, i) => <div key={i}>{value.toString()}</div>);
              sortableCell = false;
            }
          }
          // Param column is sortable if all its cells are sortable.
          if (sortableParamCols.hasOwnProperty(key)) {
            sortableParamCols[key] = sortableParamCols[key] && sortableCell;
          } else {
            sortableParamCols[key] = sortableCell;
          }
        }
        // Save flattened keys in specific property `paramKeys` for later
        rawTrial.paramKeys = Object.keys(flattenedParameters);
        const trial = { ...rawTrial, ...flattenedParameters };
        // Save statistics as already rendered components.
        trial.statistics = <ObjectToGrid object={trial.statistics} />;
        trials.push(trial);
      }
      // Prepare headers for this experiment using `paramKeys` from first trial
      // We assume paramKeys is the same for all trials
      const paramKeys = trials[0].paramKeys.slice();
      paramKeys.sort();
      const paramHeaders = paramKeys.map(paramKey => ({
        key: paramKey,
        // Ignore prefix `params.`
        header: `Parameter ${paramKey.substr(7)}`,
      }));
      const trialHeaders = [
        {
          key: 'id',
          header: 'ID',
        },
        ...paramHeaders,
        {
          key: 'submitTime',
          header: 'Submit time',
        },
        {
          key: 'startTime',
          header: 'Start time',
        },
        {
          key: 'endTime',
          header: 'End time',
        },
        {
          key: 'objective',
          header: 'Objective',
        },
        {
          key: 'statistics',
          header: 'Statistics',
        },
      ];
      const altHeaders = [
        {
          accessorKey: 'id',
          header: 'ID',
          sortingFn: 'text',
          cell: info =>
            info.getValue().length > 7 ? (
              <span title={info.getValue()}>
                {info.getValue().substr(0, 7)}...
              </span>
            ) : (
              info.getValue()
            ),
        },
        {
          // Grouped parameters columns
          header: 'Parameters',
          columns: paramKeys.map(k => {
            const p = { accessorFn: r => r[k], header: k.substr(7) };
            if (!sortableParamCols[k]) {
              // column not sortable
              p.cell = props => props.getValue();
              p.enableSorting = false;
            }
            return p;
          }),
        },
        {
          accessorKey: 'submitTime',
          header: 'Submit time',
        },
        {
          accessorKey: 'startTime',
          header: 'Start time',
        },
        {
          accessorKey: 'endTime',
          header: 'End time',
        },
        {
          accessorKey: 'objective',
          header: 'Objective',
        },
        {
          // not sortable
          accessorKey: 'statistics',
          header: 'Statistics',
          cell: props => props.getValue(),
        },
      ];
      // Map to specify sortable columns.
      const sortableCols = {
        ...sortableParamCols,
        id: true,
        submitTime: true,
        startTime: true,
        endTime: true,
        objective: true,
        statistics: false,
      };
      // Array to specify sortable columns by index.
      const sortable = trialHeaders.map(header => sortableCols[header.key]);
      this.trials[experiment] = {
        headers: trialHeaders,
        trials: trials,
        sortable: sortable,
        newHeaders: altHeaders,
      };
    }
    return this.trials[experiment];
  }
}

/**
 * Singleton to provide experiment trials.
 * @type {TrialsProvider}
 */
const TRIALS_PROVIDER = new TrialsProvider(DEFAULT_BACKEND);

const sortingIcons = {
  asc: <ArrowUp20 className="bx--table-sort__icon" />,
  desc: <ArrowDown20 className="bx--table-sort__icon" />,
};

function MyTable({ columns, data, experiment }) {
  const [sorting, setSorting] = React.useState([]);
  const [columnVisibility, setColumnVisibility] = React.useState({});
  const table = useReactTable({
    columns,
    data,
    state: { sorting, columnVisibility },
    getCoreRowModel: getCoreRowModel(),
    onSortingChange: setSorting,
    getSortedRowModel: getSortedRowModel(),
    onColumnVisibilityChange: setColumnVisibility,
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

class DatabasePage extends React.Component {
  // Control variable to avoid setting state if component was unmounted before an asynchronous API call finished.
  _isMounted = false;
  static contextType = BackendContext;
  constructor(props) {
    super(props);
    this.state = { experiment: null, trials: null };
  }
  render() {
    if (this.state.experiment === null)
      return 'No trials to display, please select an experiment.';
    if (this.state.trials === null)
      return `Loading trials for experiment "${this.state.experiment}" ...`;
    if (this.state.trials === false)
      return `Unable to load trials for experiment "${this.state.experiment}".`;
    return (
      <div className="bx--grid bx--grid--full-width bx--grid--no-gutter database-page">
        <div className="bx--row database-page__r1">
          <div className="bx--col-lg-16">
            <MyTable
              data={this.state.trials.trials}
              columns={this.state.trials.newHeaders}
              experiment={this.state.experiment}
            />
          </div>
        </div>
      </div>
    );
  }
  componentDidMount() {
    this._isMounted = true;
    const experiment = this.context.experiment;
    if (experiment !== null) {
      this.loadTrials(experiment);
    }
  }
  componentWillUnmount() {
    this._isMounted = false;
  }
  componentDidUpdate(prevProps, prevState, snapshot) {
    // We must check if selected experiment changed
    const experiment = this.context.experiment;
    if (this.state.experiment !== experiment) {
      if (experiment === null) {
        this.setState({ experiment, trials: null });
      } else {
        this.loadTrials(experiment);
      }
    }
  }
  loadTrials(experiment) {
    this.setState({ experiment, trials: null }, () => {
      TRIALS_PROVIDER.get(experiment)
        .then(trials => {
          if (this._isMounted) {
            this.setState({ trials });
          }
        })
        .catch(error => {
          console.error(error);
          if (this._isMounted) {
            this.setState({ trials: false });
          }
        });
    });
  }
}

export default DatabasePage;
