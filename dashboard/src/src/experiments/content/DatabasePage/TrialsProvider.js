import { Backend } from '../../../utils/queryServer';
import { flattenObject } from '../../../utils/flattenObject';
import { ObjectToGrid } from './ObjectToGrid';

/**
 * Utility class to provide and cache experiment trials.
 */
export class TrialsProvider {
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
      const headers = [
        {
          id: 'id',
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
          id: 'parameters',
          header: 'Parameters',
          columns: paramKeys.map(k => {
            const p = { id: k, accessorFn: r => r[k], header: k.substr(7) };
            if (!sortableParamCols[k]) {
              // column not sortable
              p.cell = props => props.getValue();
              p.enableSorting = false;
            }
            return p;
          }),
        },
        {
          id: 'submitTime',
          accessorKey: 'submitTime',
          header: 'Submit time',
        },
        {
          id: 'startTime',
          accessorKey: 'startTime',
          header: 'Start time',
        },
        {
          id: 'endTime',
          accessorKey: 'endTime',
          header: 'End time',
        },
        {
          id: 'status',
          accessorKey: 'status',
          header: 'Status',
        },
        {
          id: 'objective',
          accessorKey: 'objective',
          header: 'Objective',
        },
        {
          // not sortable
          id: 'statistics',
          accessorKey: 'statistics',
          header: 'Statistics',
          cell: props => props.getValue(),
          enableSorting: false,
        },
      ];
      this.trials[experiment] = {
        headers: headers,
        trials: trials,
      };
    }
    return this.trials[experiment];
  }
}
