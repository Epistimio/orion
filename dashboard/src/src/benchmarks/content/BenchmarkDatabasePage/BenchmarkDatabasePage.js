import React from 'react';

import TrialTable from './TrialTable';

const headers = [
  {
    key: 'id',
    header: 'ID',
  },
  {
    key: 'experiment',
    header: 'Experiment',
  },
  {
    key: 'status',
    header: 'Status',
  },
  {
    key: 'created_on',
    header: 'Created',
  },
  {
    key: 'params',
    header: 'Parameters',
  },
  {
    key: 'objective',
    header: 'Objective',
  },
];

const rows = [
  {
    id: '1',
    experiment: '1',
    status: 'Completed',
    created_on: '2020-12-01 05:05:05',
    updatedAt: 'Date',
    params: [],
    results: [{ type: 'objective', name: 'loss', value: 1 }],
  },
  {
    id: '2',
    name: 'Repo 2',
    createdAt: 'Date',
    updatedAt: 'Date',
    issueCount: '123',
    stars: '456',
    links: 'Links',
  },
  {
    id: '3',
    name: 'Repo 3',
    createdAt: 'Date',
    updatedAt: 'Date',
    issueCount: '123',
    stars: '456',
    links: 'Links',
  },
];

const getRowItems = rows =>
  rows.map(row => ({
    ...row,
    id: row.id,
    experiment: row.experiment,
    status: row.status,
    created_on: new Date(row.created_on).toLocaleDateString(),
    params: 'dunno',
    objective: 'bad',
  }));

export const BenchmarkDatabasePage = () => {
  return (
    <div className="bx--grid bx--grid--full-width bx--grid--no-gutter database-page">
      <div className="bx--row database-page__r1">
        <div className="bx--col-lg-16">
          <TrialTable headers={headers} rows={getRowItems(rows)} />
        </div>
      </div>
    </div>
  );
};
