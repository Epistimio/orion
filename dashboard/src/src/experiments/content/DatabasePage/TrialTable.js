import React from 'react';
import {
  DataTable,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableCell,
} from 'carbon-components-react';

const TrialTable = ({ rows, headers, experiment }) => {
  return (
    <DataTable
      rows={rows}
      headers={headers}
      render={({
        rows,
        headers,
        getHeaderProps,
        getRowProps,
        getTableProps,
      }) => (
        <TableContainer
          title={`Experiment Trials for "${experiment}"`}
          description={`${rows.length} trial(s) for experiment "${experiment}"`}>
          <Table {...getTableProps()}>
            <TableHead>
              <TableRow>
                {headers.map(header => (
                  <TableHeader {...getHeaderProps({ header })}>
                    {header.header}
                  </TableHeader>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map(row => (
                <TableRow key={row.id} {...getRowProps({ row })}>
                  {row.cells.map((cell, cellIndex) => (
                    <TableCell key={cell.id}>
                      {cellIndex === 0 && cell.value.length > 7 ? (
                        <span title={cell.value}>
                          {cell.value.substr(0, 7)}...
                        </span>
                      ) : (
                        cell.value
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    />
  );
};

export default TrialTable;
