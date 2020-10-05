import * as React from 'react';
import { Stack, IStackTokens } from 'office-ui-fabric-react';
import { EXPERIMENT, TRIALS } from '../static/datamodel';
import { Trial } from '../static/model/trial';
import Title1 from './overview/Title1';
import Progressed from './overview/Progress';
import SearchSpace from './overview/SearchSpace';
import BasicInfo from './overview/BasicInfo';
import TrialInfo from './overview/TrialProfile';
import '../static/style/overview.scss';
import '../static/style/logPath.scss';

interface OverviewProps {
    hpoTask: object;
    experimentUpdateBroadcast: number;
    trialsUpdateBroadcast: number;
    metricGraphMode: 'max' | 'min';
    changeMetricGraphMode: (val: 'max' | 'min') => void;
}

interface OverviewState {
    trialConcurrency: number;
}

class Overview extends React.Component<OverviewProps, OverviewState> {
    constructor(props: OverviewProps) {
        super(props);
        this.state = {
            trialConcurrency: EXPERIMENT.trialConcurrency
        };
    }

    clickMaxTop = (event: React.SyntheticEvent<EventTarget>): void => {
        event.stopPropagation();
        // #999 panel active bgcolor; #b3b3b3 as usual
        const { changeMetricGraphMode } = this.props;
        changeMetricGraphMode('max');
    }

    clickMinTop = (event: React.SyntheticEvent<EventTarget>): void => {
        event.stopPropagation();
        const { changeMetricGraphMode } = this.props;
        changeMetricGraphMode('min');
    }

    changeConcurrency = (val: number): void => {
        this.setState({ trialConcurrency: val });
    }

    render(): React.ReactNode {
        const { hpoTask } = this.props;
        const { trialConcurrency } = this.state;
        const { experimentUpdateBroadcast, metricGraphMode } = this.props;
        const searchSpace = this.convertSearchSpace();
        const bestTrials = this.findBestTrials();
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        const bestAccuracy = bestTrials.length > 0 ? bestTrials[0].accuracy! : NaN;
        const accuracyGraphData = this.generateAccuracyGraph(bestTrials);
        const noDataMessage = bestTrials.length > 0 ? '' : 'No data';

        const titleMaxbgcolor = (metricGraphMode === 'max' ? '#999' : '#b3b3b3');
        const titleMinbgcolor = (metricGraphMode === 'min' ? '#999' : '#b3b3b3');
        const stackTokens: IStackTokens = {
            childrenGap: 30,
        };
        return (
            <div className="overview">
                {/* status and experiment block */}
                <Stack>
                    <Title1 text="Experiment" icon="11.png" />
                    <BasicInfo hpoTask={hpoTask} experimentUpdateBroadcast={experimentUpdateBroadcast} />
                </Stack>

                <Stack horizontal className="overMessage">
                    {/* status block */}
                    <Stack.Item grow className="prograph overviewBoder cc">
                        <Title1 text="Status" icon="5.png" />
                        <Progressed
                            hpoTask={hpoTask}
                            bestAccuracy={bestAccuracy}
                            concurrency={trialConcurrency}
                            changeConcurrency={this.changeConcurrency}
                            experimentUpdateBroadcast={experimentUpdateBroadcast}
                        />
                    </Stack.Item>
                    {/* experiment parameters search space tuner assessor... */}
                    <Stack.Item grow styles={{root: {width: 450}}} className="overviewBoder">
                        <Title1 text="Search space" icon="10.png" />
                        <Stack className="experiment">
                            <SearchSpace searchSpace={searchSpace} />
                        </Stack>
                    </Stack.Item>
                    <Stack.Item grow styles={{root: {width: 450}}}>
                        <Title1 text="Config" icon="4.png" />
                        <Stack className="experiment">
                            {/* the scroll bar all the trial profile in the searchSpace div*/}
                            <div className="experiment searchSpace">
                                <TrialInfo
                                    experimentUpdateBroadcast={experimentUpdateBroadcast}
                                    concurrency={trialConcurrency}
                                />
                            </div>
                        </Stack>
                    </Stack.Item>
                </Stack>

                <Stack>
                    <Stack horizontal className="top10bg">
                    </Stack>
                    <Stack horizontal tokens={stackTokens}>
                    </Stack>
                </Stack>
            </div>
        );
    }

    private convertSearchSpace(): object {
        const searchSpace = Object.assign({}, EXPERIMENT.searchSpace);
        Object.keys(searchSpace).map(item => {
            const key = searchSpace[item]._type;
            const value = searchSpace[item]._value;
            switch (key) {
                case 'quniform':
                case 'qnormal':
                case 'qlognormal':
                    searchSpace[item]._value = [value[0], value[1]];
                    break;
                default:
            }
        });
        return searchSpace;
    }

    private findBestTrials(): Trial[] {
        const bestTrials = TRIALS.sort();
        if (this.props.metricGraphMode === 'max') {
            bestTrials.reverse().splice(10);
        } else {
            bestTrials.splice(10);
        }
        return bestTrials;
    }

    private generateAccuracyGraph(bestTrials: Trial[]): object {
        const xSequence = bestTrials.map(trial => trial.sequenceId);
        const ySequence = bestTrials.map(trial => trial.accuracy);

        return {
            // support max show 0.0000000
            grid: {
                left: 67,
                right: 40
            },
            tooltip: {
                trigger: 'item'
            },
            xAxis: {
                name: 'Trial',
                type: 'category',
                data: xSequence
            },
            yAxis: {
                name: 'Default metric',
                type: 'value',
                scale: true,
                data: ySequence
            },
            series: [{
                symbolSize: 6,
                type: 'scatter',
                data: ySequence
            }]
        };
    }
}

export default Overview;
