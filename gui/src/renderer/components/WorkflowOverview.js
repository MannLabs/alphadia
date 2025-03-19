import * as React from 'react'
import { useMethod } from '../logic/context'
import { useMethodDispatch } from '../logic/context';
import { Masonry } from '@mui/lab';
import { Box, Typography, Switch, useTheme, List, ListItem, ListItemText, Stack } from '@mui/material';

import ReactFlow, {
    ReactFlowProvider,
    Background,
    Handle,
    Position
} from 'reactflow';
import 'reactflow/dist/style.css';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import QuestionMarkIcon from '@mui/icons-material/QuestionMark';
import InfoTooltip from './InfoTooltip';


const CustomNodeBase = ({ data, id, onEnabledChange, children, tooltipTitle }) => {
    const theme = useTheme();
    const isActive = !data.isSwitchable || data.enabled;
    return (
        <div style={{
            padding: '10px',
            border: `1px solid ${isActive ? theme.palette.primary.main : theme.palette.divider}`,
            borderRadius: '5px',
            background: theme.palette.background.default,
            width: '200px',
            minHeight: '160px',
            transition: 'border-color 0.3s ease-in-out',
            pointerEvents: 'auto',
        }}>
            <InfoTooltip
                placement="right"
                title={tooltipTitle}
            >
            {data.isTarget && isActive && (
                <Handle
                    type="target"
                    position={Position.Left}
                    isConnectable={false}
                    style={{
                        width: '8px',
                        height: '8px',
                        background: theme.palette.text.secondary
                    }}
                />
            )}
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px', justifyContent: 'space-between' }}>
                <Typography
                    variant="subtitle1"
                    style={{
                        fontWeight: 'bold',
                        color: isActive ? theme.palette.primary.main : theme.palette.divider,
                        transition: 'color 0.3s ease-in-out',
                    }}
                >
                    {data.label}
                </Typography>

                {data.isSwitchable && (
                    <div style={{ pointerEvents: 'auto' }}>
                        <Switch
                            size="small"
                            checked={data.enabled}
                            onChange={() => onEnabledChange(id)}
                            sx={{
                                '& .MuiSwitch-thumb': {
                                    transition: 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                                },
                                '& .MuiSwitch-track': {
                                    transition: 'opacity 0.3s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                                },
                            }}
                        />
                    </div>
                )}
            </div>
            {/* Add custom component rendering with opacity transition */}
            {children && (
                <div style={{
                    opacity: isActive ? 1 : 0.5,
                    transition: 'opacity 0.3s ease-in-out'
                }}>
                    {children}
                </div>
            )}

            {data.isSource && isActive && (
                <Handle
                    type="source"
                    position={Position.Right}
                    isConnectable={false}
                    style={{
                        width: '8px',
                        height: '8px',
                        background: theme.palette.text.secondary
                    }}
                />
            )}
            </InfoTooltip>
        </div>
    );
};

// fixed positions for the nodes
const X0 = 50;
const DELTA = 250;

// Add this custom edge component after the CustomNode component
const AnimatedEdge = ({
    sourceX,
    sourceY,
    targetX,
    targetY,
}) => {
    const theme = useTheme();
    const path = `M ${sourceX} ${sourceY} L ${targetX} ${targetY}`;

    return (
        <path
            d={path}
            className="animated-edge"
            fill="none"
            stroke={theme.palette.primary.main}
            strokeWidth={4}
            strokeDasharray={4}
        />
    );
};

const InputNode = ({ data, id, onEnabledChange }) => {
    const method = useMethod();
    const theme = useTheme();

    const hasLibrary = method.library.path !== "";
    const hasFasta = method.fasta_list.path.length > 0;
    const numFasta = method.fasta_list.path.length;

    const hasRawFiles = method.raw_path_list.path.length > 0;
    const numRawFiles = method.raw_path_list.path.length;

    const getIcon = (hasItem, otherItemSet) => {
        if (hasItem) {
            return <CheckIcon sx={{ color: theme.palette.success.main, fontSize: 20 }} />;
        } else if (otherItemSet) {
            return <QuestionMarkIcon sx={{fontSize: 20 }} />;
        } else {
            return <CloseIcon sx={{ color: theme.palette.error.main, fontSize: 20 }} />;
        }
    };

    return (
        <CustomNodeBase data={data} id={id} onEnabledChange={onEnabledChange}>
            <Stack spacing={1} sx={{paddingTop: 2}}>
                <Stack direction="row" alignItems="center" spacing={1}>
                    {getIcon(hasLibrary, hasFasta)}
                    <Typography variant="body2">Spectral Library</Typography>
                </Stack>
                <Stack direction="row" alignItems="center" spacing={1}>
                    {getIcon(hasFasta, hasLibrary)}
                    <Typography variant="body2">FASTA Files ({numFasta})</Typography>
                </Stack>
                <Stack direction="row" alignItems="center" spacing={1}>
                    {hasRawFiles ? (
                        <CheckIcon sx={{ color: theme.palette.success.main, fontSize: 20 }} />
                    ) : (
                        <CloseIcon sx={{ color: theme.palette.error.main, fontSize: 20 }} />
                    )}
                    <Typography variant="body2">Raw Files ({numRawFiles})</Typography>
                </Stack>
            </Stack>
        </CustomNodeBase>
    );
};

const TransferLearningNode = ({ data, id, onEnabledChange }) => {
    return (
        <CustomNodeBase data={data} id={id} onEnabledChange={onEnabledChange} tooltipTitle={
                    <Stack spacing={0.5}>
                        <Typography sx={{ fontWeight: 'bold' }}>Transfer Learning</Typography>
                        <Typography sx={{ fontFamily: 'monospace' }}>{`[general.transfer_step_enabled]`}</Typography>
                        <Typography>
                            The transfer learning step will perform a whole search of all selected raw files before the first search. It can be used with a spectral library or with fasta files and fully predicted library.
                            <br/><br/>
                            All parameters set in the configuration will also be used for this step (except those required to switch on the specific behavior of this step).
                            <br/><br/>
                            If this step is enabled, the transfer library module as well as transfer learning module will be automatically activated for this step:
                        </Typography>
                        <Typography sx={{ fontFamily: 'monospace' }}>
                            transfer_library.enabled  = true<br/>
                            transfer_learning.enabled = true
                        </Typography>
                        <Typography>
                            The configuration section for transfer learning and transfer library offer advanced parameters to tune the behavior of the transfer learning step.
                        </Typography>
                    </Stack>
                }
            >
                <Typography variant="body2" sx={{paddingTop: 2, pointerEvents: 'auto'}}>
                    Search all files and train custom PeptDeep model. Will be used for main search.
                </Typography>
        </CustomNodeBase>
    );
};

const LibrarySearchNode = ({ data, id, onEnabledChange }) => (
    <CustomNodeBase data={data} id={id} onEnabledChange={onEnabledChange} tooltipTitle={
        <Stack spacing={0.5}>
            <Typography sx={{ fontWeight: 'bold' }}>Main Search</Typography>
            <Typography>
                Perform search of all raw files using the spectral library or fully predicted library.
                <br/><br/>
                If a transfer learning step is used, the main search will use the learned PeptDeep model to re-predict the library or fasta files.
            </Typography>
        </Stack>
    }>
        <Typography variant="body2" sx={{paddingTop: 2}}>
            Main search step. Performs spectral library prediction if needed.
        </Typography>
    </CustomNodeBase>
);

const MatchBetweenRunsNode = ({ data, id, onEnabledChange }) => (
    <CustomNodeBase data={data} id={id} onEnabledChange={onEnabledChange} tooltipTitle={
        <Stack spacing={0.5}>
            <Typography sx={{ fontWeight: 'bold' }}>Match Between Runs</Typography>
            <Typography sx={{ fontFamily: 'monospace' }}>{`[general.mbr_step_enabled]`}</Typography>
            <Typography>
                Search all raw files using spectral library from main search.
                <br/><br/>
                Whether to perform a 'second search' step after the first search. All parameters set here will also be used for this step (except those required to switch on the specific behavior of this step).
                <br/><br/>
                This step will perform FDR controlled requantification of all peptides identified across the experiment. Generally recommended for samples with proteome overlap as it increases data completeness.
                If automatic hyperparameter optimization was activated, the median optimal MS1 and MS2 error tolerances from the main search will be used as optimal fixed parameters.
            </Typography>
        </Stack>
    }>
        <Typography variant="body2" sx={{paddingTop: 2}}>
            (Recommended)<br/>
            Use spectral library from main search to requantify raw files.
        </Typography>
    </CustomNodeBase>
);

const OutputNode = ({ data, id, onEnabledChange }) => {
    const method = useMethod();
    const theme = useTheme();

    const hasOutputFolder = method.output_directory.path !== "";

    return (
        <CustomNodeBase data={data} id={id} onEnabledChange={onEnabledChange}>
            <Stack spacing={1} sx={{paddingTop: 2}}>
                <Stack direction="row" alignItems="center" spacing={1}>
                    {hasOutputFolder ? (
                        <CheckIcon sx={{ color: theme.palette.success.main, fontSize: 20 }} />
                    ) : (
                        <CloseIcon sx={{ color: theme.palette.error.main, fontSize: 20 }} />
                    )}
                    <Typography variant="body2">Output Folder</Typography>
                </Stack>
            </Stack>
        </CustomNodeBase>
    );
};

// Update createNodeTypes to use the renamed component
const createNodeTypes = (handleEnabledChange) => ({
    inputIO: (props) => <InputNode {...props} onEnabledChange={handleEnabledChange} />,
    transferLearning: (props) => <TransferLearningNode {...props} onEnabledChange={handleEnabledChange} />,
    librarySearch: (props) => <LibrarySearchNode {...props} onEnabledChange={handleEnabledChange} />,
    matchBetweenRuns: (props) => <MatchBetweenRunsNode {...props} onEnabledChange={handleEnabledChange} />,
    outputIO: (props) => <OutputNode {...props} onEnabledChange={handleEnabledChange} />
});

// Add this after createNodeTypes
const edgeTypes = {
    animated: AnimatedEdge,
};

const getNodesForNodeStates = (transferLearningEnabled, matchBetweenRunsEnabled) => {
    return [
        {
            id: '1',
            position: { x: X0, y: 75 },
            data: {
                label: 'Input',
                isSwitchable: false,
                enabled: true,
                isSource: true,
                isTarget: false
            },
            type: 'inputIO'
        },
        {
            id: '2',
            position: { x: X0 + DELTA, y: 75 },
            data: {
                label: 'Transfer Learning',
                isSwitchable: true,
                enabled: transferLearningEnabled,
                isSource: true,
                isTarget: true
            },
            type: 'transferLearning'
        },
        {
            id: '3',
            position: { x: X0 + 2 * DELTA, y: 75 },
            data: {
                label: 'Main Search',
                isSource: true,
                isTarget: true,
                isSwitchable: false,
                enabled: true
            },
            type: 'librarySearch'
        },
        {
            id: '4',
            position: { x: X0 + 3 * DELTA, y: 75 },
            data: {
                label: 'Match Between Runs',
                isSource: true,
                isTarget: true,
                isSwitchable: true,
                enabled: matchBetweenRunsEnabled
            },
            type: 'matchBetweenRuns'
        },
        {
            id: '5',
            position: { x: X0 + 4 * DELTA, y: 75 },
            data: {
                label: 'Output',
                isSource: false,
                isTarget: true,
                isSwitchable: false,
                enabled: true
            },
            type: 'outputIO'
        },
    ];
};

// Update all edge definitions to include the custom type
const basicEdges = [
    { id: 'e1-3', source: '1', target: '3', type: 'animated' },
    { id: 'e3-5', source: '3', target: '5', type: 'animated' },
];

const transferLearningEdges = [
    { id: 'e1-2', source: '1', target: '2', type: 'animated' },
    { id: 'e2-3', source: '2', target: '3', type: 'animated' },
    { id: 'e3-5', source: '3', target: '5', type: 'animated' },
];

const matchBetweenRunsEdges = [
    { id: 'e1-3', source: '1', target: '3', type: 'animated' },
    { id: 'e3-4', source: '3', target: '4', type: 'animated' },
    { id: 'e4-5', source: '4', target: '5', type: 'animated' },
];

const allEdges = [
    { id: 'e1-2', source: '1', target: '2', type: 'animated' },
    { id: 'e2-3', source: '2', target: '3', type: 'animated' },
    { id: 'e3-4', source: '3', target: '4', type: 'animated' },
    { id: 'e4-5', source: '4', target: '5', type: 'animated' },
];

const getEdgesForNodeStates = (node2Enabled, node4Enabled) => {
    if (node2Enabled && node4Enabled) {
        return allEdges;
    } else if (node2Enabled) {
        return transferLearningEdges;
    } else if (node4Enabled) {
        return matchBetweenRunsEdges;
    }
    return basicEdges;
};

const getWorkflowStates = (method) => {
    const generalParameterGroup = method.config.find(group => group.id === "general");

    const transferLearningStep = generalParameterGroup.parameters.find(parameter => parameter.id === "transfer_step_enabled");
    const transferLearningStepEnabled = transferLearningStep.value;

    const matchBetweenRunsStep = generalParameterGroup.parameters.find(parameter => parameter.id === "mbr_step_enabled");
    const matchBetweenRunsStepEnabled = matchBetweenRunsStep.value;

    return {
        edges: getEdgesForNodeStates(transferLearningStepEnabled, matchBetweenRunsStepEnabled),
        nodes: getNodesForNodeStates(transferLearningStepEnabled, matchBetweenRunsStepEnabled)
    };
};

const WorkflowOverview = () => {
    const reactFlowInstance = React.useRef(null);
    const method = useMethod();

    const initialStates = React.useMemo(() => getWorkflowStates(method), []);
    const [edges, setEdges] = React.useState(initialStates.edges);
    const [nodes, setNodes] = React.useState(initialStates.nodes);

    const dispatch = useMethodDispatch();

    React.useEffect(() => {
        const { edges: newEdges, nodes: newNodes } = getWorkflowStates(method);
        setEdges(newEdges);
        setNodes(newNodes);
    }, [method]);

    React.useEffect(() => {
        const handleResize = () => {
            if (reactFlowInstance.current) {
                reactFlowInstance.current.fitView();
            }
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const handleEnabledChange = (nodeId) => {

        const parameterId = nodeId === '2' ? "transfer_step_enabled" : "mbr_step_enabled";
        const oldState = nodes.find(node => node.id === nodeId)?.data.enabled || false;

        dispatch({
            type: 'updateParameter',
            parameterGroupId: "general",
            parameterId: parameterId,
            value: !oldState
        })

    };

    // Memoize the nodeTypes object
    const nodeTypes = React.useMemo(
        () => createNodeTypes(handleEnabledChange),
        [handleEnabledChange]
    );

    return (
        <Box sx={{
            height: '180px',
            width: '100%',
            marginBottom: '10px',
            '& .react-flow__node': {
                zIndex: '-1 !important'
            },
            '& .animated-edge': {
                animation: 'dashdraw 15s linear infinite',
            },
            '@keyframes dashdraw': {
                '0%': {
                    strokeDashoffset: 200,
                },
                '100%': {
                    strokeDashoffset: 0,
                },
            },
        }}>
            <ReactFlowProvider>
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    nodeTypes={nodeTypes}
                    edgeTypes={edgeTypes}
                    fitView
                    onInit={instance => {
                        reactFlowInstance.current = instance;
                    }}
                    proOptions={{ hideAttribution: true }}
                    nodesDraggable={false}
                    nodesConnectable={false}
                    zoomOnScroll={false}
                    panOnScroll={false}
                    panOnDrag={false}
                    zoomOnDoubleClick={false}
                    zoomOnPinch={false}
                    selectNodesOnDrag={false}
                    selectionOnDrag={false}
                    elementsSelectable={false}
                    preventScrolling={true}
                    style={{
                        pointerEvents: 'none',
                        cursor: 'default'
                    }}
                >
                    <Background
                        variant="dots"
                        gap={12}
                        size={1}
                        style={{ pointerEvents: 'none' }}
                    />
                </ReactFlow>
            </ReactFlowProvider>
        </Box>
    );
};

export default WorkflowOverview;
