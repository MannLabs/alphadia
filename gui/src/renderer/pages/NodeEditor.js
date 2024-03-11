import React, { useCallback, useRef } from 'react';

import { Box } from '@mui/system'
import { useMethod, useMethodDispatch } from '../logic/context'
import { SingleSelect } from '../components'
import { ParameterNode } from '../nodes';
import { useTheme } from '@mui/material/styles';

import { BaseEdge, BezierEdge,  } from 'reactflow';


function AlphaBaseEdge(props) {

    const theme = useTheme();

    const edgeColor = theme.palette.node.edgeColor[props.sourceHandleId] || theme.palette.divider

    

    return <BezierEdge {...props} style={{ stroke: edgeColor, strokeWidth: 2 }} />
    
}

import ReactFlow, {
    MiniMap,
    Controls,
    Background,
    useNodesState,
    useEdgesState,
    addEdge,
    updateEdge,
    ReactFlowProvider,
  } from 'reactflow';
 
import 'reactflow/dist/style.css';

const proOptions = { hideAttribution: true };

const nodeTypes = { parameterNode: ParameterNode };

const edgeTypes = {
    alphaBaseEdge: AlphaBaseEdge,
  };

const DEFAULT_EDGE_PROPS = {
    type: 'alphaBaseEdge',
    updatable: 'target'
};
 
const initialNodes = [
  { id: '1', position: { x: 0, y: 0 }, data: { title: 'LoadFastaFile', targetHandles:[], sourceHandles:['fasta'], parameters:[
    {

        "id": "fasta_file",
        "name": "Fasta file",
        "value": "",
        "description": "Path to the fasta file.",
        "type": "file"
    }

  ]} , type: 'parameterNode'},
  { id: '2', position: { x: 300, y: 0 }, data: { title: 'LibraryDigest', targetHandles:['fasta'], sourceHandles:['speclib'], parameters:[{   
    "id": "enzyme",
    "name": "Enzyme",
    "value": "trypsin",
    "description": "Enzyme used for in-silico digest.",
    "type": "dropdown",
    "options": [
        "arg-c",
        "asp-n",
        "bnps-skatole",
        "caspase 1",
        "caspase 2",
        "caspase 3",
        "caspase 4",
        "caspase 5",
        "caspase 6",
        "caspase 7",
        "caspase 8",
        "caspase 9",
        "caspase 10",
        "chymotrypsin high specificity",
        "chymotrypsin low specificity",
        "chymotrypsin",
        "clostripain",
        "cnbr",
        "enterokinase",
        "factor xa",
        "formic acid",
        "glutamyl endopeptidase",
        "glu-c",
        "granzyme b",
        "hydroxylamine",
        "iodosobenzoic acid",
        "lys-c",
        "lys-n",
        "ntcb",
        "pepsin ph1.3",
        "pepsin ph2.0",
        "proline endopeptidase",
        "proteinase k",
        "staphylococcal peptidase i",
        "thermolysin",
        "thrombin",
        "trypsin_full",
        "trypsin_exception",
        "trypsin_not_p",
        "trypsin",
        "trypsin/p",
        "non-specific",
        "no-cleave"
    ]
},
{
    "id": "missed_cleavages",
    "name": "Missed cleavages",
    "value": 1,
    "description": "Number of missed cleavages for in-silico digest.",
    "type": "integer"
},
{   
    "id": "precursor_len",
    "name": "Precursor length",
    "value": [7, 35],
    "description": "Number of amino acids of generated precursors.",
    "type": "integer_range"
},

{   
    "id": "precursor_charge",
    "name": "Precursor charge",
    "value": [2, 4],
    "description": "Charge states of generated precursors.",
    "type": "integer_range"
}, 
{   
    "id": "precursor_mz",
    "name": "Precursor mz",
    "value": [400.0, 1200.0],
    "description": "Size limit for generated precursors.",
    "type": "float_range"
},
]} , type: 'parameterNode'},
{ id: '3', position: { x: 900, y: 0 }, data: { title: 'ModifyPrecursors', targetHandles:['speclib'], sourceHandles:['speclib'], parameters:[{
    "id": "fixed_modifications",
    "name": "Fixed modifications",
    "value": "Carbamidomethyl@C",
    "description": "Fixed modifications for in-silico digest. Semicolon separated list \n Format: Modification@AminoAcid \n Example: Carbamidomethyl@C;Dimethyl@N-term",
    "type": "string"
},
{
    "id": "variable_modifications",
    "name": "Variable modifications",
    "value": "Oxidation@M;Acetyl@Protein N-term",
    "description": "Variable modifications for in-silico digest. At the moment localisation is not supported. Semicolon separated list \n Example: Oxidation@M;Acetyl@ProteinN-term",
    "type": "string"
},
{
    "id": "max_var_mod_num",
    "name": "Maximum variable modifications",
    "value": 1,
    "description": "Variable modifications for in-silico digest. At the moment localisation is not supported. Semicolon separated list \n Example: Oxidation@M;Acetyl@ProteinN-term",
    "type": "integer"
}]} , type: 'parameterNode'},
{ id: '4', position: { x: 600, y: 0 }, data: { title: 'PeptDeepPredict', targetHandles:['model','speclib'], sourceHandles:['speclib'], parameters:[
{   
    "id": "fragment_mz",
    "name": "Fragment mz",
    "value": [200.0, 2000.0],
    "description": "Size limit for generated fragments.",
    "type": "float_range"
},
{
    "id": "nce",
    "name": "NCE",
    "value": 25.0,
    "description": "Normalized collision energy for fragment generation.",
    "type": "float"
},
{
    "id": "instrument",
    "name": "Instrument",
    "value": "Fusion",
    "description": "Instrument used for ms2 spectrum prediction.",
    "type": "dropdown",
    "options": [
        "Astral",
        "QE",
        "timsTOF",
        "SciexTOF",
        "Fusion",
        "Eclipse",
        "Velos",
        "Elite",
        "OrbitrapTribrid",
        "ThermoTribrid",
        "QE+",
        "QEHF",
        "QEHFX",
        "Exploris",
        "Exploris480"
    ]
}
]} , type: 'parameterNode'},
  { id: '5', position: { x: 900, y: 0 }, data: { title: 'SaveLibraryHDF', targetHandles:['speclib'], sourceHandles:[], parameters:[{
    "id": "hdf_file",
    "name": "HDF file",
    "value": "",
    "description": "Path to the HDF file.",
    "type": "file"
}]} , type: 'parameterNode'},
  { id: '6', position: { x: 300, y: -300 }, data: { title: 'SelectPretrainedModel', targetHandles:[], sourceHandles:['model'], parameters:[{
    "id": "peptdeep_model",
    "name": "PeptDeep model",
    "value": "",
    "description": "Pretrained PeptDeep model.",
    "type": "file"
}]} , type: 'parameterNode'},
];

const initialEdges = [
];


const NodeEditor = () => {

    const edgeUpdateSuccessful = useRef(true);
    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

    const onEdgeUpdateStart = useCallback(() => {
        edgeUpdateSuccessful.current = false;
    }, []);

    const onEdgeUpdateEnd = useCallback((_, edge) => {
        if (!edgeUpdateSuccessful.current) {
          setEdges((eds) => eds.filter((e) => e.id !== edge.id));
        }
    
        edgeUpdateSuccessful.current = true;
    }, []);
        
 
    const onConnect = useCallback(
        (newConnection) => setEdges((eds) => {
            if (newConnection.sourceHandle == newConnection.targetHandle) {
                return addEdge({...newConnection, ...DEFAULT_EDGE_PROPS}, eds)
            } else {
                return eds
            }
            
        }),
        []
    );

    const onEdgeUpdate = useCallback(
        (oldEdge, newConnection) => {
            edgeUpdateSuccessful.current = true;
            if (newConnection.sourceHandle == newConnection.targetHandle) {
                setEdges((eds) => updateEdge(oldEdge, newConnection, eds))
            } else {
                setEdges((eds) => eds.filter((e) => e.id !== oldEdge.id))
            }
        },
        []
    );

    return (
    <Box sx={{ width: '100%', height: "calc(100% - 50px)" }}>
        <ReactFlowProvider>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onEdgeUpdate={onEdgeUpdate}
                onEdgeUpdateStart={onEdgeUpdateStart}
                onEdgeUpdateEnd={onEdgeUpdateEnd}
                onConnect={onConnect}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                proOptions={proOptions}
            >
            <Controls />
            <Background variant="dots" gap={12} size={1} />
            </ReactFlow>
        </ReactFlowProvider>
    </Box>

    )}

export default NodeEditor