import { useCallback } from 'react';
import { Handle, Position } from 'reactflow';
import { Box } from '@mui/system';

import { Typography, Grid, Stack, Tooltip, TextField, IconButton } from '@mui/material';
import { useTheme } from '@emotion/react';
import ClearIcon from '@mui/icons-material/Clear';
import { useReactFlow } from 'reactflow';

import { ParameterInput } from '../components';

function AlphaBaseHandle(props) {
    const theme = useTheme();

    const edgeId = props.id
    const handleColor = theme.palette.node.edgeColor[edgeId] || theme.palette.divider

    const newProps = {
        ...props,
        style: {
            ...props.style,
            background: handleColor,
            width: "8px",
            height: "8px",
        }
    }

    return (

        <Handle {...newProps} />

    )
}


//<AlphaBaseHandle type="target" position={Position.Left}  style={{top: "20px"}} id='fasta'/>
//<AlphaBaseHandle type="target" position={Position.Left}  style={{top: "40px"}} id='speclib'/>
//<AlphaBaseHandle type="source" position={Position.Right}  style={{bottom: "20px", top: "auto"}} id='fasta'/>
//<AlphaBaseHandle type="source" position={Position.Right}  style={{bottom: "40px", top: "auto"}} id='speclib'/>

function ParameterNode(props) {

    const reactFlow = useReactFlow();

    const nodeName = props.data.title;
    const theme = useTheme();

    const onChange = useCallback((evt) => {
    console.log(evt.target.value);
    }, []);

    console.log(props)



  return (
    <Box sx={{
            
            border: 1,
            borderRadius: 2,
            borderColor: theme.palette.divider,
            backgroundColor: theme.palette.node.background,
        }}
        >

        {props.data.targetHandles.map((handleId, index) => {
            return (<AlphaBaseHandle type="target" position={Position.Left}  style={{top: `${index*20+20}px`}} id={handleId}/>)
        })
        }
        {props.data.sourceHandles.map((handleId, index) => {
            return (<AlphaBaseHandle type="source" position={Position.Right}  style={{bottom: `${index*20+20}px`, top: "auto"}} id={handleId}/>)
        })
        }

        <Stack direction={"row"} justifyContent="space-between" alignItems="center" sx={{borderBottom: 1,  borderColor: theme.palette.divider}}>
            <Typography component="div" sx={{
                padding: 1,
                fontSize:"0.85rem", 
                fontFamily:"Roboto Mono",
                }}>
                {nodeName}
            </Typography>
            <IconButton aria-label="delete" disableRipple size="small" sx={{padding: 1}} onClick={() => reactFlow.deleteElements({nodes: [props]})}>
                <ClearIcon fontSize="0.85rem" />
            </IconButton>
        </Stack>

        <Box sx={{padding: 1}}>
       
        {props.data.parameters.map((parameter, index) => {
            return (
         
                    <ParameterInput
                        parameter = {parameter}
                    /> 
            
            )
        })}
   
        </Box>
        
    </Box>
  );
}

export default ParameterNode;