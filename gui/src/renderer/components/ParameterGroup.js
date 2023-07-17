import * as React from 'react'
import { useMethodDispatch } from '../logic/context';

import { Box, Grid, Stack, Typography, ButtonBase, Collapse } from '@mui/material'

import ArrowRightIcon from '@mui/icons-material/ArrowRight';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import ParameterInput from './ParameterInput';


const ParameterGroup = ({parameterGroup, sx}) => {

    const [open, setOpen] = React.useState(!parameterGroup.hidden);

    const dispatch = useMethodDispatch();

    // make Grid which takes 100% of the height
    // The last row should grow to fill the remaining space
    return (
        <Box sx={[
            {
            padding: 1,
            border: 1,
            borderRadius: "4px",
            borderColor: "grey.300",
          },
            ...(Array.isArray(sx) ? sx : [sx]),
            ]}
            >   
                <Grid container spacing={0}>
                    <Grid item xs={12} sx={{paddingBottom: 1}}>
                        <ButtonBase onClick={() => {setOpen(!open)}}>
                            <Stack direction="row" alignItems="center" spacing={0}>
                            {
                                open ?
                                <ArrowRightIcon/>
                                :
                                <ArrowDropDownIcon/>
                            }
                            <Typography component="span" sx={{fontWeight: 500, fontSize: "12px"}}>
                                {parameterGroup.name}
                            </Typography>
                            </Stack>
                        </ButtonBase>
                    </Grid>
                    <Collapse in={open} sx={{width: "100%"}} >
                    {parameterGroup.parameters.map((parameter, index) => {
                        return (
                            <Grid item xs={12}>
                                <ParameterInput
                                    parameter = {parameter}
                                    onChange = {(value) => {dispatch(
                                        {type: 'updateParameter', 
                                        parameterGroupId: parameterGroup.id,
                                        parameterId: parameter.id,
                                        value: value
                                        })}}
                                /> 
                            </Grid>
                        )
                    })}
                    </Collapse>
                </Grid>
                
        </Box>  
    )
}

export default ParameterGroup
