import * as React from 'react'
import { useMethodDispatch } from '../logic/context';

import { Box, Grid, Stack, Typography, ButtonBase, Collapse } from '@mui/material'

import ArrowRightIcon from '@mui/icons-material/ArrowRight';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import ParameterInput from './ParameterInput';


const ParameterGroup = ({parameterGroup, sx}) => {
    const [advancedOpen, setAdvancedOpen] = React.useState(false);
    const dispatch = useMethodDispatch();

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
                    <Stack direction="row" alignItems="center" spacing={0}>
                        <Typography component="span" sx={{fontWeight: 500, fontSize: "12px"}}>
                            {parameterGroup.name}
                        </Typography>
                    </Stack>
                </Grid>
                {parameterGroup.parameters
                    .filter(parameter => !parameter.hidden)
                    .map((parameter, index) => {
                    return (
                        <Grid item xs={12} key={parameter.id}>
                            <ParameterInput
                                parameter = {parameter}
                                parameter_group_id = {parameterGroup.id}
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

                {/* Advanced parameters section (if there are any non-hidden parameters) */}
                {parameterGroup.parameters_advanced &&
                 parameterGroup.parameters_advanced.filter(parameter => !parameter.hidden).length > 0 && (
                    <Grid item xs={12} sx={{mt: 1}}>
                        <ButtonBase onClick={() => {setAdvancedOpen(!advancedOpen)}}>
                            <Stack direction="row" alignItems="center" spacing={0}>
                                {advancedOpen ? <ArrowDropDownIcon/> : <ArrowRightIcon/>}
                                <Typography component="span" sx={{fontWeight: 500, fontSize: "12px", color: "text.secondary"}}>
                                    Advanced Parameters
                                </Typography>
                            </Stack>
                        </ButtonBase>
                        <Collapse in={advancedOpen} sx={{width: "100%"}}>
                            {parameterGroup.parameters_advanced
                                .filter(parameter => !parameter.hidden)
                                .map((parameter, index) => (
                                <Grid item xs={12} key={parameter.id}>
                                    <ParameterInput
                                        parameter={parameter}
                                        parameter_group_id={parameterGroup.id}
                                        onChange={(value) => {dispatch({
                                            type: 'updateParameterAdvanced',
                                            parameterGroupId: parameterGroup.id,
                                            parameterId: parameter.id,
                                            value: value
                                        })}}
                                    />
                                </Grid>
                            ))}
                        </Collapse>
                    </Grid>
                )}
            </Grid>
        </Box>
    )
}

export default ParameterGroup
