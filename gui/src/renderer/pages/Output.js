import * as React from 'react'

import { Box } from '@mui/system'
import { useMethod, useMethodDispatch } from '../logic/context'
import { SingleSelect } from '../components'

const Output = () => {

    const method  = useMethod();
    const dispatch = useMethodDispatch();
    return (
    <Box>
        <SingleSelect 
            type="folder"
            label="Output Folder"
            active={method.output_directory.active}
            path={method.output_directory.path}
            tooltipText="Select the folder where you would like to save the output."
            onChange={(path) => {dispatch({type: 'updateOutput', path: path})}}
        />
    </Box>

    )}

export default Output