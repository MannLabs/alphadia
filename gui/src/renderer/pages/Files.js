import * as React from 'react'
import styled from '@emotion/styled'
import { useMethod, useMethodDispatch } from '../logic/context'

import { Box } from '@mui/material'
import { SingleSelect, InputFileSelect, FileViewer } from '../components'

const FullWidthBox = styled(Box)(({ theme }) => ({
    width: '100%'
}))

const Files = () => {

    const method  = useMethod();
    const dispatch = useMethodDispatch();

    return (
        <Box sx={{
            height: 'calc(100% - 64px)',
            display: 'flex',
            flexDirection: 'column',
            gap: 1
        }}>
                <FullWidthBox>
                    <SingleSelect
                        label="Spectral Library"
                        active={method.library.active}
                        path={method.library.path}
                        tooltipText="Select the spectral library which you would like to use."
                        onChange={(path) => {dispatch({type: 'updateLibrary', path: path})}}
                    />
                </FullWidthBox>
                <FullWidthBox>
                    <SingleSelect 
                            label="Fasta File"
                            active={method.fasta.active}
                            path={method.fasta.path}
                            tooltipText="Select the fasta file which you would like to use."
                            onChange={(path) => {dispatch({type: 'updateFasta', path: path})}}
                    />
                </FullWidthBox>
                <FullWidthBox>
                    <InputFileSelect
                            active={method.files.active}
                            path={method.files.path}
                            onChange={(path) => {dispatch({type: 'updateFiles', path: path})}}
                    />
                </FullWidthBox>
                <FullWidthBox sx={{flexGrow: 1}}>
                    <FileViewer path={method.files.path}/>
                </FullWidthBox>
        </Box>  
    )
}

export default Files
