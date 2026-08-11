import * as React from 'react'
import styled from '@emotion/styled'
import { useMethod, useMethodDispatch } from '../logic/context'

import { Box } from '@mui/material'
import { SingleSelect, MultiSelect, InputFileSelect, FileViewer } from '../components'

const FullWidthBox = styled(Box)(({ theme }) => ({
    width: '100%'
}))

// Leaves room for the grid's toolbar, header, footer and a few rows once the
// selectors above it have grown (the fasta selector grows with each file added).
const FILE_VIEWER_MIN_HEIGHT = 240

const Files = () => {

    const method  = useMethod();
    const dispatch = useMethodDispatch();

    return (
        <Box sx={{
            height: '100%',
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
                    <MultiSelect
                            label="Fasta File(s)"
                            active={method.fasta_list.active}
                            path={method.fasta_list.path}
                            tooltipText="Select the fasta file which you would like to use."
                            onAppend={(path) => {dispatch({type: 'appendFasta', path: path})}}
                            onChange={(path) => {dispatch({type: 'updateFasta', path: path})}}
                    />
                </FullWidthBox>
                <FullWidthBox>
                    <InputFileSelect
                            active={method.raw_path_list.active}
                            path={method.raw_path_list.path}
                            onChange={(path) => {dispatch({type: 'appendFiles', path: path})}}
                    />
                </FullWidthBox>
                <FullWidthBox sx={{flexGrow: 1, minHeight: FILE_VIEWER_MIN_HEIGHT, overflow: 'hidden'}}>
                    <FileViewer
                            path={method.raw_path_list.path}
                            onChange={(path) => {dispatch({type: 'updateFiles', path: path})}}/>
                </FullWidthBox>
                <FullWidthBox>
                    <SingleSelect
                        type="folder"
                        label="Output Folder"
                        active={method.output_directory.active}
                        path={method.output_directory.path}
                        tooltipText="Select the folder where you would like to save the output."
                        onChange={(path) => {dispatch({type: 'updateOutput', path: path})}}
                    />
                </FullWidthBox>
        </Box>
    )
}

export default Files
