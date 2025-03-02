import * as React from 'react';
import styled from '@emotion/styled';
import { useMethod, useMethodDispatch } from '../logic/context';
import { Box, Typography } from '@mui/material';
import { Masonry } from '@mui/lab';
import { SingleSelect, MultiSelect, InputFileSelect, FileViewer, ParameterGroup } from '../components';
import WorkflowOverview from '../components/WorkflowOverview';
import { useSearchParams } from 'react-router-dom';

const FullWidthBox = styled(Box)(({ theme }) => ({
    width: '100%',
    marginBottom: '8px'
}))

const Files = () => {
    const method = useMethod();
    const dispatch = useMethodDispatch();

    return (
        <Box sx={{
            display: 'flex',
            flexDirection: 'column',
            height: '100%',

        }}>
            <Box sx={{ flex: '0 0 auto' }}>
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
            </Box>
            <FullWidthBox sx={{ flex: '1 1 auto'}}>
                <FileViewer
                    path={method.raw_path_list.path}
                    onChange={(path) => {dispatch({type: 'updateFiles', path: path})}}
                />
            </FullWidthBox>
        </Box>
    )
}

const Config = () => {
    const method = useMethod();
    return (
        <Box>
            <Masonry columns={{ xs: 1, sm: 2, md: 2, lg: 3, xl: 3 }} spacing={1}>
                {method.config.map((parameterGroup, index) => (
                <ParameterGroup
                    parameterGroup={parameterGroup}
                    index={index}
                />
                ))}
            </Masonry>
        </Box>
    )
}

const Method = () => {
    const [searchParams] = useSearchParams();
    const tab = searchParams.get('tab');

    return (
        <Box sx={{
            display: 'flex',
            flexDirection: 'column',
            height: '100%'
        }}>
            <Box sx={{ flex: '0 0 auto' }}>
                <WorkflowOverview />
            </Box>
            <Box sx={{ flex: '1 1 auto' }}>
                {tab === 'files' && <Files />}
                {tab === 'config' && <Config />}
            </Box>
        </Box>
    )
}

export default Method
