import * as React from 'react';
import { Routes, Route } from "react-router-dom"
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { useMediaQuery } from '@mui/material';

import getDesignTokens from './theme';
import { useMethodDispatch } from './logic/context';
import styled from '@emotion/styled';
import './App.css';

import { Box, CssBaseline } from '@mui/material';
import { MenuDrawer, UtilMonitor, ExecutionEngine } from './components';
import { Home, Files, Method, Output, Run } from './pages';



// Define the styles for the layout
const AppLayout = styled('div')(({ theme }) => ({
  display: 'flex',
  height: "100%"
}));

const App = () => {

    const mode = useMediaQuery('(prefers-color-scheme: dark)') ? 'dark' : 'light';
    const [modeState, setMode] = React.useState(mode );
    const theme = createTheme(getDesignTokens( modeState ))

    const dispatch = useMethodDispatch();

    const [profile, setProfile] = React.useState({
        workflows: [],
        currentWorkflow: null,
        environment: {},
        running: false,
    });

    function handleSetRunningState(isRunning) {
        setProfile((profile) => ({...profile, running: isRunning}));
    }

    React.useEffect(() => {
        window.electronAPI.getWorkflows().then((result) => {
            if (result.length === 0) {
                alert("No workflows found. Please create a new workflow.")
            } else {
                setProfile((profile) => {
                    return {
                        ...profile,
                        workflows: result,
                        currentWorkflow: result[0].name
                    }
                });
                dispatch({
                    type: 'updateWorkflow',
                    workflow: result[0]
                });
            }
        }).catch((error) => {
            alert(error);
        });

        window.electronAPI.onThemeChange((_event, value) => {
            setMode(value ? 'dark' : 'light');
        })

        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const handleWorkflowChange = (workflowName) => {
        console.log(workflowName);
        setProfile({
            ...profile,
            currentWorkflow: workflowName
        });

        const activeWorkflow = profile.workflows.find((workflow) => workflow.name === workflowName);
        dispatch({
            type: 'updateWorkflow',
            workflow: activeWorkflow
        });
    }

    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <AppLayout>
        <MenuDrawer
            workflows={profile.workflows.map((workflow) => workflow.name)}
            currentWorkflow={profile.currentWorkflow}
            onWorkflowChange={handleWorkflowChange}
            onSetRunningState={handleSetRunningState}
            profile={profile}
            />
        <Box sx={{flexGrow: 1, overflowX: 'hidden', display: 'flex', flexDirection: 'column', height: '100%'}}>
            <Box sx={{
                height: '40px',
                minHeight: '40px',
                flexShrink: 0,
                display: 'flex',
                alignItems: 'center',
                paddingLeft: theme.spacing(1),
                paddingRight: theme.spacing(1)
            }}>
                <ExecutionEngine environment={profile.environment} sx={{ marginRight: "auto"}}/>
                <UtilMonitor sx={{ marginLeft: "auto"}}/>
            </Box>
            <Box sx={{
                flexGrow: 1,
                paddingLeft: theme.spacing(1),
                paddingRight: theme.spacing(1),
                overflow: 'auto'
            }}>

                <Routes>
                    <Route path="/" element={ <Home /> } />
                    <Route path="/files" element={ <Files/> } />
                    <Route path="/method" element={ <Method/> } />
                    <Route path="/output" element={ <Output/> } />
                    <Route path="/run" element={ <Run/> } />
                </Routes>
            </Box>
        </Box>
        </AppLayout>
    </ThemeProvider>
  );
};

export default App;
