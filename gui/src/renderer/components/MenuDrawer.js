import * as React from 'react'
import { styled } from '@mui/system';
import { Button } from '@mui/material';
import { Drawer, List, ListItemIcon, ListItemText, ListItemButton, ListSubheader, Box} from '@mui/material';
import { WorkflowMenu, RunButton } from '.';
import { useNavigate, useLocation } from 'react-router-dom';

import HomeIcon from '@mui/icons-material/Home';
import FolderIcon from '@mui/icons-material/Folder';
import SettingsApplicationsIcon from '@mui/icons-material/SettingsApplications';
import SaveIcon from '@mui/icons-material/Save';
import TerminalIcon from '@mui/icons-material/Terminal';

const DrawerContainer = styled('div')(({ theme }) => ({
    width: 240,
    flexShrink: 0,

    '& .MuiPaper-root': {
      backgroundColor: theme.palette.background.default,
      width: 240,
      boxSizing: 'border-box',

    },
  }));

// modify style for ListItemButton
const ListStyled = styled(List)(({ theme }) => ({
    paddingTop: 0,
    paddingBottom: 0,

    ' .MuiList-root': {
        paddingTop: 0,
        paddingBottom: 0,
    },

    ' .MuiListSubheader-root': {
        lineHeight: 1.5,
        paddingTop: theme.spacing(1),
    }
}));

const ListItemButtonStyled = styled(ListItemButton)(({ theme }) => ({
        paddingTop: 4,
        paddingBottom: 4,
        margin: theme.spacing(1),
        borderRadius: 8,

        '&:hover': {
            backgroundColor: theme.palette.primary.hover,
            color: theme.palette.primary.main,
        },

        '&.Mui-selected': {
            backgroundColor: theme.palette.primary.selected,
            color: theme.palette.primary.main,
            '&:hover': {
                backgroundColor: theme.palette.primary.hover,
            },
        }
}));

const MenuDrawer = ({
    workflows=[],
    currentWorkflow="",
    onWorkflowChange,
    onSetRunningState,
    profile
}) => {
    const navigate = useNavigate();
    const location = useLocation();

    // Add function to check if a path is active
    const isPathActive = (path) => {
        const currentPath = location.pathname;
        const currentSearch = new URLSearchParams(location.search);
        const currentTab = currentSearch.get('tab');

        if (path === '/') {
            return currentPath === '/';
        }

        if (path.includes('?')) {
            const [pathBase, search] = path.split('?');
            const pathParams = new URLSearchParams(search);
            const pathTab = pathParams.get('tab');
            return currentPath === pathBase && currentTab === pathTab;
        }

        return currentPath === path;
    };

    return (
    <DrawerContainer>
        <Drawer
            variant="permanent"
            sx={{
              width: 240,
              flexShrink: 0,
            }}
        >
        <Box sx={{ minHeight: '40px', px:1, display: "flex"}}>
            <WorkflowMenu
                workflows={workflows}
                currentWorkflow={currentWorkflow}
                onWorkflowChange={onWorkflowChange}
            />
        </Box>

        <ListStyled>
            <ListItemButtonStyled
                key={"home"}
                onClick={() => navigate('/')}
                selected={isPathActive('/')}
            >
                <ListItemIcon>
                    <HomeIcon />
                </ListItemIcon>
                <ListItemText primary="Home" />
            </ListItemButtonStyled>

            <ListSubheader component="div" sx={{backgroundColor: "transparent"}}>
                Method Setup
            </ListSubheader>

            <ListItemButtonStyled
                key={"files"}
                onClick={() => navigate('/method?tab=files')}
                selected={isPathActive('/method?tab=files')}
            >
                <ListItemIcon>
                    <FolderIcon />
                </ListItemIcon>
                <ListItemText primary="Input & Output" />
            </ListItemButtonStyled>

            <ListItemButtonStyled
                key={"method"}
                onClick={() => navigate('/method?tab=config')}
                selected={isPathActive('/method?tab=config')}
            >
                <ListItemIcon>
                    <SettingsApplicationsIcon />
                </ListItemIcon>
                <ListItemText primary="Method Settings" />
            </ListItemButtonStyled>
        </ListStyled>

        <List sx={{padding: 0, position: "absolute", bottom: 0, width: "100%"}}>
            <ListItemButtonStyled
                key={"console"}
                onClick={() => navigate('/run')}
                selected={isPathActive('/run')}
            >
                <ListItemIcon>
                    <TerminalIcon />
                </ListItemIcon>
                <ListItemText primary="Console Output" />
            </ListItemButtonStyled>

            <RunButton
                onSetRunningState={onSetRunningState}
                profile={profile}
            />
        </List>
        </Drawer>
    </DrawerContainer>
    );
};

export default MenuDrawer
