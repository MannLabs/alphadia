import * as React from 'react'
import { NavLink as NavLinkBase } from "react-router-dom"
import { styled } from '@mui/system';

import { Drawer, List, ListItemIcon, ListItemText, ListItemButton, ListSubheader, Box} from '@mui/material';
import { WorkflowMenu, RunButton } from '.';

import HomeIcon from '@mui/icons-material/Home';
import FolderIcon from '@mui/icons-material/Folder';
import SettingsApplicationsIcon from '@mui/icons-material/SettingsApplications';
import SaveIcon from '@mui/icons-material/Save';
import TerminalIcon from '@mui/icons-material/Terminal';
import SchemaIcon from '@mui/icons-material/Schema';

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
        },
}));

const NavLink = React.forwardRef((props, ref) => (
    <NavLinkBase
      ref={ref}
      to={props.to}
      className={({ isActive }) => `${props.className} ${isActive ? props.activeClassName : ''}`}
    >
      {props.children}
    </NavLinkBase>
  ));

const MenuDrawer = ({
    workflows=[],
    currentWorkflow="",
    onWorkflowChange,
    onSetRunningState,
    profile
}) => (
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
                component={NavLink} 
                to="/" 
                activeClassName="Mui-selected">
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
                component={NavLink} 
                to="/files" 
                activeClassName="Mui-selected" >
                <ListItemIcon>
                    <FolderIcon />
                </ListItemIcon>
                <ListItemText primary="Input Files" />
            </ListItemButtonStyled >
            <ListItemButtonStyled 
                key={"node-editor"}
                component={NavLink}
                to="/node-editor"
                activeClassName="Mui-selected" >
                <ListItemIcon>
                    <SchemaIcon />
                </ListItemIcon>
            <ListItemText primary="Node Editor" />
            </ListItemButtonStyled>
            <ListItemButtonStyled 
                key={"method"}
                component={NavLink}
                to="/method"
                activeClassName="Mui-selected" >
                <ListItemIcon>
                    <SettingsApplicationsIcon />
                </ListItemIcon>
            <ListItemText primary="Method Settings" />
            </ListItemButtonStyled>
            <ListItemButtonStyled 
                key={"output"}
                component={NavLink}
                to="/output"
                activeClassName="Mui-selected" >
                <ListItemIcon>
                    <SaveIcon />
                </ListItemIcon>
                <ListItemText primary="Output Files" />
            </ListItemButtonStyled>
            
        </ListStyled>
        <List sx={{padding: 0, position: "absolute", bottom: 0, width: "100%"}}>
            <ListItemButtonStyled 
                key={"console"}
                component={NavLink} 
                to="/run" 
                activeClassName="Mui-selected" >
                <ListItemIcon>
                    <TerminalIcon />
                </ListItemIcon>
                <ListItemText primary="Console Output" />
            </ListItemButtonStyled >
            <RunButton
                onSetRunningState={onSetRunningState}
                profile={profile}
            />
        </List>
        </Drawer>
    </DrawerContainer>
)

export default MenuDrawer