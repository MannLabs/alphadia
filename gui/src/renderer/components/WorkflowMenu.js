import * as React from 'react'
import { useTheme } from '@mui/material/styles';
import { usePopupState, bindTrigger, bindMenu } from 'material-ui-popup-state/hooks'

import { Box, Typography, ButtonBase, Menu, MenuItem, ListItemIcon, Divider } from '@mui/material'
import MenuBookIcon from '@mui/icons-material/MenuBook';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown'
import ArrowRightIcon from '@mui/icons-material/ArrowRight';

import StyledLink from './StyledLink';

const WorkflowMenu = ({
    sx = [],
    workflows=[],
    currentWorkflow="",
    onWorkflowChange=() => {},
}) => {

    const popupState = usePopupState({ variant: 'popover', popupId: 'demoMenu' })
    const theme = useTheme();

    return (
        <Box display="flex" flexDirection="row" alignItems="center" padding={1}
        sx={[
            {

            },
            ...(Array.isArray(sx) ? sx : [sx]),
            ]}
        >
                <ButtonBase {...bindTrigger(popupState)}>
                    {
                        popupState.isOpen ?
                        <ArrowRightIcon sx={{color: theme.palette.success.main}}/>
                        :
                        <ArrowDropDownIcon sx={{color: theme.palette.success.main}}/>
                    }
                    <Typography component="div" sx={{color: theme.palette.success.main, fontFamily:"Roboto Mono", fontSize:"0.8rem"}}>
                        {currentWorkflow !== "" ? currentWorkflow : "Select Workflow"}
                    </Typography>
                </ButtonBase>
                <Menu
                    {...bindMenu(popupState)}
                    anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
                    transformOrigin={{ vertical: 'top', horizontal: 'left' }}
                    dense="true"
                    sx={{
                        fontFamily: "Roboto Mono",
                        fontSize: "0.8rem",
                        minHeight: 0,
                    }}
                >
                    {workflows.map((workflowName, index) => (
                        <MenuItem
                            key={`workflow-${workflowName}`}
                            onClick={() => {onWorkflowChange(workflowName); popupState.close()}}
                            sx={{color: theme.palette.success.main}}
                        >
                            {workflowName}
                        </MenuItem>
                    ))}

                    {/*TODO create workflow documentation and add link here*/}
                    {/*<Divider />*/}
                    {/*<MenuItem onClick={popupState.close} sx={{fontFamily:"Roboto Mono", fontSize:"0.8rem"}} key={999} >*/}
                    {/*    <ListItemIcon>*/}
                    {/*        <MenuBookIcon fontSize="small" />*/}
                    {/*    </ListItemIcon>*/}
                    {/*    <StyledLink  onClick={() => window.electronAPI.openLink("http://www.google.com")} >*/}
                    {/*        Workflow Documentation*/}
                    {/*    </StyledLink>*/}
                    {/*</MenuItem>*/}
                </Menu>
        </Box>
    )
}

export default WorkflowMenu
