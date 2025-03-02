import React, { useState } from 'react';
import useTheme from '@mui/material/styles/useTheme';
import { useProfileDispatch, useProfile } from '../logic/profile';

import { Stack, Box, Typography, Popover, Grid } from '@mui/material';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import CheckIcon from '@mui/icons-material/Check';
import { usePopupState, bindTrigger, bindMenu } from 'material-ui-popup-state/hooks'

import { ButtonBase, Menu, MenuItem, ListItemIcon, Divider } from '@mui/material'
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown'
import ArrowRightIcon from '@mui/icons-material/ArrowRight';
import CircularProgress from '@mui/material/CircularProgress';


const ExecutionEngineItem = ({
    sx = [],
    children,
    key=0,
    active=true,
    available=true,
    onClick = () => {}
}) => {
    const theme = useTheme();

    return (
        <MenuItem
        key={key}
        disabled={!available}
        sx={{
            display: "flex",
            p:0,
            ...(Array.isArray(sx) ? sx : [sx]),
        }}
        onClick={onClick}
        >
            <Box sx={{
                width: 400,
                flexGrow: 1,
                paddingLeft: theme.spacing(2),
                paddingRight: theme.spacing(2),
                paddingTop: theme.spacing(1),
                paddingBottom: theme.spacing(1),
                borderLeft: "5px solid " + (active ? theme.palette.success.main : "transparent"),
                backgroundColor: active ? theme.palette.action.hover : "transparent",
            }}>
                {children}
            </Box>
        </MenuItem>
)}

const ExecutionEngine = ({ environment = {}, sx = []}) => {

    const popupState = usePopupState({ variant: 'popover', popupId: 'demoMenu' })
    const theme = useTheme();

    const profileDispatch = useProfileDispatch();
    const profile = useProfile();

    const [isLoading, setIsLoading] = useState(true);

    const activeName = profile.activeIdx >= 0 ? profile.executionEngines[profile.activeIdx].name : "None"

    React.useEffect(() => {
        window.electronAPI.getEngineStatus().then((result) => {
            profileDispatch({
                type: 'setExecutionEngines',
                executionEngines: result
            });
            profileDispatch({
                type: 'setExecutionEngineIdx',
                idx: result.findIndex((item) => item.available)
            });
            setIsLoading(false);

        }).catch((error) => {
            alert(error);
        });

    }, []);

    const menu = (
        <>
        <ButtonBase {...bindTrigger(popupState)}>
            {
                popupState.isOpen ? <ArrowDropDownIcon/> : <ArrowRightIcon/>
            }
            <Typography component="div" sx={{fontFamily:"Roboto Mono", fontSize:"0.85rem", fontWeight: 500}}>
                {activeName}
            </Typography>
        </ButtonBase>
        <Menu
        {...bindMenu(popupState)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
        transformOrigin={{ vertical: 'top', horizontal: 'left' }}
        sx={{
            fontFamily: "Roboto Mono",
            fontSize: "0.8rem",
            minHeight: 0,
            p: 0,
        }}>
            {profile.executionEngines.map((item, index) => {
                return (
                    <ExecutionEngineItem
                        key={index}
                        active={index == profile.activeIdx}
                        available={item.available}
                        onClick={() => {profileDispatch({type: 'setExecutionEngineIdx', idx: index})}}
                        >
                        <Grid container spacing={0}>
                            <Grid item xs={9}>
                                <Typography component="div" sx={{fontSize:"0.85rem", fontFamily:"Roboto Mono"}}>
                                    {item.name}
                                </Typography>
                            </Grid>
                            <Grid item xs={3}>
                                {item.available ?
                                    <Stack direction="row" spacing={0} alignItems="center" >
                                        <CheckIcon sx={{color: theme.palette.success.main, fontSize:"0.85rem", marginRight: theme.spacing(0.5)}}/>
                                        <Typography component="div" sx={{fontSize:"0.75rem", color: theme.palette.success.main}}>
                                            Available
                                        </Typography>

                                    </Stack> :
                                    <Stack direction="row" spacing={0} alignItems="center" >
                                        <ErrorOutlineIcon sx={{color: theme.palette.error.main, fontSize:"0.85rem", marginRight: theme.spacing(0.5)}}/>
                                        <Typography component="div" sx={{fontSize:"0.75rem", color: theme.palette.error.main}}>
                                            Not Available
                                        </Typography>
                                    </Stack>
                                }
                            </Grid>
                            {item.error === "" ? null : <Grid item xs={12}>
                                <Typography component="div" sx={{fontSize:"0.75rem", textWrap: "wrap", color: theme.palette.error.main}}>
                                    {item.error}
                                </Typography>
                            </Grid>}
                            <Grid item xs={12}>
                                <Typography component="div" sx={{fontSize:"0.75rem", textWrap: "wrap", paddingTop: theme.spacing(1), paddingBottom: theme.spacing(1)}}>
                                    {item.description}
                                </Typography>
                            </Grid>
                            <Grid item xs={12}>

                            {item.versions.map((version, index) => {
                                return (
                                    <Typography component="div" sx={{fontSize:"0.75rem", fontFamily:"Roboto Mono" }} key={index}>
                                        {version.name}: {version.version}
                                    </Typography>
                                )
                            })}
                            </Grid>
                        </Grid>
                    </ExecutionEngineItem>
                )}
            )}
        </Menu>
        </>
    );

  return (
    <Box
      sx={[
        { display: 'flex', alignItems: 'center', justifyContent: 'center' },
        ...(Array.isArray(sx) ? sx : [sx]),
      ]}
    >
        {isLoading ?
            <>
            <CircularProgress size={13} sx={{color: theme.palette.text.secondary, marginRight: theme.spacing(1)}}/>
            <Typography component="div" sx={{fontFamily:"Roboto Mono", fontSize:"0.85rem", fontWeight: 500}}>
                Loading...
            </Typography>
            </> : menu}
    </Box>
  );
};

export default ExecutionEngine;
