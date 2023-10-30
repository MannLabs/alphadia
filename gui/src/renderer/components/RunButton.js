import { useTheme } from "@mui/material";
import { useMethod } from "../logic/context";
import { useProfile } from "../logic/profile";
import { useNavigate } from "react-router-dom";

import { ListItemButton, ListItemIcon, ListItemText } from "@mui/material";

import DoubleArrowIcon from '@mui/icons-material/DoubleArrow';
import CircularProgress from '@mui/material/CircularProgress';
import CloseIcon from '@mui/icons-material/Close';

const validateMethod = (method) => {
    if (method.library.required && (method.library.path === "")) {
        return {valid: false, message: "Please select a spectral library."}
    }
    if (method.fasta.required && (method.fasta.path === "")) {
        return {valid: false, message: "Please select a fasta file."}
    }
    if (method.files.required && (method.files.path.length === 0)) {
        return {valid: false, message: "Please select at least one Bruker .d folder."}
    }
    return {valid: true, message: ""}
}

const buttonStyle = (theme) => ({
    paddingTop: 0.5,
    paddingBottom: 0.5,
    margin: 1,
    borderRadius: "8px",
    fontWeight: 500,
})

const runActiveStyle = (theme) => ({
    background:"linear-gradient(215deg, rgb(130 212 91) 0%, rgb(51, 222, 67) 35%, rgb(24 149 82) 100%)",
    filter: theme.palette.mode === 'light' ? "drop-shadow(1px 3px 5px rgba(159, 255, 189, 1))" : "none",
    color: "#fff",
})

const abortActiveStyle = (theme) => ({
    background:"linear-gradient(215deg, rgb(212 91 91) 0%, rgb(222 67 67) 35%, rgb(148 50 50) 100%)",
    filter: theme.palette.mode === 'light' ? "drop-shadow(1px 3px 5px rgba(255 189 189, 1))" : "none",
    color: "#fff",
})

const inactiveStyle = (theme) => ({
    backgroundColor: theme.palette.divider,
    color: theme.palette.text.main,
    '&:hover': {
        backgroundColor: theme.palette.divider,
        color: theme.palette.text.main,
    }
})

const RunButton = ({
    parameter,
    onSetRunningState,
    profile
}) => {

    const navigate = useNavigate();
    const method = useMethod();
    const profileNew = useProfile();
    const theme = useTheme();

    const runStyle = profile.running ? {...buttonStyle(theme), ...inactiveStyle(theme)} : {...buttonStyle(theme), ...runActiveStyle(theme)}
    const abortStyle = profile.running ? {...buttonStyle(theme), ...abortActiveStyle(theme)} : {...buttonStyle(theme), ...inactiveStyle(theme)}

    const runIcon = profile.running ? <CircularProgress color="inherit"size={20} /> : <DoubleArrowIcon />
    const runText = profile.running ? "Running" : "Run Workflow"

    function handleRunClick() {
        // check if workflow is already running
        if (profile.running) {
            return;
        }
        
        // check if method is valid
        const validation = validateMethod(method);
        if (!validation.valid) {
            alert(validation.message);     
            return;
        }

        // check if valid engine has been selected
        if (profileNew.activeIdx < 0) {
            alert("Please select a valid execution engine.");
            return;
        }
        if (! profileNew.executionEngines[profileNew.activeIdx].available) {
            alert("The selected execution engine is not available.");
            return;
        }

        onSetRunningState(true)
        navigate("/run");
        window.electronAPI.startWorkflowNew(method, profileNew.activeIdx).then((result) => {
            onSetRunningState(false)
        })
    }

    function handleAbortClick() {
        onSetRunningState(!profile.running)
        window.electronAPI.abortWorkflowNew(-1);
    }

    return (
        <>
        <ListItemButton
            key="run"
            sx={runStyle}
            onClick={handleRunClick}
            >
            <ListItemIcon>
                {runIcon}
            </ListItemIcon>
            <ListItemText primary={runText} />
        </ListItemButton>
        <ListItemButton
            key="abort"
            disabled={!profile.running}
            sx={abortStyle}
            onClick={handleAbortClick}
            >
            <ListItemIcon>
                <CloseIcon />
            </ListItemIcon>
            <ListItemText primary="Abort" />
        </ListItemButton>
        </>
    )
}

export default RunButton;