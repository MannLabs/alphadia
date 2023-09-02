import { Grid, Stack, Typography, Tooltip, Button } from "@mui/material";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";

const InputFileSelect = ({
    active = true,
    onChange = () => {},
    ...props
    }) => {

    const color = active ? "primary" : "divider";

    const handleDSelect = () => {
        window.electronAPI.getMultipleFolders().then((files) => {
            onChange(files);
        }).catch((err) => {
            console.log(err);
        })
    }
    const handleRawSelect = () => {
        window.electronAPI.getMultipleFiles().then((files) => {
            onChange(files);
        }).catch((err) => {
            console.log(err);
        })
    }

    const tooltipText = "Select the input files which you would like to use.";

    return (
    <Grid container spacing={2} sx={{color}} wrap="nowrap">
        <Grid item xs={6}>
            <Stack direction="row" alignItems="center" gap={1}>
                <Typography component="span">Input Files</Typography>
                <Tooltip title = {active && tooltipText} sx= {{color}}>
                    <HelpOutlineIcon fontSize="small" />
                </Tooltip>
            </Stack>
        </Grid>
        <Grid item xs={6} position={'relative'}>
            <Button 
                variant="outlined" 
                sx={{float: 'right', ml:1, minWidth: "115px", textTransform: 'none', minWidth: "50px"}} 
                disabled={!active}
                onClick={handleRawSelect}>
                .raw
            </Button>
            <Button 
                variant="outlined" 
                sx={{float: 'right', ml:1, minWidth: "115px", textTransform: 'none', minWidth: "50px"}}  
                disabled={!active}
                onClick={handleDSelect}>
                .d
            </Button>
        </Grid>
    </Grid>
)}

export default InputFileSelect;