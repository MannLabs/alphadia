import { Grid, Stack, Typography, Tooltip, Button } from "@mui/material";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";

const MultipleFileSelect = ({
    label = "File",
    active = true,
    tooltipText = "",
    onChange = () => {},
    ...props
    }) => {

    const color = active ? "primary" : "divider";

    const handleSelect = () => {
        window.electronAPI.getMultiple().then((files) => {
            onChange(files);
        }).catch((err) => {
            console.log(err);
        })
    }

    return (
    <Grid container spacing={2} sx={{color}} wrap="nowrap">
        <Grid item xs={6}>
            <Stack direction="row" alignItems="center" gap={1}>
                <Typography component="span">{label}</Typography>
                <Tooltip title = {active && tooltipText} sx= {{color}}>
                    <HelpOutlineIcon fontSize="small" />
                </Tooltip>
            </Stack>
        </Grid>
        <Grid item xs={6} position={'relative'}>
            <Button 
                variant="outlined" 
                sx={{float: 'right', ml:1, minWidth: "115px"}} 
                disabled={!active}
                onClick={handleSelect}>
                Select Files
            </Button>
        </Grid>
    </Grid>
)}

export default MultipleFileSelect;