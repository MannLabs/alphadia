import { Grid, Stack, Typography, Tooltip, Button } from "@mui/material";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";

const MultiSelect = ({
    label = "File",
    active = true,
    type = "file",
    path = [],
    tooltipText = "",
    onChange = () => {},
    ...props
}) => {

    // set color based on activity status
    const color = active ? "primary" : "divider";
    // get file name from path
    // eslint-disable-next-line no-useless-escape
    const fileNames = path.map((path) => path.replace(/^.*[\\\/]/, ''));

    // handle select button click

    let handleSelect = () => {}

    if (type === "folder") {
        handleSelect = () => {
            window.electronAPI.getMultipleFolders().then((folder) => {
                onChange(folder);
            }).catch((err) => {
                console.log(err);
            })
        }
    } else {
        handleSelect = () => {
            window.electronAPI.getMultipleFiles().then((file) => {
                onChange(file);
            }).catch((err) => {
                console.log(err);
            })
        }
    }

    return (
    <Grid container spacing={2} sx={{color, minHeight:"100px"}} wrap="nowrap">
        <Grid item xs={3}>
            <Stack direction="row" alignItems="center" gap={1}>
                <Typography component="span">{label}</Typography>
                <Tooltip title = {active && tooltipText} sx= {{color}}>
                    <HelpOutlineIcon fontSize="small" />
                </Tooltip>
            </Stack>
        </Grid>
        <Grid item xs={6} zeroMinWidth sx={{overflow: 'hidden', textOverflow: 'ellipsis'}}>
            <Typography component="span" noWrap >
                {
                    // iterate fileNames and display with line break
                    fileNames.map((fileName) => {
                        return (
                            <Typography component="span" noWrap key={fileName}>
                                {fileName}
                                <br/>
                            </Typography>
                        )
                    })
                }
            </Typography>

        </Grid>
        <Grid item xs={3} position={'relative'}>
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

export default MultiSelect;