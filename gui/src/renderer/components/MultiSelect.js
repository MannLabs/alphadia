import { Grid, Stack, Typography, Tooltip, Button, Chip } from "@mui/material";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";
import { useTheme } from '@emotion/react';

const MultiSelect = ({
    label = "File",
    active = true,
    type = "file",
    path = [],
    tooltipText = "",
    onChange = () => {},
    ...props
}) => {

    const theme = useTheme();

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
            {
                // iterate fileNames and display with line break
                fileNames.map((fileName, i) => {
                    return (
                        <Chip
                        sx={{marginRight: theme.spacing(0.5), marginBottom: theme.spacing(0.5)}}
                        label={fileName}
                        key={i}
                        size="small"
                        onDelete={
                            () => {
                                let newPaths = path.slice();
                                newPaths.splice(i, 1);
                                onChange(newPaths);
                            }
                        }
                        />
                    )
                })
            }
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
