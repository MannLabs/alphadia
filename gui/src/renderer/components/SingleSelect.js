import { Grid, Stack, Typography, Tooltip, Button, Chip } from "@mui/material";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";

const SingleSelect = ({
    label = "File",
    active = true,
    type = "file",
    path = "",
    tooltipText = "",
    onChange = () => {},
    ...props
}) => {

    // set color based on activity status
    const color = active ? "primary" : "divider";
    // get file name from path
    // eslint-disable-next-line no-useless-escape
    const fileName = path.replace(/^.*[\\\/]/, '')
    // handle select button click

    let handleSelect = () => {}

    if (type === "folder") {
        handleSelect = () => {
            window.electronAPI.getSingleFolder().then((folder) => {
                onChange(folder);
            }).catch((err) => {
                console.log(err);
            })
        }
    } else {
        handleSelect = () => {
            window.electronAPI.getSingleFile().then((file) => {
                onChange(file);
            }).catch((err) => {
                console.log(err);
            })
        }
    }

    return (
    <Grid container spacing={2} sx={{color}} wrap="nowrap">
        <Grid item xs={3}>
            <Stack direction="row" alignItems="center" gap={1}>
                <Typography component="span">{label}</Typography>
                <Tooltip title = {active && tooltipText} sx= {{color}}>
                    <HelpOutlineIcon fontSize="small" />
                </Tooltip>
            </Stack>
        </Grid>
        <Grid item xs={6} zeroMinWidth sx={{overflow: 'hidden', textOverflow: 'ellipsis'}}>

            {fileName == ''? '':
                <Chip

                label={fileName}
                size="small"
                onDelete={() => {onChange("")}}
                />
            }

        </Grid>
        <Grid item xs={3} position={'relative'}>
            <Button
                variant="outlined"
                sx={{float: 'right', ml:1, minWidth: "130px"}}
                disabled={!active}
                onClick={handleSelect}>
                Select {type == "folder" ? "Folder" : "File"}
            </Button>
        </Grid>
    </Grid>
)}

export default SingleSelect;
