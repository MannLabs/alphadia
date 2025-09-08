import * as React from 'react'
import styled from '@emotion/styled'
import useTheme from '@mui/material/styles/useTheme';
import InfoTooltip from './InfoTooltip'

import { Box, Chip, Button, Checkbox, FormControl, MenuItem, Select, Stack, Typography, TextField } from '@mui/material'

const StyledCheckbox = styled(Checkbox)(({ theme }) => ({
    padding: 0.5
}))



const SingleFolderSelection = ({parameter, onChange = () => {}}) => {

    const handleSelect = () => {
        window.electronAPI.getSingleFolder().then((folder) => {
            onChange(folder);
        }).catch((err) => {
            console.log(err);
        })
    }
    const folderName = parameter ? parameter.replace(/^.*[\\\/]/, '') : ''

    return (
        <>
            {folderName == ''? '':
                <Chip
                label={folderName}
                onDelete={() => {onChange("")}}
                size="small"
                />
            }
            <Button
                variant="outlined"
                size="small"
                sx={{width: "150px"}}
                onClick={handleSelect}>
                Select Folder
            </Button>
        </>

    )
}

const ParameterInput = ({
        parameter,
        parameter_group_id,
        onChange = () => {},
        searchTerm = '',
        sx
    }) => {
        const theme = useTheme();

        // Check if parameter matches search term
        const isMatch = React.useMemo(() => {
            if (!searchTerm || searchTerm.trim() === '') return false;
            const search = searchTerm.toLowerCase();
            const nameMatch = parameter.name.toLowerCase().includes(search);
            const descriptionMatch = parameter.description && parameter.description.toLowerCase().includes(search);
            return nameMatch || descriptionMatch;
        }, [searchTerm, parameter.name, parameter.description]);

        let input = null;

        switch (parameter.type) {
            case "integer":
                input = (
                    <TextField
                    id="outlined-number"
                    type="number"
                    variant="standard"
                    size="small"
                    sx = {{width: "150px"}}
                    value={parameter.value}
                    onChange={(event) => {onChange(parseInt(event.target.value))}}
                    inputProps={{step: "1", lang:"en-US"}}
                />)
                break;
            case "float":
                input = (
                    <TextField
                    id="outlined-number"
                    type="number"
                    variant="standard"
                    size="small"
                    sx = {{width: "150px"}}
                    value={parameter.value}
                    onChange={(event) => {onChange(parseFloat(event.target.value))}}
                    inputProps={{step: "0.01", lang:"en-US"}}
                />)
                break;
            case "string":
                input = (
                    <TextField
                    id="outlined-number"
                    type="text"
                    variant="standard"
                    size="small"
                    sx = {{width: "150px"}}
                    value={parameter.value}
                    onChange={(event) => {onChange(event.target.value)}}
                />)
                break;
            case "boolean":
                input = (
                    <Box sx={{width: "150px"}}>
                        <StyledCheckbox
                            checked={parameter.value}
                            size='small'
                            onChange={(event) => {onChange(event.target.checked)}}
                            />
                    </Box>
                )
                break;
            case "dropdown":
                input = (
                    <FormControl variant="standard" size="small" sx={{width: "150px", minHeight: "0px"}}>
                        <Select
                            value={parameter.value}
                            onChange={(event) => {onChange(event.target.value)}}
                            >
                            {parameter.options.map((option) => {
                                return (
                                    <MenuItem value={option}>{option}</MenuItem>
                                )
                            })}
                        </Select>
                    </FormControl>
                )
                break;
            case "integer_range":
                input = (
                    <Box sx={{width: "150px"}}>
                    <TextField
                        id="outlined-number"
                        type="number"
                        variant="standard"
                        size="small"
                        sx = {{width: "70px"}}
                        value={parameter.value[0]}
                        onChange={(event) => {onChange([parseInt(event.target.value), parameter.value[1]])}}
                        inputProps={{step: "1", lang:"en-US"}}
                    />
                    <TextField
                        id="outlined-number"
                        type="number"
                        variant="standard"
                        size="small"
                        sx = {{width: "70px", marginLeft: "10px"}}
                        value={parameter.value[1]}
                        onChange={(event) => {onChange([parameter.value[0], parseInt(event.target.value)])}}
                        inputProps={{step: "1", lang:"en-US"}}
                    />
                    </Box>)
                break;

            case "float_range":
                input = (
                    <Box sx={{width: "150px"}}>
                    <TextField
                        id="outlined-number"
                        type="number"
                        variant="standard"
                        size="small"
                        sx = {{width: "70px"}}
                        value={parameter.value[0]}
                        onChange={(event) => {onChange([parseFloat(event.target.value), parameter.value[1]])}}
                        inputProps={{step: "0.01", lang:"en-US"}}
                    />
                    <TextField
                        id="outlined-number"
                        type="number"
                        variant="standard"
                        size="small"
                        sx = {{width: "70px", marginLeft: "10px"}}
                        value={parameter.value[1]}
                        onChange={(event) => {onChange([parameter.value[0], parseFloat(event.target.value)])}}
                        inputProps={{step: "0.01", lang:"en-US"}}
                    />
                    </Box>)
                break;
            case "singleFolderSelection":
                input = (
                    <SingleFolderSelection
                        parameter={parameter.value}
                        onChange={onChange}
                    />)
                break;

            case "multi_select":
                input = (
                    <FormControl variant="standard" size="small" sx={{width: "150px", minHeight: "0px"}}>
                        <Select
                            multiple
                            value={parameter.value}
                            onChange={(event) => {onChange(event.target.value)}}
                            renderValue={(selected) => (
                                <Typography>
                                    {selected.length} selected
                                </Typography>
                            )}
                        >
                            {parameter.options.map((option) => (
                                <MenuItem key={option} value={option}>
                                    {option}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                )
                break;

            default:
                input = (
                    <Typography>
                        {parameter.value}
                    </Typography>
                )
        }

    // make Grid which takes 100% of the height
    // The last row should grow to fill the remaining space
    let defaultText = parameter.type === "boolean" ? (parameter.default ? "true" : "false") : parameter.default
    return (

            <Stack
            direction="row"
            justifyContent="space-between"
            alignItems="center"
            spacing={2}
            sx={{minHeight: "30px"}}
            >
            <InfoTooltip title={
                <Stack spacing={0.5}>
                    <Typography sx={{ fontWeight: 'bold' }}>{parameter.name} (default: {defaultText})</Typography>
                    <Typography sx={{ fontFamily: 'monospace' }}>{`[${parameter_group_id}.${parameter.id}]`}</Typography>
                    <Typography>{parameter.description}</Typography>
                </Stack>
            }>
                <Typography sx={{
                    fontWeight: 400,
                    fontSize: "12px",
                    color: isMatch ? 'red' : (parameter.value !== parameter.default ? theme.palette.primary.main : 'inherit')
                }}>
                    {parameter.name}
                </Typography>
            </InfoTooltip>
                {input}
            </Stack>


    )
}

export default ParameterInput
