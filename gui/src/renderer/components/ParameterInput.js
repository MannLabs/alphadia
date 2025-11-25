import * as React from 'react'
import styled from '@emotion/styled'
import useTheme from '@mui/material/styles/useTheme';
import InfoTooltip from './InfoTooltip'
import { useMethod } from '../logic/context';

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

        // Check if parameter matches search term (space-separated terms are treated as AND)
        const isMatch = React.useMemo(() => {
            if (!searchTerm || searchTerm.trim() === '') return false;

            // Split search term by spaces and filter out empty strings
            const searchTerms = searchTerm.toLowerCase().split(' ').filter(term => term.length > 0);

            // Combine parameter name and description for searching
            const searchableText = (parameter.name + ' ' + (parameter.description || '')).toLowerCase();

            // All terms must be found in either name or description
            return searchTerms.every(term => searchableText.includes(term));
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
            case "textarea":
                input = (
                    <TextField
                    id="outlined-textarea"
                    type="text"
                    variant="standard"
                    size="small"
                    multiline
                    minRows={1}
                    maxRows={16}
                    sx = {{width: "150px"}}
                    value={parameter.value}
                    onChange={(event) => {
                        const valueWithoutLineBreaks = event.target.value.replace(/[\r\n]+/g, '');
                        onChange(valueWithoutLineBreaks);
                    }}
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
                                <Typography sx={{fontSize: "14px", whiteSpace: "normal", wordWrap: "break-word"}}>
                                    {selected.join(', ')}
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

    // Show description only when searching AND there's a match
    const showDescription = searchTerm && searchTerm.trim() !== '' && isMatch;

    // Function to highlight matching terms in text
    const highlightText = (text, searchTerms) => {
        if (!searchTerms || searchTerms.length === 0) return text;

        const fragments = [];
        let lastIndex = 0;

        // Create a regex pattern that matches any of the search terms
        const pattern = searchTerms.map(term => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
        const regex = new RegExp(`(${pattern})`, 'gi');

        const matches = [...text.matchAll(regex)];

        matches.forEach((match) => {
            // Add text before match
            if (match.index > lastIndex) {
                fragments.push(
                    <span key={`text-${lastIndex}`}>
                        {text.substring(lastIndex, match.index)}
                    </span>
                );
            }
            // Add highlighted match
            fragments.push(
                <span key={`match-${match.index}`} style={{ color: theme.palette.primary.selected, fontWeight: 'bold' }}>
                    {match[0]}
                </span>
            );
            lastIndex = match.index + match[0].length;
        });

        // Add remaining text
        if (lastIndex < text.length) {
            fragments.push(
                <span key={`text-${lastIndex}`}>
                    {text.substring(lastIndex)}
                </span>
            );
        }

        return fragments.length > 0 ? fragments : text;
    };

    const searchTerms = searchTerm.toLowerCase().split(' ').filter(term => term.length > 0);

    return (
            <Box>
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
                        color: isMatch ? theme.palette.primary.selected : (parameter.value !== parameter.default ? theme.palette.primary.main : 'inherit')
                    }}>
                        {parameter.name}
                    </Typography>
                </InfoTooltip>
                    {input}
                </Stack>
                {showDescription && parameter.description && (
                    <Typography sx={{
                        fontSize: "11px",
                        color: "text.secondary",
                        mt: 0.5,
                        ml: 0,
                        fontStyle: "italic"
                    }} component="div">
                        {highlightText(parameter.description, searchTerms)}
                    </Typography>
                )}
            </Box>


    )
}

export default ParameterInput
