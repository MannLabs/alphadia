import * as React from 'react'
import { useTheme } from '@mui/material/styles';
import { Box } from '@mui/material';

const Card = ({sx = [], children}) => {
    const theme = useTheme();
    return (
    <Box display="flex" flexDirection="column" alignItems="stretch" padding={1}
    sx={[
        {
        padding: 2,
        border: 1,
        borderRadius: 4,
        borderColor: theme.palette.divider,
    },
        ...(Array.isArray(sx) ? sx : [sx]),
        ]}
    >
        {children}
    </Box>
)}

export default Card