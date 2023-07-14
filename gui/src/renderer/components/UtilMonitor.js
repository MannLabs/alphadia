import * as React from 'react'

import Box from '@mui/material/Box'
import { Stack, Typography } from '@mui/material'

const UtilMonitor = ({sx = []}) => {

    const [utilisation, setUtilisation] = React.useState({
        cpu: 0,
        freeMemMb: 0,
        usedMemMb: 0,
        totalMemMb: 0,
        freeMemPercentage: 0,
        usedMemPercentage: 0
    });

    
    React.useEffect(() => {
        setInterval(() => {
            window.electronAPI.getUtilisation().then((utilisation) => {
                setUtilisation(utilisation);
            });
        }, 1000);
    }, []);

    return (
    <Box sx={[
        {
            display: "flex",
            fontFamily: "Roboto Mono",
        },
        ...(Array.isArray(sx) ? sx : [sx]),
        ]}
        >
        <Stack direction="row" spacing={2} alignItems="center" justifyContent="space-between">
            <Typography 
                component="div" 
                sx={{
                    fontWeight: 400, 
                    fontSize: "0.8rem", 
                    fontFamily: "Roboto Mono", 
                    minWidth:"70px"}}>
                    CPU: {utilisation.cpu.toFixed(2)}%
                </Typography>
            <Typography component="div" sx={{fontWeight: 400, fontSize: "0.8rem", fontFamily: "Roboto Mono", minWidth:"70px"}}>
                RAM: {(utilisation.usedMemMb/1024).toFixed(2)} GB / {(utilisation.totalMemMb/1024).toFixed(2)} GB ({utilisation.usedMemPercentage}%)
            </Typography>
        </Stack>
    
        </Box>
    )
}
  

export default UtilMonitor