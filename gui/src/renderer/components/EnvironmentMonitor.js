import React, { useState } from 'react';
import useTheme from '@mui/material/styles/useTheme';

import { Stack, Box, Typography, Popover } from '@mui/material';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import CheckIcon from '@mui/icons-material/Check';

const EnvironmentMonitor = ({ environment = {}, sx = [] }) => {

  const [anchorEl, setAnchorEl] = useState(null);
  const handlePopoverOpen = (event) => setAnchorEl(event.currentTarget);
  const handlePopoverClose = () => setAnchorEl(null);
  const open = Boolean(anchorEl);
  const theme = useTheme();

  return (
    <Box
      sx={[
        { display: 'flex' },
        ...(Array.isArray(sx) ? sx : [sx]),
      ]}
    >
      <Stack
        direction="row"
        spacing={1}
        alignItems="center"
        justifyContent="space-between"
        onMouseEnter={handlePopoverOpen}
        onMouseLeave={handlePopoverClose}
      >
        {environment.ready ? (
          <>
            <CheckIcon sx={{ color: theme.palette.success.main }} />
            <Typography sx={{ color: theme.palette.success.main }}>ready</Typography>
          </>
        ) : (
          <>
            <ErrorOutlineIcon sx={{ color: theme.palette.error.main }} />
            <Typography sx={{ color: theme.palette.error.main }}>error</Typography>
          </>
        )}
      </Stack>
      <Popover
        id="mouse-over-popover"
        sx={{
          pointerEvents: 'none',
        }}
        open={open}
        anchorEl={anchorEl}
        anchorOrigin={{
          vertical: 'center',
          horizontal: 'left',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'left',
        }}
        onClose={handlePopoverClose}
        disableRestoreFocus
      >
        <Box sx={{ p: 2, width: '200px' }}>
          <Typography sx={{ fontWeight: 500, mb: 1 }}>Environment Status</Typography>
          {[
            { label: 'Name:', value: environment.envName },
            { label: 'Conda:', value: environment?.versions?.conda },
            { label: 'Python:', value: environment?.versions?.python },
            { label: 'AlphaDIA:', value: environment?.versions?.alphadia },
          ].map((item, index) => (
            <Stack key={index} direction="row" spacing={1} alignItems="center" justifyContent="space-between">
              <Typography>{item.label}</Typography>
              <Typography sx={{ fontFamily: 'Roboto Mono' }}>{item.value}</Typography>
            </Stack>
          ))}
        </Box>
      </Popover>
    </Box>
  );
};

export default EnvironmentMonitor;
