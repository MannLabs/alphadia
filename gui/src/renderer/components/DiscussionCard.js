import * as React from 'react'

import { Card, StyledLink } from '.';
import { Box, List, ListItem, ListItemText, ListItemSecondaryAction, Typography, CircularProgress, IconButton, Tooltip, Divider } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { useGitHub } from '../logic/context';

const DiscussionCard = ({sx = []}) => {
    //display the 20 most recent issues and releases from the github repos using context
    const {
        githubData,
        loading,
        lastUpdated,
        formatLastUpdated,
        refreshData
    } = useGitHub();

    // Get the combinedItems from githubData if available
    const items = githubData?.combinedItems || [];

    return (
    <Card sx={[
        {
            minHeight: "100px",
            marginBottom: "16px",
            border: "none"
        },
        ...(Array.isArray(sx) ? sx : [sx]),
        ]}
        >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography component="h1" variant="h1" sx={{fontWeight: 700, fontSize: "1rem", mt:0}}>
                    Recent Updates in AlphaX
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    {lastUpdated && (
                        <Typography variant="caption" sx={{ mr: 1, color: 'text.secondary' }}>
                            Updated {formatLastUpdated(lastUpdated)}
                        </Typography>
                    )}
                    <Tooltip title="Refresh">
                        <IconButton
                            size="small"
                            onClick={refreshData}
                            disabled={loading}
                            sx={{ padding: '2px' }}
                        >
                            <RefreshIcon fontSize="small" />
                        </IconButton>
                    </Tooltip>
                </Box>
            </Box>

            {loading ? (
                <Box sx={{display: 'flex', justifyContent: 'center', alignItems: 'center', height: '200px'}}>
                    <CircularProgress/>
                </Box>
            ) : (
                <List sx={{ width: '100%', mt: 1}}>
                    {items.map((item) => (
                        <ListItem
                            key={item.url}
                            disablePadding
                            sx={item.type === 'release' ? {
                                border: '1px solid',
                                borderColor: 'divider',
                                borderRadius: '4px',
                                marginBottom: '6px',
                                marginTop: '6px',
                                padding: '4px 8px'
                            } : {padding: '4px 8px'}}
                        >
                            <ListItemText
                                primary={
                                    <Typography
                                        sx={item.type === 'release' ? {
                                            fontWeight: 'bold'
                                        } : {}}
                                    >
                                        {item.title}
                                    </Typography>
                                }
                                secondary={
                                    <>
                                        <Typography variant="caption" component="div">
                                            {(item.days_ago > 0 ? `${item.days_ago} d ` : ``) +
                                            `${item.hours_ago} h ago - ${item.name} ${item.type === 'release' ? '📦' : ''}`}
                                        </Typography>


                                    </>
                                }
                            />
                            <ListItemSecondaryAction>
                                <StyledLink onClick={() => window.electronAPI.openLink(item.url)}>
                                    View
                                </StyledLink>
                            </ListItemSecondaryAction>
                        </ListItem>
                    ))}
                </List>
            )}
        </Card>
    )
}


export default DiscussionCard
