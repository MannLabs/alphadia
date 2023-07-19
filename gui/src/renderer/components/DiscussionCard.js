import * as React from 'react'

import { Card, StyledLink } from '.';
import { Box, List, ListItem, ListItemText, ListItemSecondaryAction, Typography, CircularProgress } from '@mui/material';

function fetchGithubIssues(repos) {
    // iterate all repos
    return Promise.all(repos.map(repo => {
        // fetch the issues
        return fetch(repo.url).then((res) => {
            if (!res.ok) {
                throw new Error(`HTTP error ${res.status} for ${repo.url}`);
            } else {
                return res.json();
            }
        }).then(issues => {
            return {
                name: repo.name,
                issues: issues
            }
        }).catch(err => {
            console.log(err);
            return []
        })
    })).then(data => {
        // combine all the issues into one array and add the repo name
        // sort them by updated_at date
        // take top 20
        const issues = data.reduce((acc, repo) => {
            return acc.concat(repo.issues.map(issue => {
                return {
                    name: repo.name,
                    title: issue.title,
                    url: issue.html_url,
                    updated_at: new Date(issue.updated_at),
                    days_ago: Math.floor((new Date() - new Date(issue.updated_at)) / (1000 * 60 * 60 * 24)),
                    hours_ago: Math.floor((new Date() - new Date(issue.updated_at)) / (1000 * 60 * 60)) % 24,

                }
            }))
        }, []).sort((a, b) => {
            return b.updated_at - a.updated_at
        }).slice(0, 20)
        return issues;
    }).catch(err => {
        console.log(err);
        return []
    })
}




const DiscussionCard = ({sx = []}) => {

    //display the 20 most recent issues from the github repos
    // display loading while fetching

    const [issues, setIssues] = React.useState([]);
    
    
    React.useEffect(() => {

            fetchGithubIssues([
                {url: "https://api.github.com/repos/MannLabs/alphabase/issues", name: "AlphaBase"},
                {url: "https://api.github.com/repos/MannLabs/alphapept/issues", name: "AlphaPept"},
                {url: "https://api.github.com/repos/MannLabs/alphatims/issues", name: "AlphaTims"},
                {url: "https://api.github.com/repos/MannLabs/alphapeptdeep/issues", name: "AlphaPeptDeep"},
            ]).then((data) => {
                console.log("data", data);
                setIssues(data)
            }
                
            );

    }, []);

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
            <Typography component="h1" variant="h1" sx={{fontWeight: 700, fontSize: "1rem", mt:0}}>
                Recent Discussions
            </Typography>

            {
                issues.length === 0 &&
                <Box sx={{display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100px'}}>
                    <CircularProgress/>
                </Box>
            }

            <List sx={{ width: '100%'}}>
                {issues.map((issue) => (
                    <ListItem key={issue.url} disablePadding>
                        <ListItemText 
                            primary={issue.title} 
                            secondary={ (issue.days_ago > 0 ? `${issue.days_ago} d ` : ``) + `${issue.hours_ago} h ago - ` + issue.name}
                        />
                        <ListItemSecondaryAction>
                            <StyledLink onClick={() => window.electronAPI.openLink(issue.url)}>
                                View
                            </StyledLink>
                        </ListItemSecondaryAction>
                    </ListItem>
                ))}
            </List>


        </Card>
    )
}
  

export default DiscussionCard

