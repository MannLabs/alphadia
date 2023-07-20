import * as React from 'react'
import styled from '@emotion/styled';

import { Grid, Box, Typography, Stack, CircularProgress} from '@mui/material'
import { Card, AlphapeptIcon, StyledLink, DiscussionCard } from '../components';

import CheckIcon from '@mui/icons-material/Check';
import MenuBookIcon from '@mui/icons-material/MenuBook'; 
import GitHubIcon from '@mui/icons-material/GitHub';
import { useTheme } from '@emotion/react';


const StyledMenuBookIcon = styled(MenuBookIcon)(({ theme }) => ({
    color: theme.palette.divider,
    fontSize: "5rem",
}))

const StyledGitHubIcon = styled(GitHubIcon)(({ theme }) => ({
    color: theme.palette.divider,
    fontSize: "5rem",
}))

const WelcomeCard = ({children}) => (
    <Box display="flex" flexDirection="column" alignItems="stretch" padding={1}
    sx={{
      background: "linear-gradient(199deg, rgba(224,195,252,0.8) 0%, rgba(142,197,252,0.8) 100%), url(./images/noise.svg);",
      filter: 'contrast(130%) brightness(100%)',
      borderRadius: 4,
      borderColor: "grey.300",
      padding: 2,
      height: '200px'
      
    }}
    >
      {children}
    </Box>
)

const VersionCard = ({environment}) => {

    const theme = useTheme();

    let versionOutput = ""
    let updateOutput = ""

    const versionString = environment?.versions?.alphadia
    if (versionString === undefined) {
        versionOutput = <CircularProgress />
        updateOutput = ""
    } else {
        versionOutput = versionString
        if (versionString === "1.2.0") {
            updateOutput = (
                <Stack direction="row" alignItems="center" gap={1}>
                    <CheckIcon sx={{color:"rgb(75, 211, 26)"}}/>
                    <Typography component="span" sx={{fontWeight: 400, fontSize: "1rem", color:"rgb(75, 211, 26)" }}> Latest Version</Typography>
                </Stack>
            )
        } else {
            updateOutput = (
                <Stack direction="row" alignItems="center" gap={1}>
                    <Typography component="span" sx={{fontWeight: 400, fontSize: "1rem", color:"rgb(255, 0, 0)" }}> Update Available</Typography>
                </Stack>
            )
        }
    }
   
    return (
    <Card sx={{height: '200px', position:'relative'}}>
        <Box component="div" sx={{fontWeight:200, position:'absolute', left:"0", bottom:"0", p:2}}>
            <Typography component="span" sx={{fontWeight: 200, fontSize: "3rem", fontFamily:"Roboto Mono", letterSpacing:"-0.2em" }}>{versionOutput}</Typography>
            <br/>
            {updateOutput}            
        </Box>
    </Card>
)
}
const Home = ({environment}) => (
  
    <Grid container spacing={2}>
        {/*create a grid item with a width of 4*/}
        <Grid item xs={8}>
            <WelcomeCard sx={{height: '200px', position:'relative'}}>
                <Box component="div" sx={{position:'absolute', left:"0", bottom:"0", p:2}}>
                    <Typography component="span" sx={{fontWeight: 200, fontSize: "1.5rem"}}>Welcome to</Typography>
                    <br/>
                    <Typography component="span" sx={{fontWeight: 700, fontSize: "3rem"}}>alphaDIA</Typography>
                    
                </Box>
            </WelcomeCard>
        </Grid>
        {/*create a grid item with a width of 4*/}
        <Grid item xs={4}>
            <VersionCard environment={environment}/>
        </Grid>
        {/*create a grid item with a width of 4*/}
        <Grid item xs={4}>
            <Card sx={{height: '200px', position:'relative'}}>
                <Box component="div" sx={{position:'absolute', left:"0", bottom:"0", p:2}}>
                    <StyledMenuBookIcon/><br />
                    <Typography component="span" sx={{fontWeight: 700, fontSize: "1rem"}}>
                        Documentation
                    </Typography>
                    <br/>
                    <StyledLink onClick={() => window.electronAPI.openLink("http://www.google.com")}>
                        Link
                    </StyledLink>
                </Box>
            </Card>
        </Grid>
        <Grid item xs={4}>
            <Card sx={{height: '200px', position:'relative'}}>
                <Box component="div" sx={{position:'absolute', left:"0", bottom:"0", p:2}}>
                    <StyledGitHubIcon/><br />
                    <Typography component="span" sx={{fontWeight: 700, fontSize: "1rem"}}>
                        GitHub
                    </Typography>
                    <br/>
                    <StyledLink onClick={() => window.electronAPI.openLink("http://www.google.com")}>
                        Link
                    </StyledLink>
                </Box>
            </Card>
        </Grid>
        <Grid item xs={4}>
            <Card sx={{
                height: '200px', 
                position:'relative', 
                background: "linear-gradient(18deg, rgb(35 55 73 / 81%) 0%, rgb(11 49 87 / 80%) 100%), url(./images/noise.svg)", 
                filter: 'contrast(130%) brightness(100%)'}}>
                <Box component="div" sx={{position:'absolute', left:"0", bottom:"0", p:2}}>
                    <AlphapeptIcon sx={{fontSize: "5rem"}}/><br />
                    <Typography component="span" sx={{fontWeight: 700, fontSize: "1rem",color: "white"}}>
                        AlphaPept<br/>
                        Universe
                    </Typography>
                    <br/>
                    <StyledLink onClick={() => window.electronAPI.openLink("http://www.google.com")} sx={{color: "white !important"}}>
                        Link
                    </StyledLink>
                </Box>
            </Card>
        </Grid>
        <Grid item xs={12}>
            <DiscussionCard/>
        </Grid>
    </Grid>


)

export default Home