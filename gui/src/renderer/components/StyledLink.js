
import { Link } from "@mui/material";
import styled from '@emotion/styled';

const StyledLink = styled(Link)(({ theme }) => ({
    color: theme.palette.text.primary,
    fontFamily: theme.typography.fontFamilyMono,
    textDecoration: "none",
    borderBottom: "1px dashed",

    '&:hover': {
        textDecoration: "none",
        borderBottom: "1px solid",
        cursor: "pointer",
    },
    '&:active': {
        textDecoration: "none",
        borderBottom: "1px solid",
    },
    '&:visited': {
        textDecoration: "none",
        borderBottom: "1px dashed",
    },
}))

export default StyledLink;