import { Tooltip, tooltipClasses } from '@mui/material'
import styled from '@emotion/styled'

const InfoTooltip = styled(({ className, ...props }) => (
    <Tooltip {...props} classes={{ popper: className }} placement="right-start" />
))(({ theme }) => ({
    [`& .${tooltipClasses.tooltip}`]: {
        backgroundColor: theme.palette.background.default,
        color: theme.palette.text.primary,
        maxWidth: 400,
        fontSize: theme.typography.pxToRem(12),
        padding: '8px 12px',
        boxShadow: `0px 0px 10px 0px rgba(0, 0, 0, 0.1)`,
        border: `1px solid ${theme.palette.divider}`,
        '& .MuiTypography-root': {
            fontSize: 'inherit'
        }
    },
}));

export default InfoTooltip;
