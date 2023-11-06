import { grey } from '@mui/material/colors';

const buildPalette = (mode) => {
    const isDark = mode === 'dark';
    return {
        mode,
        primary: {
            main: isDark? "rgb(25,218,228)" : "rgb(0, 155, 163)",
            selected: isDark? "rgb(17,71,74)" : "#bbf3f6",
            hover: isDark? "#282828" : "#e8fbfc",
            dark: "#8eebf0",
            contrastText: "#009ba3"            
        },
        success: {
            main: "rgb(75, 211, 26)",
        },
        text: {
            primary: isDark? "rgba(255,255,255,1)" : grey[900],
            secondary: isDark? "rgba(255,255,255,0.7)" : grey[800],
            disabled: isDark? "rgba(255,255,255,0.5)" : grey[500],
        },
        background: {
            default: isDark? "#222" : "#fff",
            paper: isDark? "#111" : "#fff",
        },
        divider: isDark? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.12)",
    }
}

const getDesignTokens = (mode) => ({
    palette: buildPalette(mode),
    typography: {
        fontSize: 12,
        fontFamilyMono: "Roboto Mono",
    },
    breakpoints: {
        values: {
            xs: 0,
            sm: 1000,
            md: 1300,
            lg: 1600,
            xl: 1900,
        },
        },
    components: {
        Link: {
            styleOverrides: (themeParam) => {
                return {
                    root: {
                        color: themeParam.palette.text.primary,
                        textDecoration: "none",
                        borderBottom: "1px dashed",
                    }
                }
            }
        },
        MuiCssBaseline: {
            styleOverrides: (themeParam) => `
            :root {
                color-scheme: ${themeParam.palette.mode};
            }

            h1 {
                margin-top: ${themeParam.spacing(1)};
                font-size: 1.5rem;
                font-weight: 700;
            }

            .MuiLink-root {
                color: ${themeParam.palette.text.primary};
                text-decoration: none; 
                border-bottom:1px dashed;
            }

            .MuiMenuItem-root{
                min-height: 0px !important;
            }

            .MuiListItemIcon-root {
                color: inherit !important;
            }
            
            `,
        },
    },
    
});

export default getDesignTokens;