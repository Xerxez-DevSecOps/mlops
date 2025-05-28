import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { useTheme, useMediaQuery } from '@mui/material';
import { green } from '@mui/material/colors';

const EditorHeader = () => {
    const theme = useTheme();
    const isMd = useMediaQuery(
        theme.breakpoints.up('md'),
        { defaultMatches: true }
    );
    
    return (
        < >
            <Typography
                variant='h2'
                color={theme.palette.text.primary}
            >
                Form
            </Typography>
            <Box paddingTop={2}>
                <Typography
                    variant='h5'
                    color={theme.palette.text.secondary}
                >
                    Fill the 
                    {' '}
                    <span style={{ color: green[600] }}>
                    Detail 
                    </span>
                    {' '}
                     to get the Cost
                </Typography>
            </Box>
        </>
    );
};

export default EditorHeader;