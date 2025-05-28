import React from 'react';
import Box from '@mui/material/Box';
import Divider from '@mui/material/Divider';
import Grid from '@mui/material/Grid';
import Link from '@mui/material/Link';
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import Typography from '@mui/material/Typography';
import { useTheme } from '@mui/material/styles';
import useMediaQuery from '@mui/material/useMediaQuery';

const Footer = () => {
    const theme = useTheme();
    const isMdUp = useMediaQuery(theme.breakpoints.up('md')); // replaces <Hidden mdDown>

    return (
        <React.Fragment>
            <Box
                backgroundColor={theme.palette.background.default}
                paddingTop='1px'
                paddingBottom='1px'
                position='fixed'
                bottom='0'
                left='0'
                width='100%'
            >
                <Divider />
                <Box
                    backgroundColor={theme.palette.background.paper}
                    position='relative'
                    padding={theme.spacing(0.25)}
                >
                    <Grid container spacing={0}>
                        {isMdUp && (
                            <Grid container item xs={12} md={4}>
                                {/* You can add content here for larger screens */}
                            </Grid>
                        )}
                        <Grid container item xs={12} md={4} justifyContent='center'>
                            <List>
                                <ListItemButton>
                                    <ListItemText 
                                        primary={
                                            <Typography 
                                                variant='body2'
                                                color={theme.palette.text.secondary}
                                            >
                                                Copyright &copy; {new Date().getFullYear()} Xerxez Solutions.
                                            </Typography>
                                        }
                                    />
                                </ListItemButton>
                            </List>
                        </Grid>
                    </Grid>
                </Box>
            </Box>
        </React.Fragment>
    );
};

export default Footer;
