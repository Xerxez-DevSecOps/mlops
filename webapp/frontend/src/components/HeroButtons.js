import React from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import { useTheme, useMediaQuery } from '@mui/material';
 
import InfoIcon from '@mui/icons-material/HelpOutline';
import PlayIcon from '@mui/icons-material/PlayCircleOutlineOutlined';
import { Link } from 'react-router-dom';

const HeroButtons = () => {
    const theme = useTheme();
    const isMd = useMediaQuery(
        theme.breakpoints.up('md'),
        { defaultMatches: true }
    );
    
    return (
        < >
            <Box
                display='flex'
                flexDirection={{ xs: 'column', sm: 'row' }}
                alignItems={{ xs: 'stretched', sm: 'flex-start' }}
                justifyContent='center'
                marginTop={4}
            >
                    <Link to='/insurance-form' style={{ textDecoration: 'none' }}>

                <Button
                    component='a'
                    variant='contained'
                    color='primary'
                    size='large'
                    startIcon={<InfoIcon />}
                    fullWidth={isMd ? false : true}
                    disableElevation={true}
                    sx={{
                        padding: '18px 34px',
                        marginRight: '15px',
                        fontSize: '18px',
                        border: '2px solid ' + theme.palette.primary.main,
                        '&:hover': {
                            backgroundColor: 'transparent',
                            color: theme.palette.primary.main,
                            border: '2px solid ' + theme.palette.primary.main
                        }
                    }}
                >
                    Get Started
                </Button></Link>
               
            </Box>
        </>
    );
};

export default HeroButtons;