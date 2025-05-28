import React from 'react';
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography';
import { useTheme } from '@mui/material';

// Font Awesome Icons
import { library } from '@fortawesome/fontawesome-svg-core';
import { faEdit as EditIcon } from '@fortawesome/free-regular-svg-icons';
import { faDownload as DownloadIcon } from '@fortawesome/free-solid-svg-icons';
import { faShareSquare as ShareSquareIcon } from '@fortawesome/free-regular-svg-icons';
import { faLaptopCode as LaptopCodeIcon } from '@fortawesome/free-solid-svg-icons';

// Custom component
import DescriptionItem from './DescriptionItem';

// Add icons to library
library.add(EditIcon, DownloadIcon, ShareSquareIcon, LaptopCodeIcon);

const Description = () => {
    const theme = useTheme();

    return (
        <Box
            maxWidth={{ sm: 720, md: 1236 }}
            width={1}
            margin="0 auto"
            paddingTop={2}
            paddingBottom={2}
        >
            <Box
                data-aos="fade-up"
                paddingTop={4}
                backgroundColor={theme.palette.background.default}
            >
                <Container
                    maxWidth="md"
                    display="flex"
                    sx={{
                        alignItems: 'center',
                        flexDirection: 'column',
                        px: { md: '15px !important' },
                    }}
                >
                    <Typography
                        variant="h1"
                        color={theme.palette.text.primary}
                        align="center"
                        marginTop="30px"
                        data-aos="fade-up"
                    >
                        How Does It Work?
                    </Typography>
                    <Typography
                        variant="h4"
                        color={theme.palette.text.secondary}
                        align="center"
                        paddingTop={3}
                        paddingBottom={3}
                        marginBottom="15px"
                        data-aos="fade-up"
                    >
                        A step-by-step guide on how to use the app
                    </Typography>
                    <Grid container spacing={4} data-aos="fade-up">
                        <DescriptionItem
                            color="rgb(31, 102, 239)"
                            icon={EditIcon}
                            title="Fill the Form"
                            subtitle="First, fill the form to best of your knowledge so it can give you estimated insurance cost precisely."
                        />
                        <DescriptionItem
                            color={theme.palette.error.dark}
                            icon={ShareSquareIcon}
                            title="Send the Data for Prediction"
                            subtitle="Click on Result button to know the prediction of model."
                        />
                        <DescriptionItem
                            color={theme.palette.primary.main}
                            icon={LaptopCodeIcon}
                            title="Get the Prediction Result"
                            subtitle="Once you have sent your data to the machine learning model, the model returns the cost of insurance for your whole family."
                        />
                        <DescriptionItem
                            color={theme.palette.warning.dark}
                            icon={DownloadIcon}
                            title="Download Your Prediction Report"
                            subtitle="You can download your report by pressing the Download button."
                        />
                    </Grid>
                </Container>
            </Box>
        </Box>
    );
};

export default Description;
