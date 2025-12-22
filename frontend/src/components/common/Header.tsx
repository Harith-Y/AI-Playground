import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Button,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import { useNavigate } from 'react-router-dom';

interface HeaderProps {
  onMenuClick: () => void;
}

const Header: React.FC<HeaderProps> = ({ onMenuClick }) => {
  const navigate = useNavigate();

  const handleLogoClick = () => {
    navigate('/');
  };

  return (
    <AppBar
      position="fixed"
      sx={{
        zIndex: (theme) => theme.zIndex.drawer + 1,
        backgroundColor: 'background.paper',
        borderBottom: '1px solid #334155',
        boxShadow: 'none',
      }}
    >      <Toolbar>
        <IconButton
          color="inherit"
          aria-label="open drawer"
          edge="start"
          onClick={onMenuClick}
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>

        <Typography
          variant="h6"
          noWrap
          component="div"
          sx={{
            flexGrow: 0,
            cursor: 'pointer',
            fontWeight: 600,
            mr: 4,
          }}
          onClick={handleLogoClick}
        >
          AI Playground
        </Typography>

        <Box sx={{ flexGrow: 1, display: 'flex', gap: 2 }}>
          <Button
            color="inherit"
            onClick={() => navigate('/datasets')}
            sx={{
              textTransform: 'none',
              '&:hover': {
                backgroundColor: 'rgba(96, 165, 250, 0.1)',
              },
            }}
          >
            Datasets
          </Button>
          <Button
            color="inherit"
            onClick={() => navigate('/models')}
            sx={{
              textTransform: 'none',
              '&:hover': {
                backgroundColor: 'rgba(96, 165, 250, 0.1)',
              },
            }}
          >
            Models
          </Button>
          <Button
            color="inherit"
            onClick={() => navigate('/experiments')}
            sx={{
              textTransform: 'none',
              '&:hover': {
                backgroundColor: 'rgba(96, 165, 250, 0.1)',
              },
            }}
          >
            Experiments
          </Button>
        </Box>

        <IconButton color="inherit" aria-label="account">
          <AccountCircleIcon />
        </IconButton>
      </Toolbar>
    </AppBar>
  );
};

export default Header;

