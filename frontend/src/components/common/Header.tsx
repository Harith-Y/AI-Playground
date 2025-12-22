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
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid',
        borderColor: 'divider',
        boxShadow: 'none',
        color: 'text.primary',
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
                backgroundColor: (theme) => theme.palette.primary.light + '20',
                color: 'primary.main',
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
                backgroundColor: (theme) => theme.palette.primary.light + '20',
                color: 'primary.main',
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
                backgroundColor: (theme) => theme.palette.primary.light + '20',
                color: 'primary.main',
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

