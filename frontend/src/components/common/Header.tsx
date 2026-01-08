import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Button,
  Tooltip,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import { useNavigate } from 'react-router-dom';
import { useThemeContext } from '../../contexts/ThemeContext';

interface HeaderProps {
  onMenuClick: () => void;
}

const Header: React.FC<HeaderProps> = ({ onMenuClick }) => {
  const navigate = useNavigate();
  const { mode, toggleTheme } = useThemeContext();

  const handleLogoClick = () => {
    navigate('/');
  };

  return (
    <AppBar
      position="fixed"
      sx={{
        zIndex: (theme) => theme.zIndex.drawer + 1,
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid',
        borderColor: 'divider',
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

        <Tooltip title={mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}>
          <IconButton onClick={toggleTheme} color="inherit" aria-label="toggle theme" sx={{ mr: 1 }}>
            {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
          </IconButton>
        </Tooltip>

        <IconButton color="inherit" aria-label="account">
          <AccountCircleIcon />
        </IconButton>
      </Toolbar>
    </AppBar>
  );
};

export default Header;

