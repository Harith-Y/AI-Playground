import React, { useState } from 'react';
import { Box, useMediaQuery, useTheme } from '@mui/material';
import Header from './Header';
import Sidebar from './Sidebar';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = useState(false);
  const [desktopOpen, setDesktopOpen] = useState(true);

  const handleDrawerToggle = () => {
    if (isMobile) {
      setMobileOpen(!mobileOpen);
    } else {
      setDesktopOpen(!desktopOpen);
    }
  };

  return (
    <Box sx={{
      display: 'flex',
      minHeight: '100vh',
      background: theme.palette.mode === 'light'
        ? 'linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%)'
        : 'linear-gradient(135deg, #0F172A 0%, #1E293B 100%)'
    }}>
      {/* Header */}
      <Header onMenuClick={handleDrawerToggle} />

      {/* Sidebar */}
      {isMobile ? (
        <Sidebar
          open={mobileOpen}
          onClose={() => setMobileOpen(false)}
          variant="temporary"
        />
      ) : (
        <Sidebar
          open={desktopOpen}
          onClose={() => {}}
          variant="persistent"
        />
      )}

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 0,
          mt: 8,
          width: '100%',
          // backgroundColor: 'background.default',
          minHeight: '100vh',
        }}
      >
        {children}
      </Box>
    </Box>
  );
};

export default Layout;

