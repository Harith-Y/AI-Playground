import { createTheme } from '@mui/material/styles';

export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2563EB', // Royal Blue
      light: '#60A5FA',
      dark: '#1D4ED8',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#7C3AED', // Violet
      light: '#A78BFA',
      dark: '#5B21B6',
      contrastText: '#ffffff',
    },
    success: {
      main: '#10B981', // Emerald
      light: '#34D399',
      dark: '#059669',
    },
    error: {
      main: '#EF4444', // Red
      light: '#F87171',
      dark: '#DC2626',
    },
    warning: {
      main: '#F59E0B', // Amber
      light: '#FBBF24',
      dark: '#D97706',
    },
    info: {
      main: '#3B82F6', // Blue
      light: '#60A5FA',
      dark: '#2563EB',
    },
    background: {
      default: '#F8FAFC', // Slate 50
      paper: '#FFFFFF',
    },
    text: {
      primary: '#1E293B', // Slate 800
      secondary: '#64748B', // Slate 500
    },
    divider: '#E2E8F0', // Slate 200
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      letterSpacing: '-0.02em',
      color: '#1E293B',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 700,
      letterSpacing: '-0.01em',
      color: '#1E293B',
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
      letterSpacing: '-0.01em',
      color: '#1E293B',
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
      color: '#1E293B',
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
      color: '#1E293B',
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
      color: '#1E293B',
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
          padding: '8px 20px',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          },
        },
        containedPrimary: {
          background: 'linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: '#FFFFFF',
          border: '1px solid #E2E8F0',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
  },
});

export default theme;

