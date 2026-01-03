import { createTheme, Theme } from '@mui/material/styles';

type ThemeMode = 'light' | 'dark';

export const createAppTheme = (mode: ThemeMode): Theme => {
  const isLight = mode === 'light';

  return createTheme({
    palette: {
      mode,
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
        default: isLight ? '#F8FAFC' : '#0F172A', // Slate 50 / Slate 900
        paper: isLight ? '#FFFFFF' : '#1E293B', // White / Slate 800
      },
      text: {
        primary: isLight ? '#1E293B' : '#F1F5F9', // Slate 800 / Slate 100
        secondary: isLight ? '#64748B' : '#94A3B8', // Slate 500 / Slate 400
      },
      divider: isLight ? '#E2E8F0' : '#334155', // Slate 200 / Slate 700
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      h1: {
        fontSize: '2.5rem',
        fontWeight: 700,
        letterSpacing: '-0.02em',
      },
      h2: {
        fontSize: '2rem',
        fontWeight: 700,
        letterSpacing: '-0.01em',
      },
      h3: {
        fontSize: '1.75rem',
        fontWeight: 600,
        letterSpacing: '-0.01em',
      },
      h4: {
        fontSize: '1.5rem',
        fontWeight: 600,
      },
      h5: {
        fontSize: '1.25rem',
        fontWeight: 600,
      },
      h6: {
        fontSize: '1rem',
        fontWeight: 600,
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
              boxShadow: isLight
                ? '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
                : '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)',
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
            backgroundColor: isLight ? '#FFFFFF' : '#1E293B',
            border: `1px solid ${isLight ? '#E2E8F0' : '#334155'}`,
            boxShadow: isLight
              ? '0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)'
              : '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)',
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
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: isLight ? 'rgba(255, 255, 255, 0.8)' : 'rgba(15, 23, 42, 0.8)',
            color: isLight ? '#1E293B' : '#F1F5F9',
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundColor: isLight ? '#FFFFFF' : '#1E293B',
            borderRight: `1px solid ${isLight ? '#E2E8F0' : '#334155'}`,
          },
        },
      },
    },
  });
};

// Export a default light theme for backward compatibility
export const theme = createAppTheme('light');

export default theme;

