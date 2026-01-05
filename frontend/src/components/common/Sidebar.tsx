import React from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Box,
  Typography,
  Chip,
} from '@mui/material';
import {
  Home as HomeIcon,
  CloudUpload as UploadIcon,
  BarChart as ExplorationIcon,
  Transform as PreprocessIcon,
  TrendingUp as FeatureIcon,
  ModelTraining as ModelIcon,
  Assessment as EvaluationIcon,
  Tune as TuneIcon,
  Code as CodeIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const DRAWER_WIDTH = 260;

interface NavItem {
  title: string;
  path: string;
  icon: React.ReactElement;
  step?: number;
  description?: string;
}

const navItems: NavItem[] = [
  {
    title: 'Home',
    path: '/',
    icon: <HomeIcon />,
  },
  {
    title: 'Dataset Upload',
    path: '/dataset-upload',
    icon: <UploadIcon />,
    step: 1,
    description: 'Upload & explore data',
  },
  {
    title: 'Exploration',
    path: '/exploration',
    icon: <ExplorationIcon />,
    step: 2,
    description: 'Visualize & analyze',
  },
  {
    title: 'Preprocessing',
    path: '/preprocessing',
    icon: <PreprocessIcon />,
    step: 3,
    description: 'Clean & transform',
  },
  {
    title: 'Feature Engineering',
    path: '/features',
    icon: <FeatureIcon />,
    step: 4,
    description: 'Select & engineer features',
  },
  {
    title: 'Modeling',
    path: '/modeling',
    icon: <ModelIcon />,
    step: 5,
    description: 'Train models',
  },
  {
    title: 'Evaluation',
    path: '/evaluation',
    icon: <EvaluationIcon />,
    step: 6,
    description: 'Assess performance',
  },
  {
    title: 'Hyperparameter Tuning',
    path: '/tuning',
    icon: <TuneIcon />,
    step: 7,
    description: 'Optimize parameters',
  },
  {
    title: 'Code Generation',
    path: '/code-generation',
    icon: <CodeIcon />,
    step: 8,
    description: 'Export & deploy',
  },
];

interface SidebarProps {
  open: boolean;
  onClose: () => void;
  variant?: 'permanent' | 'temporary' | 'persistent';
}

const Sidebar: React.FC<SidebarProps> = ({ open, onClose, variant = 'permanent' }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleNavigation = (path: string) => {
    navigate(path);
    if (variant === 'temporary') {
      onClose();
    }
  };

  const drawerContent = (
    <Box sx={{ overflow: 'auto', mt: 8 }}>
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
          ML WORKFLOW
        </Typography>
      </Box>

      <List>
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;

          return (
            <ListItem key={item.path} disablePadding>
              <ListItemButton
                onClick={() => handleNavigation(item.path)}
                selected={isActive}
                sx={{
                  mx: 1,
                  my: 0.5,
                  borderRadius: 2,
                  transition: 'all 0.2s',
                  '&.Mui-selected': {
                    backgroundColor: 'primary.main',
                    color: 'primary.contrastText',
                    '&:hover': {
                      backgroundColor: 'primary.dark',
                    },
                    '& .MuiListItemIcon-root': {
                      color: 'primary.contrastText',
                    },
                  },
                  '&:hover': {
                    backgroundColor: 'rgba(96, 165, 250, 0.1)',
                  },
                }}
              >
                <ListItemIcon
                  sx={{
                    minWidth: 40,
                    color: isActive ? 'inherit' : 'text.secondary',
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body2" sx={{ fontWeight: isActive ? 600 : 400 }}>
                        {item.title}
                      </Typography>
                      {item.step && (
                        <Chip
                          label={item.step}
                          size="small"
                          sx={{
                            height: 20,
                            fontSize: '0.7rem',
                            backgroundColor: isActive ? 'rgba(255,255,255,0.3)' : 'action.hover',
                          }}
                        />
                      )}
                    </Box>
                  }
                  secondary={
                    item.description && (
                      <Typography
                        variant="caption"
                        sx={{
                          color: isActive ? 'rgba(255,255,255,0.8)' : 'text.secondary',
                          display: 'block',
                          mt: 0.5,
                        }}
                      >
                        {item.description}
                      </Typography>
                    )
                  }
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>

      <Divider sx={{ my: 2 }} />

      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary">
          Follow the steps above to build your ML pipeline
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Drawer
      variant={variant}
      open={open}
      onClose={onClose}
      sx={{
        width: variant === 'temporary' ? undefined : (open ? DRAWER_WIDTH : 0),
        flexShrink: 0,
        transition: (theme) => theme.transitions.create('width', {
          easing: theme.transitions.easing.sharp,
          duration: theme.transitions.duration.enteringScreen,
        }),
        '& .MuiDrawer-paper': {
          width: DRAWER_WIDTH,
          boxSizing: 'border-box',
          backgroundColor: 'background.paper',
          borderRight: '1px solid #334155',
        },
      }}
    >
      {drawerContent}
    </Drawer>
  );
};

export default Sidebar;
export { DRAWER_WIDTH };

