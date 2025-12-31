/**
 * Download Options Component
 * 
 * Provides various download options for generated code
 */

import React, { useState } from 'react';
import {
  Box,
  Button,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControlLabel,
  Checkbox,
  Typography,
  Alert,
} from '@mui/material';
import {
  Download as DownloadIcon,
  InsertDriveFile as FileIcon,
  FolderZip as ZipIcon,
  GitHub as GitHubIcon,
  CloudDownload as CloudDownloadIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';

interface DownloadOptionsProps {
  onDownloadFile: (filename: string) => void;
  onDownloadZip: (options: ZipOptions) => void;
  onExportToGitHub?: () => void;
  disabled?: boolean;
}

export interface ZipOptions {
  includeRequirements: boolean;
  includeReadme: boolean;
  includeDockerfile: boolean;
  includeTests: boolean;
  projectName: string;
}

const DownloadOptions: React.FC<DownloadOptionsProps> = ({
  onDownloadFile,
  onDownloadZip,
  onExportToGitHub,
  disabled = false,
}) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [zipDialogOpen, setZipDialogOpen] = useState(false);
  const [zipOptions, setZipOptions] = useState<ZipOptions>({
    includeRequirements: true,
    includeReadme: true,
    includeDockerfile: false,
    includeTests: false,
    projectName: 'ml-project',
  });

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleDownloadFile = () => {
    onDownloadFile('model.py');
    handleClose();
  };

  const handleDownloadZipClick = () => {
    setZipDialogOpen(true);
    handleClose();
  };

  const handleZipDownload = () => {
    onDownloadZip(zipOptions);
    setZipDialogOpen(false);
  };

  const handleExportGitHub = () => {
    onExportToGitHub?.();
    handleClose();
  };

  return (
    <>
      <Button
        variant="contained"
        startIcon={<DownloadIcon />}
        onClick={handleClick}
        disabled={disabled}
      >
        Download
      </Button>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleClose}
      >
        <MenuItem onClick={handleDownloadFile}>
          <ListItemIcon>
            <FileIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="Download as File"
            secondary="Single Python/Notebook file"
          />
        </MenuItem>

        <MenuItem onClick={handleDownloadZipClick}>
          <ListItemIcon>
            <ZipIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="Download as ZIP"
            secondary="Complete project structure"
          />
        </MenuItem>

        <Divider />

        {onExportToGitHub && (
          <MenuItem onClick={handleExportGitHub}>
            <ListItemIcon>
              <GitHubIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText
              primary="Export to GitHub"
              secondary="Create new repository"
            />
          </MenuItem>
        )}

        <MenuItem disabled>
          <ListItemIcon>
            <CloudDownloadIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="Deploy to Cloud"
            secondary="Coming soon"
          />
        </MenuItem>
      </Menu>

      {/* ZIP Options Dialog */}
      <Dialog open={zipDialogOpen} onClose={() => setZipDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SettingsIcon />
          ZIP Package Options
        </DialogTitle>
        <DialogContent>
          <Alert severity="info" sx={{ mb: 3 }}>
            Configure what to include in your project package
          </Alert>

          <TextField
            fullWidth
            label="Project Name"
            value={zipOptions.projectName}
            onChange={(e) => setZipOptions({ ...zipOptions, projectName: e.target.value })}
            sx={{ mb: 3 }}
            helperText="Name for the project folder"
          />

          <Typography variant="subtitle2" gutterBottom>
            Include Additional Files:
          </Typography>

          <Box display="flex" flexDirection="column" gap={1}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={zipOptions.includeRequirements}
                  onChange={(e) => setZipOptions({ ...zipOptions, includeRequirements: e.target.checked })}
                />
              }
              label={
                <Box>
                  <Typography variant="body2">requirements.txt</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Python dependencies list
                  </Typography>
                </Box>
              }
            />

            <FormControlLabel
              control={
                <Checkbox
                  checked={zipOptions.includeReadme}
                  onChange={(e) => setZipOptions({ ...zipOptions, includeReadme: e.target.checked })}
                />
              }
              label={
                <Box>
                  <Typography variant="body2">README.md</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Project documentation
                  </Typography>
                </Box>
              }
            />

            <FormControlLabel
              control={
                <Checkbox
                  checked={zipOptions.includeDockerfile}
                  onChange={(e) => setZipOptions({ ...zipOptions, includeDockerfile: e.target.checked })}
                />
              }
              label={
                <Box>
                  <Typography variant="body2">Dockerfile</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Container configuration
                  </Typography>
                </Box>
              }
            />

            <FormControlLabel
              control={
                <Checkbox
                  checked={zipOptions.includeTests}
                  onChange={(e) => setZipOptions({ ...zipOptions, includeTests: e.target.checked })}
                />
              }
              label={
                <Box>
                  <Typography variant="body2">test_model.py</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Unit tests
                  </Typography>
                </Box>
              }
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setZipDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleZipDownload} variant="contained" startIcon={<DownloadIcon />}>
            Download ZIP
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default DownloadOptions;
