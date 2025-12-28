/**
 * PresetManager Component
 *
 * Manages saved configuration presets for quick selection
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Stack,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemSecondaryAction,
  Divider,
  Alert,
  Tooltip,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  Save as SaveIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import type { PresetManagerProps, DataSplitPreset } from '../../types/dataSplit';

const PresetManager: React.FC<PresetManagerProps> = ({
  presets,
  selectedPresetId,
  onPresetSelect,
  onPresetSave,
  onPresetDelete,
  disabled = false,
}) => {
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [presetName, setPresetName] = useState('');
  const [presetDescription, setPresetDescription] = useState('');
  const [saveError, setSaveError] = useState<string | null>(null);

  // Handle save preset
  const handleSavePreset = () => {
    if (!presetName.trim()) {
      setSaveError('Preset name is required');
      return;
    }

    // Check for duplicate names
    const isDuplicate = presets.some(
      (p) => p.name.toLowerCase() === presetName.trim().toLowerCase()
    );
    if (isDuplicate) {
      setSaveError('A preset with this name already exists');
      return;
    }

    // Get current config from selected preset or create new
    const selectedPreset = presets.find((p) => p.id === selectedPresetId);
    if (!selectedPreset) {
      setSaveError('No configuration selected to save');
      return;
    }

    onPresetSave({
      name: presetName.trim(),
      description: presetDescription.trim(),
      config: selectedPreset.config,
    });

    // Reset and close dialog
    setPresetName('');
    setPresetDescription('');
    setSaveError(null);
    setSaveDialogOpen(false);
  };

  // Handle delete preset
  const handleDelete = (presetId: string) => {
    const preset = presets.find((p) => p.id === presetId);
    if (preset?.isDefault) {
      return; // Cannot delete default presets
    }
    onPresetDelete(presetId);
  };

  return (
    <>
      <Card elevation={2}>
        <CardContent>
          {/* Header */}
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6" fontWeight="bold">
              Configuration Presets
            </Typography>
            <Button
              startIcon={<AddIcon />}
              onClick={() => setSaveDialogOpen(true)}
              disabled={disabled || !selectedPresetId}
              size="small"
              variant="outlined"
            >
              Save Preset
            </Button>
          </Box>

          {presets.length === 0 ? (
            <Alert severity="info">
              No presets available. Configure your split and save it as a preset for quick access.
            </Alert>
          ) : (
            <List disablePadding>
              {presets.map((preset, index) => (
                <React.Fragment key={preset.id}>
                  {index > 0 && <Divider />}
                  <ListItem
                    disablePadding
                    sx={{
                      bgcolor:
                        selectedPresetId === preset.id
                          ? 'action.selected'
                          : 'transparent',
                      borderRadius: 1,
                      mb: 0.5,
                    }}
                  >
                    <ListItemButton
                      onClick={() => onPresetSelect(preset.id)}
                      disabled={disabled}
                    >
                      <ListItemText
                        primary={
                          <Box display="flex" alignItems="center" gap={1}>
                            {preset.isDefault ? (
                              <Tooltip title="Default preset">
                                <StarIcon fontSize="small" color="primary" />
                              </Tooltip>
                            ) : (
                              <StarBorderIcon fontSize="small" sx={{ color: 'text.disabled' }} />
                            )}
                            <Typography variant="body1" fontWeight={preset.isDefault ? 600 : 400}>
                              {preset.name}
                            </Typography>
                            {selectedPresetId === preset.id && (
                              <Chip label="Active" size="small" color="primary" />
                            )}
                          </Box>
                        }
                        secondary={
                          <Box mt={0.5}>
                            <Typography variant="body2" color="text.secondary">
                              {preset.description}
                            </Typography>
                            <Stack direction="row" spacing={1} mt={0.5} flexWrap="wrap">
                              <Chip
                                label={`${preset.config.splitRatios.train}/${preset.config.splitRatios.validation}/${preset.config.splitRatios.test}`}
                                size="small"
                                variant="outlined"
                              />
                              {preset.config.crossValidation.enabled && (
                                <Chip
                                  label={`${preset.config.crossValidation.folds}-Fold CV`}
                                  size="small"
                                  variant="outlined"
                                  color="primary"
                                />
                              )}
                              {preset.config.randomSeed !== null && (
                                <Chip
                                  label={`Seed: ${preset.config.randomSeed}`}
                                  size="small"
                                  variant="outlined"
                                />
                              )}
                            </Stack>
                          </Box>
                        }
                      />
                      {!preset.isDefault && (
                        <ListItemSecondaryAction>
                          <Tooltip title="Delete preset">
                            <IconButton
                              edge="end"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDelete(preset.id);
                              }}
                              disabled={disabled}
                              size="small"
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </ListItemSecondaryAction>
                      )}
                    </ListItemButton>
                  </ListItem>
                </React.Fragment>
              ))}
            </List>
          )}
        </CardContent>
      </Card>

      {/* Save Preset Dialog */}
      <Dialog
        open={saveDialogOpen}
        onClose={() => {
          setSaveDialogOpen(false);
          setSaveError(null);
          setPresetName('');
          setPresetDescription('');
        }}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">Save Configuration Preset</Typography>
            <IconButton
              onClick={() => {
                setSaveDialogOpen(false);
                setSaveError(null);
              }}
              size="small"
            >
              <CloseIcon />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Stack spacing={2} mt={1}>
            {saveError && <Alert severity="error">{saveError}</Alert>}

            <TextField
              label="Preset Name"
              value={presetName}
              onChange={(e) => {
                setPresetName(e.target.value);
                setSaveError(null);
              }}
              fullWidth
              required
              placeholder="e.g., My Custom Split"
              autoFocus
            />

            <TextField
              label="Description"
              value={presetDescription}
              onChange={(e) => setPresetDescription(e.target.value)}
              fullWidth
              multiline
              rows={3}
              placeholder="Describe when to use this preset..."
            />

            <Alert severity="info">
              The current configuration will be saved as this preset.
            </Alert>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => {
              setSaveDialogOpen(false);
              setSaveError(null);
              setPresetName('');
              setPresetDescription('');
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSavePreset}
            variant="contained"
            startIcon={<SaveIcon />}
          >
            Save Preset
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default PresetManager;
