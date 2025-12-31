/**
 * Share Experiment Component
 * 
 * Provides options to share experiment and generated code
 */

import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Box,
  Typography,
  IconButton,
  Tooltip,
  Alert,
  Tabs,
  Tab,
  InputAdornment,
  Snackbar,
} from '@mui/material';
import {
  Share as ShareIcon,
  ContentCopy as ContentCopyIcon,
  Link as LinkIcon,
  Email as EmailIcon,
  Close as CloseIcon,
  Check as CheckIcon,
} from '@mui/icons-material';

interface ShareExperimentProps {
  open: boolean;
  onClose: () => void;
  experimentName: string;
  experimentId?: string;
  shareUrl?: string;
}

const ShareExperiment: React.FC<ShareExperimentProps> = ({
  open,
  onClose,
  experimentName,
  experimentId,
  shareUrl,
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [copied, setCopied] = useState(false);
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');

  // Generate share URL
  const generatedUrl = shareUrl || `${window.location.origin}/experiments/${experimentId || 'demo'}`;

  const handleCopyLink = async () => {
    await navigator.clipboard.writeText(generatedUrl);
    setCopied(true);
  };

  const handleSendEmail = () => {
    const subject = encodeURIComponent(`Check out this ML experiment: ${experimentName}`);
    const body = encodeURIComponent(
      `${message}\n\nView the experiment here: ${generatedUrl}`
    );
    window.location.href = `mailto:${email}?subject=${subject}&body=${body}`;
  };

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: experimentName,
          text: `Check out this ML experiment: ${experimentName}`,
          url: generatedUrl,
        });
      } catch (err) {
        console.error('Error sharing:', err);
      }
    }
  };

  return (
    <>
      <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
        <DialogTitle>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Box display="flex" alignItems="center" gap={1}>
              <ShareIcon />
              Share Experiment
            </Box>
            <IconButton size="small" onClick={onClose}>
              <CloseIcon />
            </IconButton>
          </Box>
        </DialogTitle>

        <DialogContent>
          <Alert severity="info" sx={{ mb: 3 }}>
            Share your experiment and generated code with others
          </Alert>

          <Tabs value={activeTab} onChange={(_e, v) => setActiveTab(v)} sx={{ mb: 3 }}>
            <Tab icon={<LinkIcon />} label="Link" />
            <Tab icon={<EmailIcon />} label="Email" />
          </Tabs>

          {/* Link Tab */}
          {activeTab === 0 && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Share Link
              </Typography>
              <TextField
                fullWidth
                value={generatedUrl}
                InputProps={{
                  readOnly: true,
                  endAdornment: (
                    <InputAdornment position="end">
                      <Tooltip title={copied ? 'Copied!' : 'Copy link'}>
                        <IconButton onClick={handleCopyLink} edge="end">
                          {copied ? <CheckIcon color="success" /> : <ContentCopyIcon />}
                        </IconButton>
                      </Tooltip>
                    </InputAdornment>
                  ),
                }}
                sx={{ mb: 2 }}
              />

              <Typography variant="body2" color="text.secondary" paragraph>
                Anyone with this link can view the experiment details and download the generated code.
              </Typography>

              {typeof navigator !== 'undefined' && 'share' in navigator && (
                <Button
                  variant="outlined"
                  startIcon={<ShareIcon />}
                  onClick={handleShare}
                  fullWidth
                >
                  Share via...
                </Button>
              )}
            </Box>
          )}

          {/* Email Tab */}
          {activeTab === 1 && (
            <Box>
              <TextField
                fullWidth
                label="Recipient Email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="colleague@example.com"
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="Message (Optional)"
                multiline
                rows={4}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Add a personal message..."
                sx={{ mb: 2 }}
              />

              <Typography variant="caption" color="text.secondary">
                The experiment link will be automatically included in the email.
              </Typography>
            </Box>
          )}
        </DialogContent>

        <DialogActions>
          <Button onClick={onClose}>Cancel</Button>
          {activeTab === 0 ? (
            <Button
              variant="contained"
              startIcon={<ContentCopyIcon />}
              onClick={handleCopyLink}
            >
              Copy Link
            </Button>
          ) : (
            <Button
              variant="contained"
              startIcon={<EmailIcon />}
              onClick={handleSendEmail}
              disabled={!email}
            >
              Send Email
            </Button>
          )}
        </DialogActions>
      </Dialog>

      <Snackbar
        open={copied}
        autoHideDuration={2000}
        onClose={() => setCopied(false)}
        message="Link copied to clipboard"
      />
    </>
  );
};

export default ShareExperiment;
