import React, { useState } from 'react';
import { Button, CircularProgress, Snackbar, Alert, Box } from '@mui/material';
import axios from 'axios';

function UploadExcel({ onUploadSuccess }) {
  const [loading, setLoading] = useState(false);
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [message, setMessage] = useState('');
  const [severity, setSeverity] = useState('success');

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
      const response = await axios.post('http://127.0.0.1:8000/upload_stock', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setMessage(response.data.message || 'File uploaded successfully!');
      setSeverity('success');
      setOpenSnackbar(true);
      if (onUploadSuccess) onUploadSuccess();
    } catch (error) {
      console.error('Error uploading file:', error);
      setMessage(error.response?.data?.error || 'Failed to upload file.');
      setSeverity('error');
      setOpenSnackbar(true);
    } finally {
      setLoading(false);
      // Reset input value so same file can be uploaded again
      event.target.value = null;
    }
  };

  return (
    <Box sx={{ mb: 2 }}>
      <Button
        variant="contained"
        component="label"
        disabled={loading}
        color="secondary"
      >
        {loading ? <CircularProgress size={24} color="inherit" /> : 'Upload Excel Data'}
        <input
          type="file"
          hidden
          accept=".csv, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, application/vnd.ms-excel"
          onChange={handleFileChange}
        />
      </Button>

      <Snackbar 
        open={openSnackbar} 
        autoHideDuration={6000} 
        onClose={() => setOpenSnackbar(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={() => setOpenSnackbar(false)} severity={severity} sx={{ width: '100%' }}>
          {message}
        </Alert>
      </Snackbar>
    </Box>
  );
}

export default UploadExcel;
