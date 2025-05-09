import React from 'react';
import './FileUpload.css';

const FileUpload = ({ onFileUpload, error }) => {
  return (
    <div className="file-upload">
      <input
        type="file"
        accept=".mp3,.wav"
        onChange={onFileUpload}
        id="file-upload"
        className="file-input"
      />
      <label htmlFor="file-upload" className="upload-button">
        Browse my files
      </label>
      {error && <p className="error">{error}</p>}
    </div>
  );
};

export default FileUpload;
