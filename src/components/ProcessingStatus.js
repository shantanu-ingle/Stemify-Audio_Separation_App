import React from 'react';
import './ProcessingStatus.css';

const ProcessingStatus = () => {
  return (
    <div className="processing-status">
      <div className="loader"></div>
      <p>Processing your audio... This might take a while.</p>
    </div>
  );
};

export default ProcessingStatus;
