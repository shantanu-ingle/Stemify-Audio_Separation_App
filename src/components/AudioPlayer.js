import React from 'react';
import './AudioPlayer.css';

const AudioPlayer = ({ title, src }) => {
  return (
    <div className="audio-player">
      <h3 className="audio-title">{title}</h3>
      {src ? (
        <audio controls src={src} className="audio-control" />
      ) : (
        <p className="no-audio">No audio available</p>
      )}
      {src && (
        <a href={src} download={`${title}.wav`} className="download-button">
          Download
        </a>
      )}
    </div>
  );
};

export default AudioPlayer;
