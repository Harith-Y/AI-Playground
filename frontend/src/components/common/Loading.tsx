import React from 'react';
import LoadingSpinner from './LoadingSpinner';

interface LoadingProps {
  fullscreen?: boolean;
  message?: string;
  size?: number;
}

const Loading: React.FC<LoadingProps> = ({
  fullscreen = false,
  message,
  size = 40,
}) => {
  return <LoadingSpinner fullscreen={fullscreen} message={message} size={size} />;
};

export default Loading;
