import { useEffect, useRef, useState, useCallback } from 'react';

export interface UseWebSocketOptions {
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
  onMessage?: (event: MessageEvent) => void;
  reconnectAttempts?: number;
  reconnectInterval?: number;
  shouldReconnect?: boolean;
}

export interface UseWebSocketReturn {
  isConnected: boolean;
  lastMessage: string | null;
  sendMessage: (message: string | object) => void;
  disconnect: () => void;
  reconnect: () => void;
  connectionState: 'connecting' | 'connected' | 'disconnected' | 'error';
}

export const useWebSocket = (
  url: string | null,
  options: UseWebSocketOptions = {}
): UseWebSocketReturn => {
  const {
    onOpen,
    onClose,
    onError,
    onMessage,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
    shouldReconnect = true,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<string | null>(null);
  const [connectionState, setConnectionState] = useState<
    'connecting' | 'connected' | 'disconnected' | 'error'
  >('disconnected');

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectCountRef = useRef(0);
  const shouldConnectRef = useRef(true);

  const disconnect = useCallback(() => {
    shouldConnectRef.current = false;
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setConnectionState('disconnected');
  }, []);

  const connect = useCallback(() => {
    if (!url || !shouldConnectRef.current) {
      return;
    }

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      setConnectionState('connecting');
      const ws = new WebSocket(url);

      ws.onopen = (event) => {
        console.log('[WebSocket] Connected:', url);
        setIsConnected(true);
        setConnectionState('connected');
        reconnectCountRef.current = 0;
        onOpen?.(event);
      };

      ws.onclose = (event) => {
        console.log('[WebSocket] Disconnected:', event.code, event.reason);
        setIsConnected(false);
        setConnectionState('disconnected');
        wsRef.current = null;
        onClose?.(event);

        // Attempt reconnection
        if (
          shouldReconnect &&
          shouldConnectRef.current &&
          reconnectCountRef.current < reconnectAttempts
        ) {
          reconnectCountRef.current += 1;
          console.log(
            `[WebSocket] Reconnecting... (${reconnectCountRef.current}/${reconnectAttempts})`
          );
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = (event) => {
        console.error('[WebSocket] Error:', event);
        setConnectionState('error');
        onError?.(event);
      };

      ws.onmessage = (event) => {
        console.log('[WebSocket] Message received:', event.data);
        setLastMessage(event.data);
        onMessage?.(event);
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('[WebSocket] Connection error:', error);
      setConnectionState('error');
    }
  }, [url, onOpen, onClose, onError, onMessage, shouldReconnect, reconnectAttempts, reconnectInterval]);

  const reconnect = useCallback(() => {
    reconnectCountRef.current = 0;
    shouldConnectRef.current = true;
    connect();
  }, [connect]);

  const sendMessage = useCallback((message: string | object) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const data = typeof message === 'string' ? message : JSON.stringify(message);
      wsRef.current.send(data);
      console.log('[WebSocket] Message sent:', data);
    } else {
      console.warn('[WebSocket] Cannot send message - not connected');
    }
  }, []);

  // Connect on mount and when URL changes
  useEffect(() => {
    shouldConnectRef.current = true;
    connect();

    return () => {
      disconnect();
    };
  }, [url, connect, disconnect]);

  return {
    isConnected,
    lastMessage,
    sendMessage,
    disconnect,
    reconnect,
    connectionState,
  };
};
