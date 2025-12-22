import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import { ApiResponse, ApiError, UploadProgress } from '../types/api';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000; // 30 seconds

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request Interceptor - Add auth token to requests
apiClient.interceptors.request.use(
  (config) => {
    // Get token from localStorage
    const token = localStorage.getItem('accessToken');
    
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response Interceptor - Handle responses and errors
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  async (error: AxiosError) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };

    // Handle 401 Unauthorized - Token expired
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        // Try to refresh token
        const refreshToken = localStorage.getItem('refreshToken');
        
        if (refreshToken) {
          const response = await axios.post(`${API_BASE_URL}/api/v1/auth/refresh`, {
            refreshToken,
          });

          const { accessToken } = response.data;
          localStorage.setItem('accessToken', accessToken);

          // Retry original request with new token
          if (originalRequest.headers) {
            originalRequest.headers.Authorization = `Bearer ${accessToken}`;
          }
          
          return apiClient(originalRequest);
        }
      } catch (refreshError) {
        // Refresh failed, redirect to login
        localStorage.removeItem('accessToken');
        localStorage.removeItem('refreshToken');
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);

// API Error Handler
export const handleApiError = (error: any): ApiError => {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<ApiError>;
    
    if (axiosError.response) {
      // Server responded with error
      return {
        message: axiosError.response.data?.message || 'An error occurred',
        statusCode: axiosError.response.status,
        errors: axiosError.response.data?.errors,
      };
    } else if (axiosError.request) {
      // Request made but no response
      return {
        message: 'No response from server. Please check your connection.',
        statusCode: 0,
      };
    }
  }
  
  // Generic error
  return {
    message: error.message || 'An unexpected error occurred',
    statusCode: 500,
  };
};

// Generic API Methods
class ApiService {
  /**
   * GET request
   */
  async get<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
    try {
      const response = await apiClient.get<ApiResponse<T>>(url, config);
      return response.data.data;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  /**
   * POST request
   */
  async post<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<T> {
    try {
      const response = await apiClient.post<ApiResponse<T>>(url, data, config);
      return response.data.data;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  /**
   * PUT request
   */
  async put<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<T> {
    try {
      const response = await apiClient.put<ApiResponse<T>>(url, data, config);
      return response.data.data;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  /**
   * PATCH request
   */
  async patch<T = any>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig
  ): Promise<T> {
    try {
      const response = await apiClient.patch<ApiResponse<T>>(url, data, config);
      return response.data.data;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  /**
   * DELETE request
   */
  async delete<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
    try {
      const response = await apiClient.delete<ApiResponse<T>>(url, config);
      return response.data.data;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  /**
   * Upload file with progress tracking
   */
  async upload<T = any>(
    url: string,
    file: File,
    onProgress?: (progress: UploadProgress) => void,
    additionalData?: Record<string, any>
  ): Promise<T> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      // Add any additional data
      if (additionalData) {
        Object.keys(additionalData).forEach((key) => {
          formData.append(key, additionalData[key]);
        });
      }

      const response = await apiClient.post<ApiResponse<T>>(url, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const percentage = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            onProgress({
              loaded: progressEvent.loaded,
              total: progressEvent.total,
              percentage,
            });
          }
        },
      });

      return response.data.data;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  /**
   * Download file
   */
  async download(url: string, filename: string): Promise<void> {
    try {
      const response = await apiClient.get(url, {
        responseType: 'blob',
      });

      // Create blob link to download
      const blobUrl = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = blobUrl;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
      window.URL.revokeObjectURL(blobUrl);
    } catch (error) {
      throw handleApiError(error);
    }
  }
}

// Export singleton instance
export const api = new ApiService();

// Export axios instance for custom requests
export { apiClient };

// Export default
export default api;

