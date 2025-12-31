/**
 * Performance Utilities
 * 
 * Helper functions for performance optimization
 */

/**
 * Debounce function calls
 */
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: number | null = null;
  
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = window.setTimeout(() => func(...args), wait);
  };
};

/**
 * Throttle function calls
 */
export const throttle = <T extends (...args: any[]) => any>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean;
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
};

/**
 * Lazy load images
 */
export const lazyLoadImage = (src: string): Promise<string> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(src);
    img.onerror = reject;
    img.src = src;
  });
};

/**
 * Measure component render time
 */
export const measureRenderTime = (componentName: string, callback: () => void): void => {
  const start = performance.now();
  callback();
  const end = performance.now();
  console.log(`${componentName} render time: ${(end - start).toFixed(2)}ms`);
};

/**
 * Check if element is in viewport
 */
export const isInViewport = (element: HTMLElement): boolean => {
  const rect = element.getBoundingClientRect();
  return (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
  );
};

/**
 * Intersection Observer for lazy loading
 */
export const createIntersectionObserver = (
  callback: (entry: IntersectionObserverEntry) => void,
  options?: IntersectionObserverInit
): IntersectionObserver => {
  return new IntersectionObserver((entries) => {
    entries.forEach(callback);
  }, options);
};

/**
 * Memoize expensive function calls
 */
export const memoize = <T extends (...args: any[]) => any>(fn: T): T => {
  const cache = new Map();
  
  return ((...args: Parameters<T>): ReturnType<T> => {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      return cache.get(key);
    }
    const result = fn(...args);
    cache.set(key, result);
    return result;
  }) as T;
};

/**
 * Request idle callback wrapper
 */
export const runWhenIdle = (callback: () => void, timeout = 2000): void => {
  if ('requestIdleCallback' in window) {
    requestIdleCallback(callback, { timeout });
  } else {
    setTimeout(callback, 1);
  }
};

/**
 * Batch DOM updates
 */
export const batchDOMUpdates = (updates: (() => void)[]): void => {
  requestAnimationFrame(() => {
    updates.forEach(update => update());
  });
};

/**
 * Preload resources
 */
export const preloadResource = (url: string, type: 'script' | 'style' | 'image'): void => {
  const link = document.createElement('link');
  link.rel = 'preload';
  link.href = url;
  link.as = type;
  document.head.appendChild(link);
};

/**
 * Get bundle size
 */
export const getBundleSize = async (): Promise<number> => {
  if ('performance' in window && 'getEntriesByType' in performance) {
    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
    return resources.reduce((total, resource) => total + (resource.transferSize || 0), 0);
  }
  return 0;
};

/**
 * Monitor FPS
 */
export const monitorFPS = (callback: (fps: number) => void): (() => void) => {
  let lastTime = performance.now();
  let frames = 0;
  let rafId: number;
  
  const tick = () => {
    frames++;
    const currentTime = performance.now();
    
    if (currentTime >= lastTime + 1000) {
      const fps = Math.round((frames * 1000) / (currentTime - lastTime));
      callback(fps);
      frames = 0;
      lastTime = currentTime;
    }
    
    rafId = requestAnimationFrame(tick);
  };
  
  rafId = requestAnimationFrame(tick);
  
  return () => cancelAnimationFrame(rafId);
};

/**
 * Chunk large arrays for processing
 */
export const chunkArray = <T>(array: T[], size: number): T[][] => {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
};

/**
 * Process array in chunks with delay
 */
export const processInChunks = async <T, R>(
  array: T[],
  processor: (item: T) => R,
  chunkSize = 100,
  delay = 0
): Promise<R[]> => {
  const results: R[] = [];
  const chunks = chunkArray(array, chunkSize);
  
  for (const chunk of chunks) {
    results.push(...chunk.map(processor));
    if (delay > 0) {
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  
  return results;
};

export default {
  debounce,
  throttle,
  lazyLoadImage,
  measureRenderTime,
  isInViewport,
  createIntersectionObserver,
  memoize,
  runWhenIdle,
  batchDOMUpdates,
  preloadResource,
  getBundleSize,
  monitorFPS,
  chunkArray,
  processInChunks,
};
