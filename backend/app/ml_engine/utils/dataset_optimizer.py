"""
Dataset optimization utilities for handling large datasets efficiently.

This module provides memory-efficient operations for:
- Large CSV file processing
- Chunked data loading
- Incremental preprocessing
- Streaming transformations
"""

import pandas as pd
import numpy as np
from typing import Iterator, Optional, Tuple, Dict, Any, List, Callable
from pathlib import Path
import os
import gc
from dataclasses import dataclass
from enum import Enum

from app.utils.logger import get_logger
from app.utils.memory_manager import get_memory_monitor, MemoryOptimizer, memory_profiler

logger = get_logger(__name__)


class DatasetSize(Enum):
    """Dataset size categories for optimization routing."""
    SMALL = "small"  # < 100MB, < 100K rows
    MEDIUM = "medium"  # 100MB-1GB, 100K-1M rows
    LARGE = "large"  # 1GB-10GB, 1M-10M rows
    VERY_LARGE = "very_large"  # > 10GB, > 10M rows


@dataclass
class DatasetMetrics:
    """Dataset size and memory metrics."""
    file_size_bytes: int
    file_size_mb: float
    estimated_rows: Optional[int] = None
    estimated_memory_mb: Optional[float] = None
    size_category: DatasetSize = DatasetSize.SMALL

    @classmethod
    def from_file(cls, file_path: str, sample_rows: int = 1000) -> 'DatasetMetrics':
        """
        Analyze dataset file and determine size category.

        Args:
            file_path: Path to the dataset file
            sample_rows: Number of rows to sample for estimation

        Returns:
            DatasetMetrics object with size information
        """
        path = Path(file_path)
        file_size = path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        # Sample file to estimate row count and memory usage
        try:
            sample_df = pd.read_csv(file_path, nrows=sample_rows)
            bytes_per_row = sample_df.memory_usage(deep=True).sum() / len(sample_df)

            # Estimate total rows by sampling
            with open(file_path, 'r', encoding='utf-8') as f:
                # Count lines in sample
                sample_lines = sum(1 for _ in f)

            # Read small chunk to get accurate row estimate
            chunk_size = min(10000, sample_lines)
            chunk_df = pd.read_csv(file_path, nrows=chunk_size)
            actual_bytes_per_row = chunk_df.memory_usage(deep=True).sum() / len(chunk_df)

            # Estimate total rows from file size
            avg_bytes_per_line = file_size / sample_lines
            estimated_rows = int(file_size / avg_bytes_per_line)
            estimated_memory_mb = (estimated_rows * actual_bytes_per_row) / (1024 * 1024)

            del sample_df, chunk_df
            gc.collect()

        except Exception as e:
            logger.warning(f"Could not estimate dataset metrics: {e}")
            estimated_rows = None
            estimated_memory_mb = file_size_mb * 2  # Conservative estimate

        # Determine size category
        if file_size_mb < 100 or (estimated_rows and estimated_rows < 100000):
            size_category = DatasetSize.SMALL
        elif file_size_mb < 1024 or (estimated_rows and estimated_rows < 1000000):
            size_category = DatasetSize.MEDIUM
        elif file_size_mb < 10240 or (estimated_rows and estimated_rows < 10000000):
            size_category = DatasetSize.LARGE
        else:
            size_category = DatasetSize.VERY_LARGE

        logger.info(
            f"Dataset metrics: {file_size_mb:.2f}MB, "
            f"~{estimated_rows:,} rows, "
            f"~{estimated_memory_mb:.2f}MB in memory, "
            f"category: {size_category.value}"
        )

        return cls(
            file_size_bytes=file_size,
            file_size_mb=file_size_mb,
            estimated_rows=estimated_rows,
            estimated_memory_mb=estimated_memory_mb,
            size_category=size_category
        )


class ChunkedDataLoader:
    """
    Memory-efficient data loader for large datasets.

    Reads data in chunks to avoid loading entire dataset into memory.
    Supports preprocessing and transformation on each chunk.
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 10000,
        dtype: Optional[Dict[str, Any]] = None,
        usecols: Optional[List[str]] = None
    ):
        """
        Initialize chunked data loader.

        Args:
            file_path: Path to CSV file
            chunk_size: Number of rows per chunk
            dtype: Column data types for optimization
            usecols: Columns to load (None = all columns)
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.usecols = usecols

    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        """
        Iterate over dataset chunks with memory monitoring.

        Yields:
            DataFrame chunks
        """
        logger.info(f"Reading {self.file_path} in chunks of {self.chunk_size}")

        memory_monitor = get_memory_monitor()

        try:
            chunk_reader = pd.read_csv(
                self.file_path,
                chunksize=self.chunk_size,
                dtype=self.dtype,
                usecols=self.usecols,
                low_memory=True
            )

            for i, chunk in enumerate(chunk_reader):
                logger.debug(f"Processing chunk {i+1}, shape: {chunk.shape}")

                # Optimize chunk memory
                chunk = MemoryOptimizer.optimize_dataframe_memory(chunk, aggressive=False)

                # Check memory usage periodically
                if i % 10 == 0:
                    snapshot = memory_monitor.get_current_snapshot()
                    if snapshot.percent > 85.0:
                        logger.warning(f"High memory usage detected: {snapshot.percent:.1f}% - Running GC")
                        MemoryOptimizer.aggressive_gc()

                yield chunk

        except Exception as e:
            logger.error(f"Error reading chunks: {e}", exc_info=True)
            raise

    def process_chunks(
        self,
        transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
        output_path: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Process dataset in chunks with a transformation function.

        Args:
            transform_fn: Function to apply to each chunk
            output_path: Path to save processed data (None = return combined)

        Returns:
            Combined DataFrame if output_path is None, else None
        """
        chunks_processed = []
        total_rows = 0

        with memory_profiler("Chunked Data Processing"):
            for chunk in self.iter_chunks():
                try:
                    # Apply transformation
                    processed_chunk = transform_fn(chunk)
                    total_rows += len(processed_chunk)

                    if output_path:
                        # Write to file incrementally
                        mode = 'w' if total_rows == len(processed_chunk) else 'a'
                        header = total_rows == len(processed_chunk)
                        processed_chunk.to_csv(
                            output_path,
                            mode=mode,
                            header=header,
                            index=False
                        )
                    else:
                        # Store in memory (use with caution for large datasets)
                        chunks_processed.append(processed_chunk)

                    # Clean up
                    del chunk, processed_chunk
                    gc.collect()

                except Exception as e:
                    logger.error(f"Error processing chunk: {e}", exc_info=True)
                    raise

        logger.info(f"Processed {total_rows:,} total rows")

        if output_path:
            logger.info(f"Saved processed data to {output_path}")
            return None
        else:
            result = pd.concat(chunks_processed, ignore_index=True)
            del chunks_processed
            gc.collect()
            return result

    def sample_data(self, n: int = 10000, random_state: int = 42) -> pd.DataFrame:
        """
        Create a random sample from large dataset.

        Uses reservoir sampling for memory efficiency.

        Args:
            n: Number of samples to draw
            random_state: Random seed

        Returns:
            Sampled DataFrame
        """
        np.random.seed(random_state)
        reservoir = []
        total_rows = 0

        for chunk in self.iter_chunks():
            for idx, row in chunk.iterrows():
                total_rows += 1

                if len(reservoir) < n:
                    reservoir.append(row)
                else:
                    # Reservoir sampling algorithm
                    j = np.random.randint(0, total_rows)
                    if j < n:
                        reservoir[j] = row

        sample_df = pd.DataFrame(reservoir)
        logger.info(f"Sampled {len(sample_df):,} rows from {total_rows:,} total rows")
        return sample_df


class MemoryEfficientPreprocessor:
    """
    Memory-efficient preprocessing for large datasets.

    Applies preprocessing steps in chunks and manages memory usage.
    """

    def __init__(self, chunk_size: int = 10000):
        """
        Initialize memory-efficient preprocessor.

        Args:
            chunk_size: Number of rows to process at once
        """
        self.chunk_size = chunk_size
        self.fitted_stats = {}

    def fit_transform_chunked(
        self,
        file_path: str,
        output_path: str,
        steps: List[Tuple[str, Callable]],
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fit and transform data in chunks with memory profiling.

        Args:
            file_path: Input CSV path
            output_path: Output CSV path
            steps: List of (name, transform_function) tuples
            target_column: Target column to preserve

        Returns:
            Dictionary with transformation statistics
        """
        logger.info(f"Starting chunked fit_transform: {file_path} -> {output_path}")

        with memory_profiler("Memory-Efficient Preprocessing"):
            loader = ChunkedDataLoader(file_path, chunk_size=self.chunk_size)

            # First pass: Fit transformers (e.g., calculate means, medians)
            logger.info("Pass 1: Fitting transformers")
            self._fit_chunked(loader, steps, target_column)

            # Second pass: Transform data
            logger.info("Pass 2: Transforming data")
            stats = self._transform_chunked(loader, output_path, steps, target_column)

        logger.info("Chunked fit_transform completed")
        return stats

    def _fit_chunked(
        self,
        loader: ChunkedDataLoader,
        steps: List[Tuple[str, Callable]],
        target_column: Optional[str]
    ):
        """Fit transformers on chunks."""
        for step_name, step_fn in steps:
            logger.info(f"Fitting step: {step_name}")

            # Collect statistics from each chunk
            chunk_stats = []

            for chunk in loader.iter_chunks():
                if target_column and target_column in chunk.columns:
                    X = chunk.drop(columns=[target_column])
                else:
                    X = chunk

                # Extract statistics from chunk
                stats = self._extract_chunk_stats(X, step_name)
                chunk_stats.append(stats)

                del chunk, X
                gc.collect()

            # Combine statistics from all chunks
            self.fitted_stats[step_name] = self._combine_chunk_stats(
                chunk_stats,
                step_name
            )

            logger.info(f"Completed fitting: {step_name}")

    def _transform_chunked(
        self,
        loader: ChunkedDataLoader,
        output_path: str,
        steps: List[Tuple[str, Callable]],
        target_column: Optional[str]
    ) -> Dict[str, Any]:
        """Transform data in chunks and write to file."""
        total_rows = 0
        first_chunk = True

        for chunk in loader.iter_chunks():
            # Separate features and target
            if target_column and target_column in chunk.columns:
                y = chunk[target_column]
                X = chunk.drop(columns=[target_column])
            else:
                y = None
                X = chunk

            # Apply transformation steps
            for step_name, step_fn in steps:
                X = self._apply_transform(X, step_name, step_fn)

            # Recombine with target
            if y is not None:
                X[target_column] = y

            # Write to output file
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            X.to_csv(output_path, mode=mode, header=header, index=False)

            total_rows += len(X)
            first_chunk = False

            del chunk, X, y
            gc.collect()

        return {
            'total_rows': total_rows,
            'output_path': output_path,
            'steps_applied': [name for name, _ in steps]
        }

    def _extract_chunk_stats(
        self,
        chunk: pd.DataFrame,
        step_name: str
    ) -> Dict[str, Any]:
        """Extract statistics from a chunk for fitting."""
        numeric_cols = chunk.select_dtypes(include=[np.number]).columns

        stats = {
            'count': len(chunk),
            'numeric_columns': list(numeric_cols),
            'means': chunk[numeric_cols].mean().to_dict() if len(numeric_cols) > 0 else {},
            'stds': chunk[numeric_cols].std().to_dict() if len(numeric_cols) > 0 else {},
            'mins': chunk[numeric_cols].min().to_dict() if len(numeric_cols) > 0 else {},
            'maxs': chunk[numeric_cols].max().to_dict() if len(numeric_cols) > 0 else {},
        }

        return stats

    def _combine_chunk_stats(
        self,
        chunk_stats: List[Dict[str, Any]],
        step_name: str
    ) -> Dict[str, Any]:
        """Combine statistics from multiple chunks."""
        if not chunk_stats:
            return {}

        # Calculate weighted averages for means and stds
        total_count = sum(s['count'] for s in chunk_stats)

        combined = {
            'total_count': total_count,
            'numeric_columns': chunk_stats[0]['numeric_columns'],
            'means': {},
            'stds': {},
            'mins': {},
            'maxs': {}
        }

        # Combine means (weighted average)
        for col in combined['numeric_columns']:
            weighted_mean = sum(
                s['means'].get(col, 0) * s['count']
                for s in chunk_stats
            ) / total_count
            combined['means'][col] = weighted_mean

        # Combine mins and maxs
        for col in combined['numeric_columns']:
            combined['mins'][col] = min(
                s['mins'].get(col, float('inf'))
                for s in chunk_stats
            )
            combined['maxs'][col] = max(
                s['maxs'].get(col, float('-inf'))
                for s in chunk_stats
            )

        # Combine stds (using pooled standard deviation)
        for col in combined['numeric_columns']:
            # Simplified: use max std as approximation
            # For exact pooled std, would need sum of squared deviations
            combined['stds'][col] = max(
                s['stds'].get(col, 0)
                for s in chunk_stats
            )

        return combined

    def _apply_transform(
        self,
        X: pd.DataFrame,
        step_name: str,
        step_fn: Callable
    ) -> pd.DataFrame:
        """Apply fitted transformation to chunk."""
        if step_name in self.fitted_stats:
            stats = self.fitted_stats[step_name]
            return step_fn(X, stats)
        else:
            return step_fn(X)


def get_optimal_chunk_size(
    dataset_metrics: DatasetMetrics,
    available_memory_gb: float = 4.0
) -> int:
    """
    Calculate optimal chunk size based on dataset and memory constraints.

    Args:
        dataset_metrics: Dataset size metrics
        available_memory_gb: Available memory in GB

    Returns:
        Optimal chunk size (number of rows)
    """
    # Allocate 50% of available memory for chunk processing
    target_memory_mb = (available_memory_gb * 1024) * 0.5

    if dataset_metrics.estimated_memory_mb and dataset_metrics.estimated_rows:
        bytes_per_row = (dataset_metrics.estimated_memory_mb * 1024 * 1024) / dataset_metrics.estimated_rows
        optimal_rows = int((target_memory_mb * 1024 * 1024) / bytes_per_row)
    else:
        # Conservative default
        optimal_rows = 10000

    # Clamp to reasonable range
    min_chunk = 1000
    max_chunk = 1000000

    chunk_size = max(min_chunk, min(optimal_rows, max_chunk))

    logger.info(f"Optimal chunk size: {chunk_size:,} rows (target: {target_memory_mb:.0f}MB)")

    return chunk_size


def should_use_chunked_processing(dataset_metrics: DatasetMetrics) -> bool:
    """
    Determine if chunked processing should be used.

    Args:
        dataset_metrics: Dataset size metrics

    Returns:
        True if chunked processing is recommended
    """
    # Use chunked processing for LARGE and VERY_LARGE datasets
    use_chunked = dataset_metrics.size_category in [
        DatasetSize.LARGE,
        DatasetSize.VERY_LARGE
    ]

    # Also use if estimated memory exceeds 2GB
    if dataset_metrics.estimated_memory_mb and dataset_metrics.estimated_memory_mb > 2048:
        use_chunked = True

    logger.info(
        f"Chunked processing {'recommended' if use_chunked else 'not needed'} "
        f"for {dataset_metrics.size_category.value} dataset"
    )

    return use_chunked


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes to reduce memory usage.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with optimized dtypes
    """
    initial_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == 'object':
            # Check if column should be categorical
            num_unique = df[col].nunique()
            num_total = len(df[col])

            if num_unique / num_total < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')

        elif col_type == 'int64':
            # Downcast integers
            c_min = df[col].min()
            c_max = df[col].max()

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        elif col_type == 'float64':
            # Downcast floats
            df[col] = df[col].astype(np.float32)

    final_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    reduction = ((initial_memory - final_memory) / initial_memory) * 100

    logger.info(
        f"Memory optimized: {initial_memory:.2f}MB -> {final_memory:.2f}MB "
        f"({reduction:.1f}% reduction)"
    )

    return df
