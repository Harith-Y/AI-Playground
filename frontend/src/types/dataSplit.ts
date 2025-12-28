/**
 * Data Split Configuration Types
 *
 * Types for train/validation/test split configuration with cross-validation support
 */

/**
 * Split strategy type
 */
export const SplitStrategy = {
  HOLDOUT: 'holdout',
  CROSS_VALIDATION: 'cross_validation',
} as const;

export type SplitStrategy = (typeof SplitStrategy)[keyof typeof SplitStrategy];

/**
 * Data split ratios for holdout strategy
 */
export interface SplitRatios {
  train: number;
  validation: number;
  test: number;
}

/**
 * Cross-validation configuration
 */
export interface CrossValidationConfig {
  enabled: boolean;
  folds: number;
  shuffle: boolean;
}

/**
 * Complete data split configuration
 */
export interface DataSplitConfig {
  strategy: SplitStrategy;
  splitRatios: SplitRatios;
  randomSeed: number | null;
  crossValidation: CrossValidationConfig;
  stratify: boolean; // Whether to stratify splits (for classification)
}

/**
 * Preset configuration for quick selection
 */
export interface DataSplitPreset {
  id: string;
  name: string;
  description: string;
  config: DataSplitConfig;
  isDefault?: boolean;
  createdAt?: string;
}

/**
 * Validation result for data split configuration
 */
export interface DataSplitValidation {
  isValid: boolean;
  errors: {
    splitRatios?: string;
    randomSeed?: string;
    crossValidation?: string;
  };
  warnings: {
    splitRatios?: string;
    randomSeed?: string;
    crossValidation?: string;
  };
}

/**
 * Props for DataSplitConfig component
 */
export interface DataSplitConfigProps {
  config: DataSplitConfig;
  onChange: (config: DataSplitConfig) => void;
  onValidationChange?: (validation: DataSplitValidation) => void;
  disabled?: boolean;
  showStratify?: boolean; // Show stratify option (for classification tasks)
}

/**
 * Props for preset management
 */
export interface PresetManagerProps {
  presets: DataSplitPreset[];
  selectedPresetId: string | null;
  onPresetSelect: (presetId: string) => void;
  onPresetSave: (preset: Omit<DataSplitPreset, 'id' | 'createdAt'>) => void;
  onPresetDelete: (presetId: string) => void;
  disabled?: boolean;
}

/**
 * Default presets
 */
export const DEFAULT_PRESETS: DataSplitPreset[] = [
  {
    id: 'default-70-15-15',
    name: '70-15-15 Split',
    description: 'Standard 70% train, 15% validation, 15% test split',
    isDefault: true,
    config: {
      strategy: SplitStrategy.HOLDOUT,
      splitRatios: { train: 70, validation: 15, test: 15 },
      randomSeed: 42,
      crossValidation: { enabled: false, folds: 5, shuffle: true },
      stratify: true,
    },
  },
  {
    id: 'default-80-10-10',
    name: '80-10-10 Split',
    description: 'Common 80% train, 10% validation, 10% test split',
    isDefault: true,
    config: {
      strategy: SplitStrategy.HOLDOUT,
      splitRatios: { train: 80, validation: 10, test: 10 },
      randomSeed: 42,
      crossValidation: { enabled: false, folds: 5, shuffle: true },
      stratify: true,
    },
  },
  {
    id: 'default-5fold-cv',
    name: '5-Fold Cross-Validation',
    description: '5-fold cross-validation with 80-20 train-test split',
    isDefault: true,
    config: {
      strategy: SplitStrategy.CROSS_VALIDATION,
      splitRatios: { train: 80, validation: 0, test: 20 },
      randomSeed: 42,
      crossValidation: { enabled: true, folds: 5, shuffle: true },
      stratify: true,
    },
  },
  {
    id: 'default-10fold-cv',
    name: '10-Fold Cross-Validation',
    description: '10-fold cross-validation with 80-20 train-test split',
    isDefault: true,
    config: {
      strategy: SplitStrategy.CROSS_VALIDATION,
      splitRatios: { train: 80, validation: 0, test: 20 },
      randomSeed: 42,
      crossValidation: { enabled: true, folds: 10, shuffle: true },
      stratify: true,
    },
  },
];

/**
 * Helper function to validate data split configuration
 */
export function validateDataSplitConfig(
  config: DataSplitConfig
): DataSplitValidation {
  const errors: DataSplitValidation['errors'] = {};
  const warnings: DataSplitValidation['warnings'] = {};

  // Validate split ratios sum to 100
  const { train, validation, test } = config.splitRatios;
  const total = train + validation + test;

  if (Math.abs(total - 100) > 0.01) {
    errors.splitRatios = `Split ratios must sum to 100% (currently ${total.toFixed(1)}%)`;
  }

  // Validate individual ratios
  if (train < 50) {
    warnings.splitRatios = 'Training set is less than 50%, which may lead to poor model performance';
  }
  if (train > 90) {
    warnings.splitRatios = 'Training set is more than 90%, leaving very little data for validation/testing';
  }

  // Validate cross-validation
  if (config.crossValidation.enabled) {
    if (config.crossValidation.folds < 2) {
      errors.crossValidation = 'Cross-validation requires at least 2 folds';
    }
    if (config.crossValidation.folds > 20) {
      warnings.crossValidation = 'Using more than 20 folds may be computationally expensive';
    }
    if (validation > 0) {
      warnings.crossValidation = 'Validation split is ignored when using cross-validation';
    }
  }

  // Validate random seed
  if (config.randomSeed !== null) {
    if (!Number.isInteger(config.randomSeed)) {
      errors.randomSeed = 'Random seed must be an integer';
    }
    if (config.randomSeed < 0) {
      errors.randomSeed = 'Random seed must be a non-negative integer';
    }
  }

  const isValid = Object.keys(errors).length === 0;

  return {
    isValid,
    errors,
    warnings,
  };
}

/**
 * Helper function to create default config
 */
export function getDefaultDataSplitConfig(): DataSplitConfig {
  return {
    strategy: SplitStrategy.HOLDOUT,
    splitRatios: { train: 70, validation: 15, test: 15 },
    randomSeed: 42,
    crossValidation: { enabled: false, folds: 5, shuffle: true },
    stratify: true,
  };
}
