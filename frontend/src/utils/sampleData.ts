/**
 * Sample data generators for testing and demonstration purposes
 */

export const generateSampleCorrelationMatrix = (featureCount: number = 6) => {
  const features = Array.from(
    { length: featureCount },
    (_, i) => `Feature_${i + 1}`
  );

  const matrix: number[][] = [];

  for (let i = 0; i < featureCount; i++) {
    const row: number[] = [];
    for (let j = 0; j < featureCount; j++) {
      if (i === j) {
        // Diagonal is always 1 (perfect correlation with itself)
        row.push(1.0);
      } else if (j < i) {
        // Mirror the upper triangle to lower triangle (symmetric matrix)
        row.push(matrix[j][i]);
      } else {
        // Generate random correlation between -1 and 1
        const correlation = (Math.random() * 2 - 1).toFixed(3);
        row.push(parseFloat(correlation));
      }
    }
    matrix.push(row);
  }

  return {
    features,
    matrix,
  };
};

export const generateSampleFeatureImportance = (featureCount: number = 10) => {
  const features = Array.from(
    { length: featureCount },
    (_, i) => `Feature_${i + 1}`
  );

  return features
    .map((feature) => ({
      feature,
      importance: Math.random(),
    }))
    .sort((a, b) => b.importance - a.importance);
};

export const generateRealisticCorrelationMatrix = () => {
  const features = [
    'Age',
    'Income',
    'Education_Years',
    'Experience',
    'Hours_Worked',
    'Satisfaction',
  ];

  // Realistic correlation matrix with meaningful relationships
  const matrix = [
    [1.0, 0.45, 0.32, 0.78, 0.23, 0.15], // Age
    [0.45, 1.0, 0.67, 0.52, 0.34, 0.41], // Income
    [0.32, 0.67, 1.0, 0.44, 0.28, 0.38], // Education_Years
    [0.78, 0.52, 0.44, 1.0, 0.56, 0.22], // Experience
    [0.23, 0.34, 0.28, 0.56, 1.0, -0.12], // Hours_Worked
    [0.15, 0.41, 0.38, 0.22, -0.12, 1.0], // Satisfaction
  ];

  return {
    features,
    matrix,
  };
};
