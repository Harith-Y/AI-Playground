/**
 * Code Generators
 * 
 * Utilities for generating code in different formats
 */

interface GenerateCodeOptions {
  datasetName: string;
  modelName: string;
  modelType: string;
  includePreprocessing: boolean;
  includeEvaluation: boolean;
  includeVisualization: boolean;
}

/**
 * Generate Python script
 */
export const generatePythonScript = (options: GenerateCodeOptions): string => {
  const {
    datasetName,
    modelName,
    modelType,
    includePreprocessing,
    includeEvaluation,
    includeVisualization,
  } = options;

  return `"""
${modelName} - Production Script
Generated from ML Pipeline

Dataset: ${datasetName}
Model Type: ${modelType}
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
${includePreprocessing ? 'from sklearn.preprocessing import StandardScaler, LabelEncoder' : ''}
${includeEvaluation ? `from sklearn.metrics import ${modelType === 'classification' ? 'accuracy_score, classification_report, confusion_matrix' : 'mean_squared_error, r2_score, mean_absolute_error'}` : ''}
${includeVisualization ? 'import matplotlib.pyplot as plt\nimport seaborn as sns' : ''}

class MLPipeline:
    """Complete ML Pipeline for ${modelName}"""
    
    def __init__(self):
        self.model = None
        ${includePreprocessing ? 'self.scaler = StandardScaler()' : ''}
        ${includePreprocessing ? 'self.label_encoder = LabelEncoder()' : ''}
    
    def load_data(self, filepath):
        """Load dataset from file"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    ${includePreprocessing ? `def preprocess_data(self, X, y=None, fit=True):
        """Apply preprocessing transformations"""
        print("Preprocessing data...")
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Encode labels if provided
        if y is not None:
            if fit:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
        
        return X_scaled
    ` : ''}
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("Training model...")
        # TODO: Load your trained model or implement training logic
        # self.model = YourModel()
        # self.model.fit(X_train, y_train)
        print("Training complete!")
    
    def predict(self, X):
        """Make predictions"""
        ${includePreprocessing ? 'X_processed = self.preprocess_data(X, fit=False)' : 'X_processed = X'}
        predictions = self.model.predict(X_processed)
        return predictions
    
    ${includeEvaluation ? `def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("Evaluating model...")
        predictions = self.predict(X_test)
        
        ${modelType === 'classification' ? `
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        print("\\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        return {
            'accuracy': accuracy,
            'predictions': predictions
        }
        ` : `
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions
        }
        `}
    ` : ''}
    
    ${includeVisualization ? `def visualize_results(self, y_true, y_pred):
        """Visualize model results"""
        plt.figure(figsize=(12, 5))
        
        ${modelType === 'classification' ? `
        # Confusion Matrix
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        ` : `
        # Actual vs Predicted
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted')
        
        # Residuals
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        `}
        
        plt.tight_layout()
        plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'model_results.png'")
        plt.show()
    ` : ''}
    
    def save_model(self, filepath='model.pkl'):
        """Save trained model"""
        joblib.dump(self.model, filepath)
        ${includePreprocessing ? "joblib.dump(self.scaler, 'scaler.pkl')" : ''}
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='model.pkl'):
        """Load trained model"""
        self.model = joblib.load(filepath)
        ${includePreprocessing ? "self.scaler = joblib.load('scaler.pkl')" : ''}
        print(f"Model loaded from {filepath}")


def main():
    """Main execution pipeline"""
    # Initialize pipeline
    pipeline = MLPipeline()
    
    # Load data
    data = pipeline.load_data('${datasetName.toLowerCase().replace(/\s+/g, '_')}.csv')
    
    # Prepare features and target
    X = data.drop('target', axis=1)  # Replace 'target' with your target column
    y = data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    ${includePreprocessing ? `
    # Preprocess data
    X_train_processed, y_train_processed = pipeline.preprocess_data(X_train, y_train, fit=True)
    X_test_processed, y_test_processed = pipeline.preprocess_data(X_test, y_test, fit=False)
    ` : `
    X_train_processed, y_train_processed = X_train, y_train
    X_test_processed, y_test_processed = X_test, y_test
    `}
    
    # Train model
    pipeline.train(X_train_processed, y_train_processed)
    
    ${includeEvaluation ? `
    # Evaluate model
    results = pipeline.evaluate(X_test_processed, y_test_processed)
    ` : ''}
    
    ${includeVisualization ? `
    # Visualize results
    predictions = pipeline.predict(X_test_processed)
    pipeline.visualize_results(y_test_processed, predictions)
    ` : ''}
    
    # Save model
    pipeline.save_model()
    
    print("\\nPipeline execution complete!")


if __name__ == "__main__":
    main()
`;
};

/**
 * Generate Jupyter Notebook (as JSON)
 */
export const generateJupyterNotebook = (options: GenerateCodeOptions): string => {
  const pythonCode = generatePythonScript(options);
  
  // Split code into cells
  const cells = [
    {
      cell_type: 'markdown',
      metadata: {},
      source: [
        `# ${options.modelName}\n`,
        '\n',
        `**Dataset:** ${options.datasetName}\n`,
        `**Model Type:** ${options.modelType}\n`,
        '\n',
        'This notebook contains the complete ML pipeline generated from your experiment.\n',
      ],
    },
    {
      cell_type: 'code',
      execution_count: null,
      metadata: {},
      outputs: [],
      source: pythonCode.split('\n').slice(0, 20),
    },
  ];

  const notebook = {
    cells,
    metadata: {
      kernelspec: {
        display_name: 'Python 3',
        language: 'python',
        name: 'python3',
      },
      language_info: {
        name: 'python',
        version: '3.8.0',
      },
    },
    nbformat: 4,
    nbformat_minor: 4,
  };

  return JSON.stringify(notebook, null, 2);
};

/**
 * Generate FastAPI service
 */
export const generateFastAPIService = (options: GenerateCodeOptions): string => {
  const { datasetName, modelName, modelType } = options;

  return `"""
${modelName} - FastAPI Service
Generated ML API Service

Dataset: ${datasetName}
Model Type: ${modelType}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="${modelName} API",
    description="ML Model Prediction Service",
    version="1.0.0"
)

# Load model at startup
model = None
scaler = None

@app.on_event("startup")
async def load_model():
    """Load model and preprocessors on startup"""
    global model, scaler
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

# Request/Response models
class PredictionRequest(BaseModel):
    """Input data for prediction"""
    features: List[float] = Field(..., description="Input features")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [1.0, 2.0, 3.0, 4.0, 5.0]
            }
        }

class PredictionResponse(BaseModel):
    """Prediction output"""
    prediction: ${modelType === 'classification' ? 'int' : 'float'}
    probability: List[float] = None
    timestamp: str
    model_version: str = "1.0.0"

class BatchPredictionRequest(BaseModel):
    """Batch prediction input"""
    instances: List[List[float]]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "ML Model API",
        "model": "${modelName}",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input
        features = np.array(request.features).reshape(1, -1)
        
        # Scale features
        if scaler is not None:
            features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probabilities for classification
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0].tolist()
        
        return PredictionResponse(
            prediction=${modelType === 'classification' ? 'int(prediction)' : 'float(prediction)'},
            probability=probability,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input
        features = np.array(request.instances)
        
        # Scale features
        if scaler is not None:
            features = scaler.transform(features)
        
        # Make predictions
        predictions = model.predict(features)
        
        return {
            "predictions": predictions.tolist(),
            "count": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "${modelName}",
        "model_type": "${modelType}",
        "dataset": "${datasetName}",
        "version": "1.0.0",
        "features": "Auto-detected from input"
    }

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
`;
};

/**
 * Generate requirements.txt
 */
export const generateRequirements = (format: string): string => {
  const baseRequirements = [
    'pandas>=1.3.0',
    'numpy>=1.21.0',
    'scikit-learn>=1.0.0',
    'joblib>=1.1.0',
  ];

  if (format === 'fastapi') {
    baseRequirements.push(
      'fastapi>=0.95.0',
      'uvicorn>=0.21.0',
      'pydantic>=1.10.0'
    );
  }

  if (format === 'notebook') {
    baseRequirements.push(
      'jupyter>=1.0.0',
      'matplotlib>=3.5.0',
      'seaborn>=0.12.0'
    );
  }

  return baseRequirements.join('\n');
};

/**
 * Generate README.md
 */
export const generateReadme = (options: GenerateCodeOptions, format: string): string => {
  return `# ${options.modelName}

Machine Learning project generated from ML Pipeline.

## Dataset
- **Name:** ${options.datasetName}
- **Type:** ${options.modelType}

## Model
- **Name:** ${options.modelName}
- **Format:** ${format}

## Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

${format === 'python' ? `
### Running the Script
\`\`\`bash
python model.py
\`\`\`
` : ''}

${format === 'notebook' ? `
### Running the Notebook
\`\`\`bash
jupyter notebook model.ipynb
\`\`\`
` : ''}

${format === 'fastapi' ? `
### Running the API
\`\`\`bash
uvicorn main:app --reload
\`\`\`

### API Documentation
Visit http://localhost:8000/docs for interactive API documentation.

### Making Predictions
\`\`\`bash
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0]}'
\`\`\`
` : ''}

## Project Structure
\`\`\`
.
├── model.${format === 'notebook' ? 'ipynb' : 'py'}
├── requirements.txt
└── README.md
\`\`\`

## License
MIT
`;
};

export default {
  generatePythonScript,
  generateJupyterNotebook,
  generateFastAPIService,
  generateRequirements,
  generateReadme,
};
