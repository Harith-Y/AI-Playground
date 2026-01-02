# Code Generation Examples

This directory contains examples demonstrating the code generation capabilities of AI-Playground.

## Output Directory

All generated code examples are saved to `backend/generated_code/` directory, which is gitignored to keep the repository clean.

## Directory Structure

```
backend/
├── examples/                          # Example scripts (committed)
│   ├── preprocessing_code_generation_example.py
│   ├── training_code_generation_example.py
│   ├── evaluation_code_generation_example.py
│   ├── prediction_code_generation_example.py
│   └── complete_modular_pipeline_example.py
│
└── generated_code/                    # Generated output (gitignored)
    ├── preprocessing/
    ├── training/
    ├── evaluation/
    ├── prediction/
    └── complete_pipeline/
```

## Examples Overview

### 1. Preprocessing Code Generation Example

**File**: `preprocessing_code_generation_example.py`

Comprehensive example demonstrating all preprocessing code generation capabilities.

#### Examples Included

**1. Basic Preprocessing Pipeline**
- Missing value imputation (mean strategy)
- Standard scaling
- OneHot encoding
- **Output**: Complete script format

**2. Advanced Preprocessing with Outlier Detection**
- IQR-based outlier detection
- Median imputation
- Robust scaling
- Variance threshold feature selection
- **Output**: Function format

**3. Preprocessing Class (sklearn-style)**
- Label encoding
- MinMax scaling
- **Output**: Class format with fit/transform methods

**4. Multiple Imputation Strategies**
- Mean imputation for continuous variables
- Mode imputation for categorical variables
- Median imputation for skewed distributions
- **Output**: Function format

**5. Different Scaling Methods**
- Standard scaling
- MinMax scaling
- Robust scaling
- **Output**: Function format

**6. Categorical Encoding Methods**
- OneHot encoding
- Label encoding
- **Output**: Function format

**7. Complete Data Science Pipeline**
- Outlier detection
- Multiple imputation strategies
- Categorical encoding
- Feature scaling
- Feature selection
- **Output**: Complete script with all steps

**8. Save Generated Code to Files**
Demonstrates saving generated code to:
- `generated_preprocessing_script.py`
- `preprocessing_function.py`
- `preprocessing_class.py`

**9. Generate for Different Data Formats**
Creates preprocessing scripts for:
- CSV files
- Excel files
- JSON files
- Parquet files

**10. Comparison of Output Formats**
Shows differences between:
- Script format (complete standalone script)
- Function format (reusable function)
- Class format (sklearn-style transformer)

**11. Real-World Use Case - Titanic Dataset**
Complete preprocessing pipeline for Titanic survival prediction:
- Age imputation (median)
- Embarked imputation (mode)
- Categorical encoding (Sex, Embarked, Pclass)
- Numerical scaling (Age, Fare, SibSp, Parch)

#### Generated Files

After running the preprocessing example, you'll have:
1. **generated_preprocessing_script.py** - Complete preprocessing pipeline
2. **preprocessing_function.py** - Reusable preprocessing function
3. **preprocessing_class.py** - sklearn-style preprocessing class
4. **preprocess_csv.py** - CSV-specific preprocessing
5. **preprocess_excel.py** - Excel-specific preprocessing
6. **preprocess_json.py** - JSON-specific preprocessing
7. **preprocess_parquet.py** - Parquet-specific preprocessing
8. **titanic_preprocessing.py** - Real-world Titanic example

### 2. Training Code Generation Example

**File**: `training_code_generation_example.py`

Demonstrates model training code generation with various configurations.

### 3. Evaluation Code Generation Example

**File**: `evaluation_code_generation_example.py`

Shows how to generate evaluation and metrics code for different task types.

### 4. Prediction Code Generation Example

**File**: `prediction_code_generation_example.py`

Demonstrates prediction code generation including API services.

### 5. Complete Modular Pipeline Example

**File**: `complete_modular_pipeline_example.py`

Shows how to generate a complete, modular ML pipeline package.

## Supported Preprocessing Steps

The preprocessing code generator supports:

- **Missing Value Imputation**: mean, median, mode
- **Outlier Detection**: IQR, Z-score (clip or remove)
- **Feature Scaling**: Standard, MinMax, Robust
- **Categorical Encoding**: OneHot, Label, Ordinal
- **Feature Selection**: Variance threshold, correlation-based

## Output Formats

All code generators support multiple output formats:

1. **Script**: Complete standalone Python script with imports, data loading, and processing
2. **Function**: Reusable function that can be imported and used in other projects
3. **Class**: sklearn-style transformer class with fit/transform methods (where applicable)
4. **Module**: Importable Python module with clean API
5. **API**: FastAPI microservice (prediction generator only)

## Running Examples

```bash
cd backend

# Generate preprocessing code
python examples/preprocessing_code_generation_example.py

# Generate training code
python examples/training_code_generation_example.py

# Generate evaluation code
python examples/evaluation_code_generation_example.py

# Generate prediction code
python examples/prediction_code_generation_example.py

# Generate complete modular pipeline
python examples/complete_modular_pipeline_example.py
```

## Generated Files

### Preprocessing
- `generated_code/preprocessing/` - Preprocessing scripts, functions, and classes

### Training
- `generated_code/training/` - Training scripts, functions, classes, and modules

### Evaluation
- `generated_code/evaluation/` - Evaluation scripts, functions, and modules

### Prediction
- `generated_code/prediction/` - Prediction scripts, functions, APIs, and modules

### Complete Pipeline
- `ml_pipeline_modules/` - Complete modular ML pipeline package

## Modular Output

All generators support modular output formats:

- **Script**: Complete standalone script
- **Function**: Reusable function only
- **Class**: sklearn-style class (training)
- **Module**: Importable Python module
- **API**: FastAPI service (prediction only)

## Usage in Your Projects

The generated code is production-ready and can be:

1. **Used directly** - Copy generated files to your project
2. **Imported as modules** - Import functions from generated modules
3. **Deployed as services** - Deploy generated FastAPI services
4. **Customized** - Modify generated code for your specific needs

## Example: Using Generated Code

```python
# After running the examples, you can import generated modules:

# Import preprocessing
from generated_code.preprocessing.preprocess import preprocess_data

# Import training
from generated_code.training.train import train_model, save_model

# Import evaluation
from generated_code.evaluation.evaluate import evaluate_model

# Import prediction
from generated_code.prediction.predict import load_model, predict

# Use in your pipeline
df_clean = preprocess_data(df)
model = train_model(X_train, y_train)
results = evaluate_model(model, X_test, y_test)
predictions = predict(model, X_new)
```

## Customization

You can modify the configuration to generate code for your specific needs:

```python
from app.ml_engine.code_generation import generate_preprocessing_code

config = {
    'experiment_name': 'Your Project',
    'preprocessing_steps': [
        # Add your preprocessing steps here
    ],
    'dataset_info': {
        'file_path': 'your_data.csv',
        'file_format': 'csv'
    }
}

code = generate_preprocessing_code(config, output_format='script')
```

### Using Generated Preprocessing Code

```python
# Import the generated preprocessing function
from preprocessing_function import preprocess_data

# Use it on your data
df_clean = preprocess_data(df)
```

Or use the class-based approach:

```python
# Import the generated preprocessing class
from preprocessing_class import DataPreprocessor

# Create and use the preprocessor
preprocessor = DataPreprocessor()
df_train_clean = preprocessor.fit_transform(df_train)
df_test_clean = preprocessor.transform(df_test)
```

## Deploying Generated APIs

```bash
# Navigate to generated prediction API
cd generated_code/prediction

# Run FastAPI service
uvicorn churn_microservice:app --reload

# Visit API documentation
# http://localhost:8000/docs
```

## Benefits

- ✅ **Production-ready code** - All generated code follows best practices
- ✅ **Modular design** - Each component can be used independently
- ✅ **Type hints** - Full type annotations for better IDE support
- ✅ **Documentation** - Comprehensive docstrings and comments
- ✅ **Error handling** - Proper validation and error messages
- ✅ **Customizable** - Easy to modify for specific needs
- ✅ **Time-saving** - Generate production-ready code in seconds
- ✅ **Consistency** - All generated code follows the same patterns
- ✅ **Reproducibility** - Includes random seeds and configuration
- ✅ **Quality** - Valid Python code with proper structure

## Code Quality

All generated code includes:
- Proper imports
- Type hints
- Docstrings
- Error handling
- Reproducibility (random seeds)
- Comments explaining each step
- Production-ready structure
- Best practices compliance

## Notes

- Generated files are **not committed** to the repository (gitignored)
- Run examples to regenerate code whenever needed
- Modify generated code as needed for your use case
- Generated code is self-contained and has minimal dependencies
