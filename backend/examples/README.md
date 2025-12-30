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
- ✅ **Documentation** - Comprehensive docstrings
- ✅ **Error handling** - Proper validation and error messages
- ✅ **Customizable** - Easy to modify for specific needs

## Notes

- Generated files are **not committed** to the repository (gitignored)
- Run examples to regenerate code whenever needed
- Modify generated code as needed for your use case
- Generated code is self-contained and has minimal dependencies
