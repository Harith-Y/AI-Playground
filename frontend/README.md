# ML Pipeline Frontend

Modern React + TypeScript frontend for the ML Pipeline application.

## 🚀 Features

- **Dataset Management**: Upload, preview, and analyze datasets
- **Preprocessing**: Visual pipeline builder with step-by-step transformations
- **Model Training**: Configure and train ML models with real-time progress
- **Evaluation**: Comprehensive metrics and visualizations
- **Hyperparameter Tuning**: Automated optimization with multiple algorithms
- **Code Generation**: Export production-ready Python, Jupyter, or FastAPI code

## 🛠️ Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Material-UI (MUI)** - Component library
- **Redux Toolkit** - State management
- **React Router** - Navigation
- **Recharts** - Data visualization
- **Axios** - HTTP client

## 📦 Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 🏗️ Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable components
│   │   ├── common/         # Shared UI components
│   │   ├── dataset/        # Dataset-related components
│   │   ├── preprocessing/  # Preprocessing components
│   │   ├── modeling/       # Model training components
│   │   ├── evaluation/     # Evaluation components
│   │   ├── tuning/         # Hyperparameter tuning
│   │   └── codegen/        # Code generation
│   ├── pages/              # Page components
│   ├── services/           # API services
│   ├── hooks/              # Custom React hooks
│   ├── store/              # Redux store
│   ├── utils/              # Utility functions
│   ├── types/              # TypeScript types
│   └── App.tsx             # Root component
├── public/                 # Static assets
├── tests/                  # Test files
└── package.json
```

## 🎨 Key Components

### Dataset Management
- `DatasetUpload` - Drag & drop file upload
- `DatasetPreview` - Table view with pagination
- `DatasetStats` - Statistical summaries
- `VisualizationGallery` - Charts and plots

### Preprocessing
- `PreprocessingPage` - Main preprocessing interface
- `StepBuilder` - Add preprocessing steps
- `StepConfiguration` - Configure step parameters
- `PreviewPanel` - Before/after visualization

### Model Training
- `ModelingPage` - Model configuration
- `ModelSelector` - Choose ML algorithms
- `HyperparameterForm` - Parameter configuration
- `TrainingProgress` - Real-time training updates

### Evaluation
- `EvaluationPage` - Results dashboard
- `MetricsDisplay` - Performance metrics
- `ConfusionMatrix` - Classification results
- `FeatureImportance` - Feature analysis

### Hyperparameter Tuning
- `TuningPage` - Tuning interface
- `TuningConfiguration` - Search space setup
- `TuningProgress` - Real-time progress
- `TuningResults` - Best parameters

### Code Generation
- `CodeGenerationPage` - Code export interface
- `CodePreview` - Syntax-highlighted preview
- `DownloadOptions` - Multiple export formats

## 🔌 API Integration

All API calls are centralized in the `services/` directory:

```typescript
// Example API usage
import { uploadDataset } from './services/datasetService';

const handleUpload = async (file: File) => {
  const response = await uploadDataset(file);
  console.log('Dataset ID:', response.dataset_id);
};
```

## 🎯 State Management

Redux Toolkit is used for global state:

```typescript
// Example Redux usage
import { useAppDispatch, useAppSelector } from './store/hooks';
import { setDataset } from './store/datasetSlice';

const MyComponent = () => {
  const dispatch = useAppDispatch();
  const dataset = useAppSelector(state => state.dataset.current);
  
  const handleSelect = (dataset) => {
    dispatch(setDataset(dataset));
  };
};
```

## 🧪 Testing

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Generate coverage report
npm run test:coverage
```

## ♿ Accessibility

The application follows WCAG 2.1 Level AA guidelines:

- Semantic HTML elements
- ARIA labels and roles
- Keyboard navigation support
- Screen reader compatibility
- Color contrast compliance
- Focus management

## ⚡ Performance

Optimizations implemented:

- Code splitting with React.lazy()
- Memoization with React.memo()
- Virtual scrolling for large lists
- Image lazy loading
- Bundle size optimization
- Tree shaking

## 🌐 Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## 📝 Environment Variables

Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
```

## 🚢 Production Build

```bash
# Build for production
npm run build

# Output will be in dist/ directory
# Serve with any static file server
```

### Docker Deployment

```bash
# Build Docker image
docker build -t ml-pipeline-frontend .

# Run container
docker run -p 3000:80 ml-pipeline-frontend
```

## 🔧 Development

### Code Style

- ESLint for linting
- Prettier for formatting
- TypeScript strict mode

```bash
# Lint code
npm run lint

# Format code
npm run format
```

### Git Workflow

1. Create feature branch
2. Make changes
3. Run tests
4. Submit pull request

## 📚 Documentation

- [Component Documentation](./docs/components.md)
- [API Documentation](./docs/api.md)
- [State Management](./docs/state.md)
- [Testing Guide](./docs/testing.md)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/your-org/ml-pipeline/issues)
- Documentation: [Read the docs](https://docs.ml-pipeline.com)

## 🎉 Acknowledgments

Built with ❤️ using modern web technologies.
