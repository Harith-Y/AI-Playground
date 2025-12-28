import { Routes, Route } from 'react-router-dom'
import Layout from './components/common/Layout'
import ErrorBoundary from './components/common/ErrorBoundary'
import HomePage from './pages/HomePage'
import DatasetUploadPage from './pages/DatasetUploadPage'
import PreprocessingPage from './pages/PreprocessingPage'
import FeatureEngineeringPage from './pages/FeatureEngineeringPage'
import ModelingPage from './pages/ModelingPage'
import ExplorationPage from './pages/ExplorationPage'
import EvaluationPage from './pages/EvaluationPage'
import TuningPage from './pages/TuningPage'
import CodeGenerationPage from './pages/CodeGenerationPage'
import './App.css'

function App() {
  return (
    <ErrorBoundary>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/dataset-upload" element={<DatasetUploadPage />} />
          <Route path="/exploration" element={<ExplorationPage />} />
          <Route path="/preprocessing" element={<PreprocessingPage />} />
          <Route path="/features" element={<FeatureEngineeringPage />} />
          <Route path="/modeling" element={<ModelingPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
          <Route path="/tuning" element={<TuningPage />} />
          <Route path="/code-generation" element={<CodeGenerationPage />} />
        </Routes>
      </Layout>
    </ErrorBoundary>
  )
}

export default App
