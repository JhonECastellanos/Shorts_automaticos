import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import PipelineDemo from './PipelineDemo'
import './index.css'

// Modo demo: navegar a ?demo=pipeline para probar UI sin backend.
const demoMode = new URLSearchParams(window.location.search).get('demo')

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {demoMode === 'pipeline' ? <PipelineDemo /> : <App />}
  </StrictMode>,
)
