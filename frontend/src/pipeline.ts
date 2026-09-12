export type PipelineStepKey =
  | 'download'
  | 'transcribe'
  | 'diarize'
  | 'face_positions'
  | 'speaker_bind'
  | 'analyze'
  | 'edit'
  | 'rank'
  | 'export'

type PipelineSnapshot = {
  status?: string | null
  steps?: string[] | null
  currentStep?: string | null
  progress?: number | null
  stepsCompleted?: string[] | null
  elapsedSeconds?: number | null
  startedAt?: string | null
  stepDurations?: Record<string, number> | null
}

// Aliases legacy → paso v3.5 (jobs viejos en BD y SSE con nombres antiguos).
const STEP_ALIASES: Record<string, PipelineStepKey> = {
  trend: 'analyze',
  fcpxml: 'export',
  doc_export: 'export',
  // v2 legacy
  detect: 'face_positions',
  calibrate: 'speaker_bind',
  guion: 'analyze',
  validate: 'analyze',
  ranking: 'rank',
  // v3.0-v3.4 legacy (se eliminó ASD)
  face_track: 'face_positions',
  asd: 'speaker_bind',
  cmia_bind: 'speaker_bind',
}

export const MAIN_PIPELINE_STEPS: PipelineStepKey[] = [
  'download',
  'transcribe',
  'diarize',
  'face_positions',
  'speaker_bind',
  'analyze',
  'edit',
  'rank',
  'export',
]

export const PIPELINE_STEP_META: Record<PipelineStepKey, { label: string; est: string; estSecs: number; icon: string }> = {
  download:       { label: 'Descarga',          est: '3-8 min',   estSecs: 330,  icon: '⬇' },
  transcribe:     { label: 'Transcripción',     est: '10-20 min', estSecs: 900,  icon: '📝' },
  diarize:        { label: 'Diarización',       est: '1-3 min',   estSecs: 120,  icon: '🎙' },
  face_positions: { label: 'Posiciones faciales', est: '< 1 min', estSecs: 40,   icon: '👤' },
  speaker_bind:   { label: 'Voz ↔ rostro',      est: '< 1 min',   estSecs: 60,   icon: '🔗' },
  analyze:        { label: 'Análisis IA',       est: '2-5 min',   estSecs: 210,  icon: '🧠' },
  edit:           { label: 'Edición',           est: '5-15 min',  estSecs: 600,  icon: '🎬' },
  rank:           { label: 'Ranking',           est: '< 1 min',   estSecs: 30,   icon: '🏆' },
  export:         { label: 'Exportación',       est: '2-5 min',   estSecs: 210,  icon: '📦' },
}

export const STEP_DESCRIPTIONS: Record<PipelineStepKey, string> = {
  download:       'Descarga el video desde YouTube o copia desde archivo local. Extrae audio WAV.',
  transcribe:     'Transcribe el audio con Whisper (faster-whisper). Genera timestamps por palabra.',
  diarize:        'Diarización acústica: identifica SPEAKER_00..N y cuándo habla cada uno.',
  face_positions: 'Detecta las N posiciones fijas de los participantes (plano general estático).',
  speaker_bind:   'Asocia cada voz con su rostro usando los primeros minutos como calibración.',
  analyze:        'ÚNICO paso con IA: elige los mejores momentos del transcript anotado por speaker.',
  edit:           'Borradores low-res con cambios de cámara según quién habla (~10s por short).',
  rank:           'Ranking heurístico + guion DOCX estructurado. Pausa para revisión antes de exportar.',
  export:         'Render final HD + distribución a OCAMO + docs + FCPXML. Manual desde UI.',
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function normalizeStepList(steps?: string[] | null, fallback: PipelineStepKey[] = MAIN_PIPELINE_STEPS): PipelineStepKey[] {
  if (!steps || steps.length === 0) return [...fallback]

  const result: PipelineStepKey[] = []
  for (const step of steps) {
    const normalized = normalizePipelineStep(step)
    if (normalized && !result.includes(normalized)) {
      result.push(normalized)
    }
  }

  return result.length > 0 ? result : [...fallback]
}

export function normalizePipelineStep(step?: string | null): PipelineStepKey | null {
  if (!step) return null
  const normalized = STEP_ALIASES[step] ?? step
  return MAIN_PIPELINE_STEPS.includes(normalized as PipelineStepKey)
    ? (normalized as PipelineStepKey)
    : null
}

export function getPlannedPipelineSteps(steps?: string[] | null): PipelineStepKey[] {
  return normalizeStepList(steps, MAIN_PIPELINE_STEPS)
}

export function getCompletedPipelineSteps(steps?: string[] | null): PipelineStepKey[] {
  return normalizeStepList(steps, [])
}

export function getPipelineStepState(step: PipelineStepKey, snapshot: PipelineSnapshot): 'done' | 'active' | 'pending' {
  const completed = new Set(getCompletedPipelineSteps(snapshot.stepsCompleted))
  if (completed.has(step)) return 'done'

  const activeStep = normalizePipelineStep(snapshot.currentStep)
  if (activeStep === step) return 'active'

  // Steps before the first planned step are implicitly done
  // (e.g. download is done when the pipeline starts from transcribe)
  const plannedSteps = getPlannedPipelineSteps(snapshot.steps)
  const firstPlannedIndex = plannedSteps.length > 0 ? MAIN_PIPELINE_STEPS.indexOf(plannedSteps[0]) : 0
  const currentIndex = MAIN_PIPELINE_STEPS.indexOf(step)

  if (!plannedSteps.includes(step) && currentIndex !== -1 && currentIndex < firstPlannedIndex) {
    return 'done'
  }

  // When job is running/pending but no currentStep yet, treat first planned step as active
  if (!activeStep && ['running', 'pending'].includes(snapshot.status ?? '') && plannedSteps.length > 0 && step === plannedSteps[0]) {
    return 'active'
  }

  return 'pending'
}

export function getPipelineStepPercent(step: PipelineStepKey, snapshot: PipelineSnapshot): number {
  const state = getPipelineStepState(step, snapshot)
  if (state === 'done') return 100
  if (state === 'active') return clamp(Math.round(snapshot.progress ?? 0), 0, 100)
  return 0
}

export function getOverallPipelineProgress(snapshot: PipelineSnapshot): number {
  if (snapshot.status === 'completed') return 100

  const plannedSteps = getPlannedPipelineSteps(snapshot.steps)
  if (plannedSteps.length === 0) return 0

  const completed = new Set(getCompletedPipelineSteps(snapshot.stepsCompleted))
  let activeStep = normalizePipelineStep(snapshot.currentStep)
  if (!activeStep && ['running', 'pending'].includes(snapshot.status ?? '') && plannedSteps.length > 0) {
    activeStep = plannedSteps.find(s => !completed.has(s)) ?? null
  }

  // Excluir export del cálculo si no está activo ni completado
  // (awaiting_export = rank terminó pero export no ha sido lanzado manualmente)
  const exportActive = activeStep === 'export'
  const exportDone = completed.has('export')
  const effectiveSteps = (!exportActive && !exportDone)
    ? plannedSteps.filter(s => s !== 'export')
    : plannedSteps

  if (effectiveSteps.length === 0) return 0

  let completedUnits = 0
  for (const step of effectiveSteps) {
    if (completed.has(step)) {
      completedUnits += 1
      continue
    }
    if (activeStep === step) {
      completedUnits += clamp((snapshot.progress ?? 0) / 100, 0, 1)
    }
  }

  return clamp(Math.round((completedUnits / effectiveSteps.length) * 100), 0, 100)
}

export function getLiveElapsedSeconds(snapshot: PipelineSnapshot, nowMs = Date.now()): number {
  const persistedElapsed = Math.max(0, Math.round(snapshot.elapsedSeconds ?? 0))
  if (!['running', 'pending'].includes(snapshot.status ?? '')) return persistedElapsed

  const parsedStartedAt = snapshot.startedAt ? Date.parse(snapshot.startedAt) : NaN
  if (Number.isNaN(parsedStartedAt)) return persistedElapsed

  return Math.max(persistedElapsed, Math.floor((nowMs - parsedStartedAt) / 1000))
}

export function getPipelineTiming(snapshot: PipelineSnapshot, nowMs = Date.now()) {
  const plannedSteps = getPlannedPipelineSteps(snapshot.steps)
  const completed = new Set(getCompletedPipelineSteps(snapshot.stepsCompleted))
  let activeStep = normalizePipelineStep(snapshot.currentStep)
  // When job is running/pending but no currentStep yet, infer from first planned step
  if (!activeStep && ['running', 'pending'].includes(snapshot.status ?? '') && plannedSteps.length > 0) {
    activeStep = plannedSteps.find(s => !completed.has(s)) ?? null
  }
  const elapsedSecs = getLiveElapsedSeconds(snapshot, nowMs)
  const overallProgress = getOverallPipelineProgress(snapshot)

  // Excluir export de timing si no está activo ni completado
  const exportActive = activeStep === 'export'
  const exportDone = completed.has('export')
  const timingSteps = (!exportActive && !exportDone)
    ? plannedSteps.filter(s => s !== 'export')
    : plannedSteps

  const completedActualSecs = timingSteps.reduce((sum, step) => {
    if (!completed.has(step)) return sum
    return sum + (snapshot.stepDurations?.[step] ?? PIPELINE_STEP_META[step].estSecs)
  }, 0)

  const activeStepProgress = activeStep ? clamp(Math.round(snapshot.progress ?? 0), 0, 100) : 0
  const activeProgressRatio = activeStep ? clamp(activeStepProgress / 100, 0, 1) : 0
  const activeElapsedSecs = activeStep ? Math.max(0, elapsedSecs - completedActualSecs) : 0
  const activeDefaultSecs = activeStep ? PIPELINE_STEP_META[activeStep].estSecs : 0
  // Solo extrapolar cuando tenemos suficiente señal de progreso real (>20%).
  // Por debajo, faster-whisper y otros motores solo reportan un 10-15% inicial
  // mientras cargan modelos — extrapolar ahí produce estimados absurdos.
  // Además cap en 3× el estimado base para evitar runaway.
  const activePredictedTotalSecs = activeStep
    ? activeProgressRatio >= 0.20
      ? clamp(
          Math.round(activeElapsedSecs / activeProgressRatio),
          activeDefaultSecs,
          activeDefaultSecs * 3,
        )
      : Math.max(activeDefaultSecs, activeElapsedSecs)
    : 0

  const pendingSecs = timingSteps.reduce((sum, step) => {
    if (completed.has(step) || step === activeStep) return sum
    return sum + PIPELINE_STEP_META[step].estSecs
  }, 0)

  const baselineTotalSecs = timingSteps.reduce((sum, step) => sum + PIPELINE_STEP_META[step].estSecs, 0)
  const estimatedTotalSecs = Math.max(
    elapsedSecs,
    Math.round(completedActualSecs + activePredictedTotalSecs + pendingSecs),
    baselineTotalSecs,
  )

  const remainingSecs = ['running', 'pending', 'paused', 'stopped'].includes(snapshot.status ?? '')
    ? Math.max(0, estimatedTotalSecs - elapsedSecs)
    : 0

  return {
    activeStep,
    activeStepProgress,
    elapsedSecs,
    estimatedTotalSecs,
    overallProgress,
    remainingSecs,
  }
}

export function formatPipelineDuration(secs: number): string {
  if (secs <= 0) return '< 1 seg'
  if (secs < 60) return `${secs} seg`
  const minutes = Math.floor(secs / 60)
  const seconds = secs % 60
  return seconds > 0 ? `${minutes}m ${seconds}s` : `${minutes} min`
}