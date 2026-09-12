export interface Job {
  id: string
  project: string
  episode: string
  steps: string[]
  status: 'pending' | 'running' | 'paused' | 'stopped' | 'completed' | 'failed' | 'cancelled' | 'awaiting_export'
  current_step: string | null
  progress: number | null
  steps_completed: string[]
  error: string | null
  created_at: string
  updated_at: string
  started_at: string | null
  finished_at: string | null
  elapsed_seconds: number | null
  step_durations: Record<string, number>
}

export interface EpisodePipelineState {
  job_id: string | null
  project: string
  episode: string
  steps: string[]
  status: string
  current_step: string | null
  progress: number | null
  steps_completed: string[]
  error: string | null
  created_at: string | null
  updated_at: string | null
  started_at: string | null
  finished_at: string | null
  elapsed_seconds: number | null
  step_durations: Record<string, number>
  can_pause: boolean
  can_resume: boolean
  can_stop: boolean
  can_cancel: boolean
}

export interface Short {
  index: number
  start: number
  end: number
  duration: number
  score: number | null
  trend_score: number | null
  keyword_score: number | null
  hook_score: number | null
  duration_score: number | null
  base_score: number | null
  topic: string | null
  hook: string | null
  reason: string | null
  dominant_speaker: string | null
  rank: number | null
  file: string | null
  has_video: boolean
  is_draft?: boolean
  categories: string[]
}

export interface TranscriptInfo {
  srt_file: string | null
  transcript_file: string | null
  language: string | null
  segments_count: number | null
  duration_seconds: number | null
}

export interface VideoInfo {
  filename: string
  source: string | null
  title: string | null
  duration: number | null
  url: string | null
}

export interface EpisodeItem {
  name: string
  shorts_count: number
  has_transcript: boolean
  has_public_video: boolean
  status: string
  steps: string[]
  steps_completed: string[]
  last_elapsed_seconds: number | null
  current_step: string | null
  progress: number | null
  job_id: string | null
  updated_at: string | null
  started_at: string | null
  step_durations: Record<string, number>
  can_pause: boolean
  can_resume: boolean
  can_stop: boolean
  can_cancel: boolean
}

export interface ProjectListItem {
  project: string
  episodes: EpisodeItem[]
}

export interface ProgressEvent {
  job_id: string
  step: string
  message: string
  percent: number | null
}

export interface JobCreate {
  project: string
  episode: string
  url?: string
  local_file?: string
  steps?: string[]
  ai_model?: string
  diarization_provider?: string
  min_duration?: number
  max_duration?: number
}

export interface ModelStatus {
  model: string
  provider: string
  available: boolean
  error: string | null
  response_time_ms: number | null
}

export interface ModelsStatusResponse {
  models: ModelStatus[]
  gemini_keys_count: number
  default_model: string
}

export interface CustomModelConfig {
  name: string
  base_url: string
  api_key: string
  model_id: string
  provider_type: string
}

export interface HardwareInfo {
  cpu: { name: string; cores: number }
  gpu: { name: string; vram_mb: number; driver: string; compute_cap: string } | null
  capabilities: { nvenc: boolean; cuda_ai: boolean }
  recommendations: Array<{ component: string; recommendation: string; reason: string }>
}

export interface ProcessingConfig {
  device: string
  ffmpeg_encoder: string
  ffmpeg_preset: string
}
