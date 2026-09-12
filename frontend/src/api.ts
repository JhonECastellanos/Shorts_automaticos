import type { Job, JobCreate, Short, TranscriptInfo, VideoInfo, ProjectListItem, EpisodePipelineState, ModelsStatusResponse, CustomModelConfig, HardwareInfo, ProcessingConfig } from './types'

const BASE = '/api'

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status}: ${body}`)
  }
  return res.json()
}

export const api = {
  // Jobs
  createJob: (data: JobCreate) =>
    request<Job>('/jobs', { method: 'POST', body: JSON.stringify(data) }),
  listJobs: () => request<Job[]>('/jobs'),
  getJob: (id: string) => request<Job>(`/jobs/${id}`),
  pauseJob: (id: string) =>
    request<Job>(`/jobs/${id}/pause`, { method: 'POST' }),
  resumeJob: (id: string) =>
    request<Job>(`/jobs/${id}/resume`, { method: 'POST' }),
  stopJob: (id: string) =>
    request<Job>(`/jobs/${id}/stop`, { method: 'POST' }),
  cancelJob: (id: string) =>
    request<{ status: string }>(`/jobs/${id}/cancel`, { method: 'POST' }),
  updateJobConfig: (id: string, config: { ai_model?: string }) =>
    request<Job>(`/jobs/${id}/config`, { method: 'PATCH', body: JSON.stringify(config) }),

  // Videos
  listProjects: () => request<ProjectListItem[]>('/videos/projects'),
  suggestProjectName: () =>
    request<{ suggested_name: string | null; at_limit: boolean; current_count: number; max_projects: number }>(
      '/videos/projects/suggest-name',
    ),
  getProjectLimits: () =>
    request<{ max_projects: number; current_count: number; remaining_slots: number; at_limit: boolean }>(
      '/videos/projects/limits',
    ),
  getPipelineStatus: (project: string, episode: string) =>
    request<EpisodePipelineState>(`/videos/${project}/${episode}/pipeline-status`),
  deleteProject: (project: string, episode: string) =>
    request<{ status: string }>(`/videos/${project}/${episode}`, { method: 'DELETE' }),
  getVideoInfo: (project: string, episode: string) =>
    request<VideoInfo>(`/videos/${project}/${episode}`),

  // Calibration (speaker ↔ face position)
  getBindStatus: (project: string, episode: string) =>
    request<{
      positions: Array<{ position_id: number; cx: number; cy: number; w: number; h: number; confidence: number; thumb_url: string | null }>
      speakers: Array<{ speaker: string; sample_start: number; sample_end: number; current_position_id: number | null; confidence: number | null; source: string | null; needs_manual: boolean }>
      needs_calibration: boolean
    }>(`/calibration/${project}/${episode}/bind-status`),
  overrideBinding: (project: string, episode: string, mapping: Record<string, number>) =>
    request<{ status: string; binding: unknown }>(`/calibration/${project}/${episode}/bind-override`, {
      method: 'POST',
      body: JSON.stringify({ mapping }),
    }),

  // Shorts
  listShorts: (project: string, episode: string) =>
    request<Short[]>(`/shorts/${project}/${episode}`),
  updateShortTime: (project: string, episode: string, index: number, start: number, end: number) =>
    request<{ status: string }>(`/shorts/${project}/${episode}/${index}`, {
      method: 'PATCH',
      body: JSON.stringify({ start, end })
    }),
  deleteShort: (project: string, episode: string, index: number) =>
    request<{ status: string; remaining: number }>(`/shorts/${project}/${episode}/${index}`, {
      method: 'DELETE',
    }),
  exportShort: (project: string, episode: string, index: number) =>
    request<{ status: string; results: unknown[] }>(`/shorts/${project}/${episode}/${index}/export`, {
      method: 'POST',
    }),
  previewShort: (project: string, episode: string, index: number) =>
    request<{ status: string; result: { file: string; short_num: number } }>(`/shorts/${project}/${episode}/${index}/preview`, {
      method: 'POST',
    }),
  exportAllShorts: (project: string, episode: string) =>
    request<{ status: string; results: unknown[]; exported_count?: number; total?: number }>(`/shorts/${project}/${episode}/export-all`, {
      method: 'POST',
    }),
  exportShortsSelection: (project: string, episode: string, indices: number[]) =>
    request<{ status: string; results: unknown[]; exported_count?: number; total?: number }>(`/shorts/${project}/${episode}/export-selection`, {
      method: 'POST',
      body: JSON.stringify({ indices }),
    }),

  // Transcripts
  getTranscriptInfo: (project: string, episode: string) =>
    request<TranscriptInfo>(`/transcripts/${project}/${episode}`),
  getSrt: async (project: string, episode: string) => {
    const res = await fetch(`${BASE}/transcripts/${project}/${episode}/srt`)
    if (!res.ok) return ''
    return res.text()
  },
  getTranscriptText: async (project: string, episode: string) => {
    const res = await fetch(`${BASE}/transcripts/${project}/${episode}/text`)
    if (!res.ok) return ''
    return res.text()
  },

  // Logs
  getLogs: (jobId: string) =>
    request<{ job_id: string; log: Array<{ step: string; msg: string; pct: number | null }> }>(
      `/logs/${jobId}`
    ),

  // Models
  validateModels: () =>
    request<ModelsStatusResponse>('/models/status'),
  refreshModels: () =>
    request<ModelsStatusResponse>('/models/refresh', { method: 'POST' }),
  addCustomModel: (config: CustomModelConfig) =>
    request<{ status: string; model: unknown }>('/models/custom', { method: 'POST', body: JSON.stringify(config) }),
  listCustomModels: () =>
    request<{ custom_models: unknown[] }>('/models/custom'),
  deleteCustomModel: (name: string) =>
    request<{ status: string; name: string }>(`/models/custom/${encodeURIComponent(name)}`, { method: 'DELETE' }),

  // System / Hardware
  getHardware: () =>
    request<HardwareInfo>('/system/hardware'),
  getProcessingConfig: () =>
    request<ProcessingConfig>('/system/processing-config'),
  updateProcessingConfig: (config: ProcessingConfig) =>
    request<{ status: string; config: ProcessingConfig }>('/system/processing-config', { method: 'PUT', body: JSON.stringify(config) }),
}
