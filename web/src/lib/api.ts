import type {
  ChatRequest,
  ChatResponse,
  ChatStreamEvent,
  HealthResponse,
  ResearchRequest,
  ResearchResponse,
  TaskInfo,
  ModelConfig,
  ConfigResponse,
  KnowledgeBaseInfo,
  KnowledgeBaseIngestRequest,
  KnowledgeBaseJobResponse,
  KnowledgeBaseListResponse,
  KnowledgeBaseResponse,
} from '../types';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';
const CONFIG_STORAGE_KEY = 'recommendate_model_config';

export const fileUrl = (path: string) => `${API_BASE}/files?path=${encodeURIComponent(path)}`;

class ApiClient {
  private config: ModelConfig | null = null;

  constructor() {
    // 从 localStorage 加载配置
    this.loadConfigFromStorage();
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE}${endpoint}`;
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `请求失败: ${response.status}`);
    }

    return response.json();
  }

  private loadConfigFromStorage() {
    try {
      const stored = localStorage.getItem(CONFIG_STORAGE_KEY);
      if (stored) {
        this.config = JSON.parse(stored);
      }
    } catch {
      this.config = null;
    }
  }

  private saveConfigToStorage(config: ModelConfig) {
    this.config = config;
    localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(config));
  }

  getConfig(): ModelConfig | null {
    return this.config;
  }

  async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  async getTasks(): Promise<TaskInfo[]> {
    return this.request<TaskInfo[]>('/tasks');
  }

  async getServerConfig(): Promise<ConfigResponse> {
    return this.request<ConfigResponse>('/config');
  }

  async updateServerConfig(config: ModelConfig): Promise<ConfigResponse> {
    const response = await this.request<ConfigResponse>('/config', {
      method: 'POST',
      body: JSON.stringify(config),
    });
    this.saveConfigToStorage(config);
    return response;
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    return this.request<ChatResponse>('/chat', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async streamChat(
    request: ChatRequest,
    onEvent: (event: ChatStreamEvent) => void
  ): Promise<void> {
    const response = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `流式请求失败: ${response.status}`);
    }

    if (!response.body) {
      throw new Error('浏览器不支持流式响应。');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        onEvent(JSON.parse(trimmed) as ChatStreamEvent);
      }
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      onEvent(JSON.parse(buffer.trim()) as ChatStreamEvent);
    }
  }

  async uploadChat(
    file: File,
    message?: string,
    sessionId?: string,
    knowledgeBaseId?: string,
    taskTypeHint?: ChatRequest['task_type_hint'],
    userName: string = 'ZS'
  ): Promise<ChatResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (message) formData.append('message', message);
    if (sessionId) formData.append('session_id', sessionId);
    if (knowledgeBaseId) formData.append('knowledge_base_id', knowledgeBaseId);
    if (taskTypeHint) formData.append('task_type_hint', taskTypeHint);
    formData.append('user_name', userName);

    const response = await fetch(`${API_BASE}/chat/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `文件上传失败: ${response.status}`);
    }

    return response.json();
  }

  async research(request: ResearchRequest): Promise<ResearchResponse> {
    return this.request<ResearchResponse>('/research', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async listKnowledgeBases(): Promise<KnowledgeBaseInfo[]> {
    const response = await this.request<KnowledgeBaseListResponse>('/knowledge-bases');
    return response.knowledge_bases;
  }

  async createKnowledgeBase(name: string, description: string): Promise<KnowledgeBaseResponse> {
    return this.request<KnowledgeBaseResponse>('/knowledge-bases', {
      method: 'POST',
      body: JSON.stringify({ name, description }),
    });
  }

  async uploadKnowledgeBaseDocument(
    knowledgeBaseId: string,
    file: File
  ): Promise<KnowledgeBaseResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE}/knowledge-bases/${knowledgeBaseId}/documents`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `文件上传失败: ${response.status}`);
    }

    return response.json();
  }

  async startKnowledgeBaseIngest(
    knowledgeBaseId: string,
    request: KnowledgeBaseIngestRequest
  ): Promise<KnowledgeBaseJobResponse> {
    return this.request<KnowledgeBaseJobResponse>(`/knowledge-bases/${knowledgeBaseId}/ingest`, {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getKnowledgeBaseJob(jobId: string): Promise<KnowledgeBaseJobResponse> {
    return this.request<KnowledgeBaseJobResponse>(`/knowledge-bases/jobs/${jobId}`);
  }
}

export const apiClient = new ApiClient();
