export type TaskType = 'qa' | 'idea_review' | 'literature_summary' | 'paper_compare' | 'research_plan';

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface TaskInfo {
  task_type: TaskType;
  label: string;
  description: string;
}

export interface MultimodalLLMConfig {
  model_name: string;
  base_url: string;
  api_key: string;
  temperature: number;
}

export interface MultimodalEmbeddingConfig {
  model_name: string;
  base_url: string;
  api_key: string;
  embedding_dim: number;
}

export interface ModelConfig {
  multimodal_llm: MultimodalLLMConfig;
  multimodal_embedding: MultimodalEmbeddingConfig;
}

export interface ConfigResponse {
  config: ModelConfig;
  message: string;
}

export interface ChatRequest {
  message?: string;
  session_id?: string;
  user_name?: string;
  knowledge_base_id?: string;
  task_type_hint?: TaskType;
  image_path?: string;
  image_data_url?: string;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  requires_approval: boolean;
  task_type?: TaskType;
  state: Record<string, any>;
}

export interface ChatStreamEvent {
  event: 'meta' | 'progress' | 'approval_required' | 'answer_delta' | 'final' | 'error';
  session_id?: string;
  node?: string;
  title?: string;
  detail?: string;
  delta?: string;
  answer?: string;
  requires_approval?: boolean;
  task_type?: TaskType;
  state?: Record<string, any>;
}

export interface ProgressStep {
  id: string;
  title: string;
  detail: string;
  node?: string;
}

export interface RetrievedImage {
  id?: string | number;
  doc_id?: string;
  category?: string;
  paper_id?: string;
  title?: string;
  text?: string;
  filename?: string;
  image_path: string;
  page_start?: number;
  page_end?: number;
  score?: number;
}

export interface ResearchRequest {
  query: string;
  task_type?: TaskType;
  top_k?: number;
  on_demand_top_k?: number;
  use_llm_reader?: boolean;
  use_milvus?: boolean;
}

export interface ResearchResponse {
  task_type: string;
  status: string;
  retrieval_source: string;
  final_report: string;
  related_work: any[];
  task_paper_profiles: any[];
  scores: Record<string, any>;
  agent_trace: any[];
  agent_outputs: Record<string, any>;
  gaps: any[];
  innovation_suggestions: any[];
  experiment_suggestions: any[];
  risk_assessment: any[];
  next_steps: string[];
  raw: Record<string, any>;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  progressSteps?: ProgressStep[];
  progressOpen?: boolean;
  retrievedImages?: RetrievedImage[];
}

export interface ChatState {
  sessionId: string;
  messages: Message[];
  isLoading: boolean;
  requiresApproval: boolean;
  taskType?: TaskType;
  evaluationScores?: {
    evaluate_score?: number;
    response_relevancy?: number;
    context_relevance?: number;
    context_precision?: number;
    faithfulness?: number;
  };
}

export interface KnowledgeBaseDocument {
  id: string;
  filename: string;
  stored_path: string;
  size_bytes: number;
  status: string;
  created_at: string;
}

export interface KnowledgeBaseInfo {
  id: string;
  name: string;
  description: string;
  collection_name: string;
  status: string;
  document_count: number;
  chunk_count: number;
  image_asset_count: number;
  created_at: string;
  updated_at: string;
  documents: KnowledgeBaseDocument[];
}

export interface KnowledgeBaseListResponse {
  knowledge_bases: KnowledgeBaseInfo[];
}

export interface KnowledgeBaseResponse {
  knowledge_base: KnowledgeBaseInfo;
  message: string;
}

export interface KnowledgeBaseIngestRequest {
  force_ocr?: boolean;
  create_collection?: boolean;
  drop_existing?: boolean;
  use_model_descriptions?: boolean;
}

export interface KnowledgeBaseJobInfo {
  id: string;
  knowledge_base_id: string;
  status: string;
  current_step: string;
  progress: number;
  logs: string[];
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeBaseJobResponse {
  job: KnowledgeBaseJobInfo;
}
