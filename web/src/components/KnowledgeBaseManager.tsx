import { useEffect, useRef, useState } from 'react';
import { apiClient } from '../lib/api';
import type { KnowledgeBaseInfo, KnowledgeBaseJobInfo } from '../types';

interface KnowledgeBaseManagerProps {
  knowledgeBases: KnowledgeBaseInfo[];
  selectedKnowledgeBaseId: string;
  onKnowledgeBasesChange: (knowledgeBases: KnowledgeBaseInfo[]) => void;
  onSelectKnowledgeBase: (knowledgeBaseId: string) => void;
}

export function KnowledgeBaseManager({
  knowledgeBases,
  selectedKnowledgeBaseId,
  onKnowledgeBasesChange,
  onSelectKnowledgeBase,
}: KnowledgeBaseManagerProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [uploading, setUploading] = useState(false);
  const [job, setJob] = useState<KnowledgeBaseJobInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const selectedKnowledgeBase =
    knowledgeBases.find((knowledgeBase) => knowledgeBase.id === selectedKnowledgeBaseId) ||
    knowledgeBases[0] ||
    null;

  useEffect(() => {
    if (!selectedKnowledgeBaseId && knowledgeBases[0]) {
      onSelectKnowledgeBase(knowledgeBases[0].id);
    }
  }, [knowledgeBases, onSelectKnowledgeBase, selectedKnowledgeBaseId]);

  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return;

    const timer = window.setInterval(async () => {
      try {
        const response = await apiClient.getKnowledgeBaseJob(job.id);
        setJob(response.job);
        if (!['queued', 'running'].includes(response.job.status)) {
          const latest = await apiClient.listKnowledgeBases();
          onKnowledgeBasesChange(latest);
        }
      } catch (pollError) {
        setError(pollError instanceof Error ? pollError.message : '任务状态获取失败');
      }
    }, 1800);

    return () => window.clearInterval(timer);
  }, [job, onKnowledgeBasesChange]);

  const refreshKnowledgeBases = async () => {
    const latest = await apiClient.listKnowledgeBases();
    onKnowledgeBasesChange(latest);
  };

  const handleCreate = async () => {
    if (!name.trim()) return;
    try {
      setError(null);
      const response = await apiClient.createKnowledgeBase(name, description);
      const latest = await apiClient.listKnowledgeBases();
      onKnowledgeBasesChange(latest);
      onSelectKnowledgeBase(response.knowledge_base.id);
      setName('');
      setDescription('');
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : '创建知识库失败');
    }
  };

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    event.target.value = '';
    if (!selectedKnowledgeBase || !files.length) return;

    try {
      setUploading(true);
      setError(null);
      for (const file of files) {
        await apiClient.uploadKnowledgeBaseDocument(selectedKnowledgeBase.id, file);
      }
      await refreshKnowledgeBases();
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : '上传失败');
    } finally {
      setUploading(false);
    }
  };

  const handleStartIngest = async () => {
    if (!selectedKnowledgeBase) return;
    try {
      setError(null);
      const response = await apiClient.startKnowledgeBaseIngest(selectedKnowledgeBase.id, {
        create_collection: true,
        drop_existing: selectedKnowledgeBase.status === 'ready',
        force_ocr: false,
        use_model_descriptions: false,
      });
      setJob(response.job);
      await refreshKnowledgeBases();
    } catch (ingestError) {
      setError(ingestError instanceof Error ? ingestError.message : '启动入库任务失败');
    }
  };

  return (
    <div className="kb-workspace">
      <header className="kb-header">
        <div>
          <div className="workspace-eyebrow">知识库管理</div>
          <h1>构建可检索论文知识库</h1>
        </div>
        <button className="btn btn-secondary" onClick={refreshKnowledgeBases} type="button">
          刷新
        </button>
      </header>

      {error && <div className="kb-alert">{error}</div>}

      <div className="kb-layout">
        <aside className="kb-list">
          <div className="kb-create">
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="知识库名称"
            />
            <textarea
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="描述，可选"
              rows={3}
            />
            <button className="btn btn-primary" onClick={handleCreate} type="button" disabled={!name.trim()}>
              新建知识库
            </button>
          </div>

          <div className="kb-items">
            {knowledgeBases.map((knowledgeBase) => (
              <button
                key={knowledgeBase.id}
                className={`kb-item ${selectedKnowledgeBase?.id === knowledgeBase.id ? 'active' : ''}`}
                onClick={() => onSelectKnowledgeBase(knowledgeBase.id)}
                type="button"
              >
                <span>{knowledgeBase.name}</span>
                <small>{statusLabel(knowledgeBase.status)} · {knowledgeBase.document_count} PDF</small>
              </button>
            ))}
            {!knowledgeBases.length && <div className="kb-empty">还没有知识库。</div>}
          </div>
        </aside>

        <section className="kb-detail">
          {selectedKnowledgeBase ? (
            <>
              <div className="kb-detail-top">
                <div>
                  <h2>{selectedKnowledgeBase.name}</h2>
                  <p>{selectedKnowledgeBase.description || '暂无描述'}</p>
                  <code>{selectedKnowledgeBase.collection_name}</code>
                </div>
                <span className={`kb-status ${selectedKnowledgeBase.status}`}>
                  {statusLabel(selectedKnowledgeBase.status)}
                </span>
              </div>

              <div className="kb-stats">
                <Metric label="PDF" value={selectedKnowledgeBase.document_count} />
                <Metric label="文本块" value={selectedKnowledgeBase.chunk_count} />
                <Metric label="图片/表格" value={selectedKnowledgeBase.image_asset_count} />
              </div>

              <div className="kb-actions">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="application/pdf,.pdf"
                  multiple
                  onChange={handleUpload}
                  hidden
                />
                <button
                  className="btn btn-secondary"
                  onClick={() => fileInputRef.current?.click()}
                  type="button"
                  disabled={uploading}
                >
                  {uploading ? '上传中...' : '上传 PDF'}
                </button>
                <button
                  className="btn btn-primary"
                  onClick={handleStartIngest}
                  type="button"
                  disabled={!selectedKnowledgeBase.document_count || job?.status === 'running'}
                >
                  开始 OCR / 切块 / 入库
                </button>
              </div>

              <div className="kb-pipeline">
                {['上传 PDF', 'OCR', '清洗切块', '向量化', '写入数据库'].map((step, index) => (
                  <div className="kb-step" key={step}>
                    <span>{index + 1}</span>
                    <strong>{step}</strong>
                  </div>
                ))}
              </div>

              {job && (
                <div className="kb-job">
                  <div className="kb-job-head">
                    <strong>{job.current_step}</strong>
                    <span>{Math.round(job.progress * 100)}%</span>
                  </div>
                  <div className="kb-progress">
                    <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
                  </div>
                  <pre>{job.logs.slice(-16).join('\n')}</pre>
                </div>
              )}

              <div className="kb-documents">
                <h3>文档</h3>
                {selectedKnowledgeBase.documents.map((document) => (
                  <div className="kb-document" key={document.id}>
                    <span>{document.filename}</span>
                    <small>{formatBytes(document.size_bytes)} · {document.status}</small>
                  </div>
                ))}
                {!selectedKnowledgeBase.documents.length && <div className="kb-empty">请先上传 PDF。</div>}
              </div>
            </>
          ) : (
            <div className="kb-empty large">新建一个知识库后，就可以上传 PDF 并启动处理流水线。</div>
          )}
        </section>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="kb-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    empty: '空库',
    uploaded: '已上传',
    processing: '处理中',
    ready: '可检索',
    failed: '失败',
  };
  return labels[status] || status;
}

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}
