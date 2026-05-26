import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiClient, fileUrl } from '../lib/api';
import type { ChatState, HistoryRecord, KnowledgeBaseInfo, Message, RetrievedImage, TaskType, WorkspaceKey } from '../types';

interface ChatInterfaceProps {
  workspaceKey: WorkspaceKey;
  viewMode?: 'new' | 'history';
  title?: string;
  activeLabel?: string;
  inputPlaceholder?: string;
  description?: string;
  taskTypeHint?: TaskType;
  selectedKnowledgeBaseId?: string;
  knowledgeBases?: KnowledgeBaseInfo[];
  onKnowledgeBaseChange?: (knowledgeBaseId: string) => void;
  chatState: ChatState;
  setChatState: (updater: ChatState | ((previous: ChatState) => ChatState)) => void;
  onOpenChat?: () => void;
  onNewChat?: () => void;
}

export function ChatInterface({
  workspaceKey,
  viewMode = 'new',
  title = '多模态科研助手',
  activeLabel = 'MARS Scholar',
  inputPlaceholder = '输入你的问题...',
  description = '',
  taskTypeHint,
  selectedKnowledgeBaseId = '',
  knowledgeBases = [],
  onKnowledgeBaseChange,
  chatState,
  setChatState,
  onOpenChat,
  onNewChat,
}: ChatInterfaceProps) {
  const [inputText, setInputText] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [historyRecords, setHistoryRecords] = useState<HistoryRecord[]>([]);
  const [historyQuery, setHistoryQuery] = useState('');
  const [historyFilter, setHistoryFilter] = useState<'all' | 'today' | 'week'>('all');
  const [selectedHistoryIds, setSelectedHistoryIds] = useState<Set<string>>(new Set());
  const [contextMenu, setContextMenu] = useState<{ messageId: string; x: number; y: number } | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatState.messages]);

  useEffect(() => {
    if (!selectedFile || !selectedFile.type.startsWith('image/')) {
      setPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    loadHistory();
  }, [workspaceKey]);

  useEffect(() => {
    if (!contextMenu) return;
    const close = () => setContextMenu(null);
    window.addEventListener('click', close);
    window.addEventListener('scroll', close, true);
    return () => {
      window.removeEventListener('click', close);
      window.removeEventListener('scroll', close, true);
    };
  }, [contextMenu]);

  const loadHistory = async (query = historyQuery) => {
    try {
      const records = await apiClient.listHistory(workspaceKey, query.trim() || undefined);
      setHistoryRecords(records);
    } catch (error) {
      console.warn('历史记录加载失败:', error);
    }
  };

  const addMessage = (
    role: 'user' | 'assistant',
    content: string,
    retrievedImages?: RetrievedImage[]
  ) => {
    const newMessage: Message = {
      id: `${Date.now()}-${role}-${Math.random().toString(16).slice(2)}`,
      role,
      content,
      timestamp: new Date(),
      progressSteps: role === 'assistant' ? [] : undefined,
      progressOpen: role === 'assistant' ? true : undefined,
      retrievedImages,
    };
    setChatState((previous) => ({
      ...previous,
      messages: [...previous.messages, newMessage],
    }));
    return newMessage.id;
  };

  const persistHistory = async (
    messages: Message[],
    sessionId?: string,
    finalTaskType?: string,
    finalKnowledgeBaseId?: string
  ) => {
    if (!messages.length || !['idea', 'reading', 'survey'].includes(workspaceKey)) return;
    const firstUser = messages.find((message) => message.role === 'user');
    const lastAssistant = [...messages].reverse().find((message) => message.role === 'assistant');
    const titleText = firstUser?.content || activeLabel;
    const summary = lastAssistant?.content.slice(0, 500) || '';
    try {
      await apiClient.saveHistory({
        workspace: workspaceKey as 'idea' | 'reading' | 'survey',
        title: titleText.split('\n')[0].slice(0, 120),
        topic: inferTopic(titleText),
        session_id: sessionId || chatState.sessionId || '',
        knowledge_base_id: finalKnowledgeBaseId || selectedKnowledgeBaseId || null,
        task_type: finalTaskType || taskTypeHint || null,
        messages: messages.map((message) => ({
          id: message.id,
          role: message.role,
          content: message.content,
          timestamp: message.timestamp.toISOString(),
        })),
        summary,
        tags: buildHistoryTags(messages, taskTypeHint),
      });
      await loadHistory();
    } catch (error) {
      console.warn('历史记录保存失败:', error);
    }
  };

  const updateMessageContent = (id: string, content: string) => {
    setChatState((previous) => ({
      ...previous,
      messages: previous.messages.map((message) =>
        message.id === id ? { ...message, content } : message
      ),
    }));
  };

  const updateRetrievedImages = (id: string, retrievedImages: RetrievedImage[]) => {
    setChatState((previous) => ({
      ...previous,
      messages: previous.messages.map((message) =>
        message.id === id ? { ...message, retrievedImages } : message
      ),
    }));
  };

  const addProgressStep = (id: string, title: string, detail: string, node?: string) => {
    setChatState((previous) => ({
      ...previous,
      messages: previous.messages.map((message) =>
        message.id !== id
          ? message
          : {
              ...message,
              progressSteps: [
                ...(message.progressSteps || []),
                {
                  id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
                  title,
                  detail,
                  node,
                },
              ].slice(-10),
            }
      ),
    }));
  };

  const toggleProgress = (id: string) => {
    setChatState((previous) => ({
      ...previous,
      messages: previous.messages.map((message) =>
        message.id === id ? { ...message, progressOpen: !message.progressOpen } : message
      ),
    }));
  };

  const deleteMessages = (ids: string[]) => {
    if (!ids.length) return;
    const label = ids.length === 1 ? '这条对话' : `${ids.length} 条对话`;
    if (!window.confirm(`确认删除${label}吗？删除后将从当前工作区记录中移除。`)) return;
    setChatState((previous) => {
      const nextMessages = previous.messages.filter((message) => !ids.includes(message.id));
      void persistHistory(nextMessages, previous.sessionId, previous.taskType, selectedKnowledgeBaseId);
      return {
        ...previous,
        messages: nextMessages,
      };
    });
  };

  const handleSend = async () => {
    if (!inputText.trim() && !selectedFile) return;

    const sessionId = chatState.requiresApproval ? undefined : chatState.sessionId || undefined;
    setChatState((previous) => ({
      ...previous,
      sessionId: sessionId || '',
      isLoading: true,
      requiresApproval: false,
    }));

    try {
      if (selectedFile) {
        const userContent = selectedFile.name + (inputText ? `\n${inputText}` : '');
        const userMessageId = addMessage('user', userContent);
        const response = await apiClient.uploadChat(
          selectedFile,
          inputText || undefined,
          sessionId,
          selectedKnowledgeBaseId || undefined,
          taskTypeHint
        );
        const assistantMessageId = addMessage('assistant', response.answer, normalizeRetrievedImages(response.state));
        const nextMessages = [
          ...chatState.messages,
          {
            id: userMessageId,
            role: 'user' as const,
            content: userContent,
            timestamp: new Date(),
          },
          {
            id: assistantMessageId,
            role: 'assistant' as const,
            content: response.answer,
            timestamp: new Date(),
            retrievedImages: normalizeRetrievedImages(response.state),
          },
        ];
        setChatState((previous) => ({
          ...previous,
          sessionId: response.session_id,
          isLoading: false,
          requiresApproval: response.requires_approval,
          taskType: response.task_type,
          evaluationScores: {
            evaluate_score: response.state.evaluate_score,
            response_relevancy: response.state.response_relevancy,
            context_relevance: response.state.context_relevance,
            context_precision: response.state.context_precision,
            faithfulness: response.state.faithfulness,
          },
        }));
        await persistHistory(nextMessages, response.session_id, response.task_type, selectedKnowledgeBaseId);
      } else {
        const userMessageId = addMessage('user', inputText);
        await streamAnswer(inputText, sessionId, userMessageId);
      }

      setInputText('');
      setSelectedFile(null);
    } catch (error) {
      console.error('发送失败:', error);
      addMessage('assistant', `抱歉，发生错误：${error instanceof Error ? error.message : '未知错误'}`);
      setChatState((previous) => ({ ...previous, isLoading: false }));
    }
  };

  const streamAnswer = async (message: string, sessionId?: string, userMessageId?: string) => {
    const assistantMessageId = addMessage('assistant', '');
    let streamedAnswer = '';
    const userMessage: Message | undefined = userMessageId
      ? {
          id: userMessageId,
          role: 'user',
          content: message,
          timestamp: new Date(),
        }
      : undefined;

    await apiClient.streamChat(
      {
        message,
        session_id: sessionId ?? (chatState.sessionId || undefined),
        knowledge_base_id: selectedKnowledgeBaseId || undefined,
        task_type_hint: taskTypeHint,
      },
      (event) => {
        if (event.event === 'error') {
          throw new Error(event.detail || '流式请求失败');
        }

        if (event.event === 'meta' && event.session_id) {
          setChatState((previous) => ({
            ...previous,
            sessionId: event.session_id || previous.sessionId,
          }));
        }

        if (event.event === 'progress' || event.event === 'approval_required') {
          addProgressStep(
            assistantMessageId,
            event.title || '工作流进度',
            event.detail || '节点已完成。',
            event.node
          );
        }

        if (event.event === 'answer_delta') {
          streamedAnswer += event.delta || '';
          updateMessageContent(assistantMessageId, streamedAnswer);
        }

        if (event.event === 'final') {
          const finalAnswer = event.answer || streamedAnswer;
          updateMessageContent(assistantMessageId, event.answer || streamedAnswer);
          updateRetrievedImages(assistantMessageId, normalizeRetrievedImages(event.state));
          const assistantMessage: Message = {
            id: assistantMessageId,
            role: 'assistant',
            content: finalAnswer,
            timestamp: new Date(),
            progressSteps: [],
            progressOpen: true,
            retrievedImages: normalizeRetrievedImages(event.state),
          };
          const nextMessages = [
            ...chatState.messages,
            ...(userMessage ? [userMessage] : []),
            assistantMessage,
          ];
          setChatState((previous) => ({
            ...previous,
            sessionId: event.session_id || previous.sessionId,
            isLoading: false,
            requiresApproval: Boolean(event.requires_approval),
            taskType: event.task_type,
            evaluationScores: {
              evaluate_score: event.state?.evaluate_score,
              response_relevancy: event.state?.response_relevancy,
              context_relevance: event.state?.context_relevance,
              context_precision: event.state?.context_precision,
              faithfulness: event.state?.faithfulness,
            },
          }));
          void persistHistory(nextMessages, event.session_id, event.task_type, selectedKnowledgeBaseId);
        }
      }
    );
  };

  const handleApproval = async (approved: boolean) => {
    if (!chatState.sessionId) return;

    setChatState((previous) => ({ ...previous, isLoading: true }));

    try {
      addMessage('user', approved ? '接受当前回答' : '拒绝当前回答，使用互联网兜底');
      await streamAnswer(approved ? 'approve' : 'rejected', chatState.sessionId);
    } catch (error) {
      console.error('审批失败:', error);
      setChatState((previous) => ({ ...previous, isLoading: false }));
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) setSelectedFile(file);
    event.target.value = '';
  };

  const handlePaste = (event: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const imageItem = Array.from(event.clipboardData.items).find((item) =>
      item.type.startsWith('image/')
    );
    const pastedFile = imageItem?.getAsFile();
    if (!pastedFile) return;

    const extension = pastedFile.type.split('/')[1] || 'png';
    setSelectedFile(
      new File([pastedFile], `pasted-image-${Date.now()}.${extension}`, {
        type: pastedFile.type,
      })
    );
    event.preventDefault();
  };

  const latestMessage = chatState.messages[chatState.messages.length - 1];
  const assistantIsStreaming = chatState.isLoading && latestMessage?.role === 'assistant';
  const filteredHistoryRecords = filterHistoryRecords(historyRecords, historyFilter);

  if (viewMode === 'history') {
    return (
      <div className="history-workspace">
        <div className="chat-header history-page-header">
          <div>
            <div className="workspace-eyebrow">{activeLabel}</div>
            <h1>历史{activeLabel.replace('生成', '')}</h1>
            {description && <p className="workspace-description">{description}</p>}
          </div>
          <div className="task-group">
            <button className="btn btn-primary btn-compact" type="button" onClick={onNewChat}>
              新建
            </button>
          </div>
        </div>

        <section className="history-panel history-page" aria-label={`${activeLabel}历史记录`}>
          <div className="history-toolbar">
            <input
              value={historyQuery}
              onChange={(event) => setHistoryQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') void loadHistory();
              }}
              placeholder="搜索历史主题、内容或摘要"
            />
            <select value={historyFilter} onChange={(event) => setHistoryFilter(event.target.value as typeof historyFilter)}>
              <option value="all">全部时间</option>
              <option value="today">今天</option>
              <option value="week">近 7 天</option>
            </select>
            <button className="btn btn-secondary btn-compact" type="button" onClick={() => loadHistory()}>
              搜索
            </button>
            <button className="btn btn-secondary btn-compact" type="button" onClick={() => exportHistory(filteredHistoryRecords, activeLabel)}>
              导出
            </button>
            <button
              className="btn btn-danger btn-compact"
              type="button"
              disabled={!selectedHistoryIds.size}
              onClick={async () => {
                const ids = Array.from(selectedHistoryIds);
                if (!window.confirm(`确认删除 ${ids.length} 条历史记录吗？`)) return;
                await apiClient.deleteHistory(ids);
                setSelectedHistoryIds(new Set());
                await loadHistory();
              }}
            >
              删除选中
            </button>
          </div>
          <div className="history-list history-page-list">
            {filteredHistoryRecords.map((record) => (
              <article className="history-item" key={record.id}>
                <label className="history-check">
                  <input
                    type="checkbox"
                    checked={selectedHistoryIds.has(record.id)}
                    onChange={() => {
                      setSelectedHistoryIds((previous) => {
                        const next = new Set(previous);
                        if (next.has(record.id)) next.delete(record.id);
                        else next.add(record.id);
                        return next;
                      });
                    }}
                  />
                </label>
                <button
                  className="history-open"
                  type="button"
                  onClick={() => {
                    setChatState((previous) => ({
                      ...previous,
                      sessionId: record.session_id,
                      messages: record.messages.map((message) => ({
                        id: message.id,
                        role: message.role,
                        content: message.content,
                        timestamp: new Date(message.timestamp),
                      })),
                      taskType: record.task_type as TaskType | undefined,
                    }));
                    onOpenChat?.();
                  }}
                >
                  <strong>{record.title}</strong>
                  <span>{record.topic || record.summary || '暂无摘要'}</span>
                  <small>{formatDate(record.updated_at)} · {record.messages.length} 条 · {record.tags.join(' / ') || '未分类'}</small>
                </button>
                <button
                  className="history-delete"
                  type="button"
                  onClick={async () => {
                    if (!window.confirm('确认删除这条历史记录吗？')) return;
                    await apiClient.deleteHistory([record.id]);
                    await loadHistory();
                  }}
                  aria-label="删除历史记录"
                >
                  删除
                </button>
              </article>
            ))}
            {!filteredHistoryRecords.length && <div className="history-empty">暂无可显示的历史记录。</div>}
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <div>
          <div className="workspace-eyebrow">{activeLabel}</div>
          <h1>{title}</h1>
          {description && <p className="workspace-description">{description}</p>}
        </div>
        <div className="task-group">
          {onKnowledgeBaseChange && (
            <label className="kb-selector">
              <span>知识库</span>
              <select
                value={selectedKnowledgeBaseId}
                onChange={(event) => onKnowledgeBaseChange(event.target.value)}
              >
                <option value="">默认库</option>
                {knowledgeBases.map((knowledgeBase) => (
                  <option key={knowledgeBase.id} value={knowledgeBase.id}>
                    {knowledgeBase.name}
                  </option>
                ))}
              </select>
            </label>
          )}
          {taskTypeHint && <div className="task-type muted">推荐任务：{taskTypeHint}</div>}
          {chatState.taskType && <div className="task-type">当前任务：{chatState.taskType}</div>}
        </div>
      </div>

      <div className="messages-container">
        {chatState.messages.length === 0 && !chatState.isLoading && (
          <div className="empty-state">
            <div className="empty-title">多模态RAG科研助手</div>
            <div className="empty-copy">
              可以围绕各个领域科研方向进行 idea 生成、文献阅读、文献调研、知识库检索与实验方案讨论。
            </div>
          </div>
        )}

        {chatState.messages.map((message) => (
          <div key={message.id} className={`message ${message.role}`}>
            <div className="message-role">{message.role === 'user' ? '用户' : 'AI'}</div>
            <div
              className="message-content"
              onContextMenu={(event) => {
                event.preventDefault();
                setContextMenu({ messageId: message.id, x: event.clientX, y: event.clientY });
              }}
            >
              {message.role === 'assistant' && Boolean(message.progressSteps?.length) && (
                <div className="message-progress">
                  <button
                    className="progress-toggle"
                    type="button"
                    onClick={() => toggleProgress(message.id)}
                  >
                    <span>{message.progressOpen ? '收起执行过程' : '展开执行过程'}</span>
                    <span>{message.progressSteps?.length || 0} 步</span>
                  </button>
                  {message.progressOpen && (
                    <div className="progress-list">
                      {message.progressSteps?.map((step) => (
                        <div className="progress-step" key={step.id}>
                          <span className="progress-dot" />
                          <div>
                            <div className="progress-step-title">{step.title}</div>
                            <div className="progress-step-detail">{step.detail}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>

              {message.role === 'assistant' && Boolean(message.retrievedImages?.length) && (
                <div className="retrieved-images">
                  {message.retrievedImages?.map((image, index) => (
                    <figure className="retrieved-image-card" key={`${image.image_path}-${index}`}>
                      <img src={fileUrl(image.image_path)} alt={image.title || image.filename || '检索图片'} />
                      <figcaption>
                        <div className="retrieved-image-title">
                          {index + 1}. {image.title || image.filename || '知识库图片'}
                        </div>
                        <div className="retrieved-image-meta">{formatImageMeta(image)}</div>
                        {image.text && <p>{image.text}</p>}
                      </figcaption>
                    </figure>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {contextMenu && (
          <div
            className="message-context-menu"
            style={{ left: contextMenu.x, top: contextMenu.y }}
            role="menu"
          >
            <button
              type="button"
              onClick={() => {
                deleteMessages([contextMenu.messageId]);
                setContextMenu(null);
              }}
            >
              删除该对话
            </button>
          </div>
        )}

        {chatState.isLoading && !assistantIsStreaming && (
          <div className="message assistant">
            <div className="message-role">AI</div>
            <div className="message-content">思考中...</div>
          </div>
        )}

        {chatState.requiresApproval && (
          <div className="approval-container">
            <div className="approval-message">
              <p>系统检测到回答可能不够准确，请确认是否接受当前回答。</p>
              {chatState.evaluationScores?.response_relevancy && (
                <p>回答相关性分数：{chatState.evaluationScores.response_relevancy.toFixed(2)}</p>
              )}
            </div>
            <div className="approval-buttons">
              <button
                className="btn btn-approve"
                onClick={() => handleApproval(true)}
                disabled={chatState.isLoading}
              >
                接受
              </button>
              <button
                className="btn btn-reject"
                onClick={() => handleApproval(false)}
                disabled={chatState.isLoading}
              >
                拒绝（将使用互联网搜索补充）
              </button>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        {selectedFile && (
          <div className="file-preview">
            {previewUrl && <img src={previewUrl} alt="" className="file-preview-thumb" />}
            <span className="file-preview-name">{selectedFile.name}</span>
            <button onClick={() => setSelectedFile(null)} type="button" aria-label="移除已选择文件">
              &times;
            </button>
          </div>
        )}
        <div className="input-row">
          <button
            className="btn-file"
            onClick={() => document.getElementById('file-input')?.click()}
            type="button"
            title="上传图片"
            aria-label="上传图片"
          >
            📎
          </button>
          <input
            id="file-input"
            type="file"
            accept="image/*,.png,.jpg,.jpeg,.webp,.gif"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          <textarea
            value={inputText}
            onChange={(event) => setInputText(event.target.value)}
            onPaste={handlePaste}
            onKeyPress={handleKeyPress}
            placeholder={inputPlaceholder}
            rows={1}
            disabled={chatState.isLoading}
          />
          <button
            className="btn-send"
            onClick={handleSend}
            disabled={chatState.isLoading || (!inputText.trim() && !selectedFile)}
          >
            发送
          </button>
        </div>
      </div>
    </div>
  );
}

function normalizeRetrievedImages(state?: Record<string, unknown>): RetrievedImage[] {
  const images = state?.images_retrieved;
  if (!Array.isArray(images)) return [];

  return images
    .map((image) => {
      if (typeof image === 'string') return { image_path: image };
      if (image && typeof image === 'object' && 'image_path' in image) {
        return image as RetrievedImage;
      }
      return null;
    })
    .filter((image): image is RetrievedImage => Boolean(image?.image_path))
    .slice(0, 6);
}

function formatImageMeta(image: RetrievedImage): string {
  const parts: string[] = [];

  if (image.category) parts.push(image.category);
  if (image.score !== undefined && image.score !== null) {
    parts.push(`相似度 ${Number(image.score).toFixed(3)}`);
  }
  if (image.page_start !== undefined && image.page_start !== null) {
    const pageText =
      image.page_end !== undefined && image.page_end !== null && image.page_end !== image.page_start
        ? `pages ${image.page_start}-${image.page_end}`
        : `page ${image.page_start}`;
    parts.push(pageText);
  }

  return parts.join(' · ') || '来自知识库图片检索';
}

function inferTopic(text: string): string {
  const normalized = text.replace(/\s+/g, ' ').trim();
  if (!normalized) return '';
  return normalized.length > 48 ? `${normalized.slice(0, 48)}...` : normalized;
}

function buildHistoryTags(messages: Message[], taskType?: TaskType): string[] {
  const tags = new Set<string>();
  if (taskType) tags.add(taskType);
  const joined = messages.map((message) => message.content).join(' ').toLowerCase();
  if (/baseline|指标|实验|ablation|消融/.test(joined)) tags.add('实验设计');
  if (/github|code|开源|代码/.test(joined)) tags.add('代码线索');
  if (/risk|风险|局限|limitation/.test(joined)) tags.add('风险评估');
  return Array.from(tags).slice(0, 6);
}

function filterHistoryRecords(records: HistoryRecord[], filter: 'all' | 'today' | 'week'): HistoryRecord[] {
  if (filter === 'all') return records;
  const now = Date.now();
  const dayMs = 24 * 60 * 60 * 1000;
  return records.filter((record) => {
    const time = new Date(record.updated_at).getTime();
    if (Number.isNaN(time)) return false;
    if (filter === 'today') {
      return new Date(record.updated_at).toDateString() === new Date().toDateString();
    }
    return now - time <= 7 * dayMs;
  });
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function exportHistory(records: HistoryRecord[], label: string): void {
  const payload = {
    module: label,
    exported_at: new Date().toISOString(),
    records,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${label}-history-${new Date().toISOString().slice(0, 10)}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}
