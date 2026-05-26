import { useEffect, useMemo, useState } from 'react';
import { ChatInterface } from './components/ChatInterface';
import { KnowledgeBaseManager } from './components/KnowledgeBaseManager';
import { ModelConfigPanel } from './components/ModelConfigPanel';
import { apiClient } from './lib/api';
import type { ChatState, KnowledgeBaseInfo, TaskType } from './types';

type WorkspaceKey = 'idea' | 'reading' | 'survey' | 'knowledge' | 'model';

interface Workspace {
  key: WorkspaceKey;
  label: string;
  icon: string;
  title: string;
  placeholder: string;
  taskType?: TaskType;
}

const WORKSPACES: Workspace[] = [
  {
    key: 'idea',
    label: 'Idea生成',
    icon: '+',
    title: '科研 Idea 生成',
    placeholder: '输入你的研究方向、约束或初步想法...',
    taskType: 'idea_review',
  },
  {
    key: 'reading',
    label: '文献阅读',
    icon: 'R',
    title: '文献阅读',
    placeholder: '输入论文标题、方法名或需要精读的问题...',
    taskType: 'literature_summary',
  },
  {
    key: 'survey',
    label: '文献调研',
    icon: 'S',
    title: '智能调研报告生成',
    placeholder: '请输入您想调研的问题...',
    taskType: 'research_plan',
  },
  {
    key: 'knowledge',
    label: '知识库管理',
    icon: 'K',
    title: '知识库管理',
    placeholder: '输入需要检索或管理的知识库问题...',
  },
  {
    key: 'model',
    label: '模型配置',
    icon: 'M',
    title: '模型配置',
    placeholder: '输入模型、检索或评估配置相关问题...',
  },
];

const CHAT_STORAGE_KEY = 'mars-scholar.workspace-chat-states.v1';

function createChatState(): ChatState {
  return {
    sessionId: '',
    messages: [],
    isLoading: false,
    requiresApproval: false,
    taskType: undefined,
    evaluationScores: undefined,
  };
}

function createWorkspaceStates(): Record<WorkspaceKey, ChatState> {
  return {
    idea: createChatState(),
    reading: createChatState(),
    survey: createChatState(),
    knowledge: createChatState(),
    model: createChatState(),
  };
}

function loadWorkspaceStates(): Record<WorkspaceKey, ChatState> {
  if (typeof window === 'undefined') return createWorkspaceStates();

  try {
    const stored = window.localStorage.getItem(CHAT_STORAGE_KEY);
    if (!stored) return createWorkspaceStates();

    const parsed = JSON.parse(stored) as Partial<Record<WorkspaceKey, ChatState>>;
    const defaults = createWorkspaceStates();

    return Object.fromEntries(
      WORKSPACES.map((workspace) => {
        const saved = parsed[workspace.key];
        if (!saved) return [workspace.key, defaults[workspace.key]];

        return [
          workspace.key,
          {
            ...defaults[workspace.key],
            ...saved,
            isLoading: false,
            messages: (saved.messages || []).map((message) => ({
              ...message,
              timestamp: new Date(message.timestamp),
            })),
          },
        ];
      })
    ) as Record<WorkspaceKey, ChatState>;
  } catch (error) {
    console.warn('恢复聊天记录失败:', error);
    return createWorkspaceStates();
  }
}

function App() {
  const [activeWorkspace, setActiveWorkspace] = useState<WorkspaceKey>('idea');
  const [workspaceStates, setWorkspaceStates] = useState(loadWorkspaceStates);
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBaseInfo[]>([]);
  const [selectedKnowledgeBaseId, setSelectedKnowledgeBaseId] = useState('');
  const currentWorkspace = WORKSPACES.find((item) => item.key === activeWorkspace) ?? WORKSPACES[0];
  const currentChatState = workspaceStates[activeWorkspace];

  const setCurrentChatState = useMemo(
    () => (updater: ChatState | ((previous: ChatState) => ChatState)) => {
      setWorkspaceStates((previousStates) => {
        const previous = previousStates[activeWorkspace];
        const next = typeof updater === 'function' ? updater(previous) : updater;
        return {
          ...previousStates,
          [activeWorkspace]: next,
        };
      });
    },
    [activeWorkspace]
  );

  useEffect(() => {
    const serializable = Object.fromEntries(
      WORKSPACES.map((workspace) => [
        workspace.key,
        {
          ...workspaceStates[workspace.key],
          isLoading: false,
        },
      ])
    );
    window.localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(serializable));
  }, [workspaceStates]);

  useEffect(() => {
    apiClient
      .listKnowledgeBases()
      .then((items) => {
        setKnowledgeBases(items);
        setSelectedKnowledgeBaseId((current) => current || items[0]?.id || '');
      })
      .catch((error) => console.warn('知识库列表加载失败:', error));
  }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">S</div>
          <div className="brand-text">MARS Scholar</div>
          <button className="sidebar-toggle" aria-label="收起侧边栏" type="button">
            &lsaquo;
          </button>
        </div>

        <nav className="workspace-nav" aria-label="工作区菜单">
          {WORKSPACES.map((workspace) => (
            <button
              key={workspace.key}
              className={`nav-item ${workspace.key === activeWorkspace ? 'active' : ''}`}
              onClick={() => setActiveWorkspace(workspace.key)}
              type="button"
            >
              <span className="nav-icon">{workspace.icon}</span>
              <span>{workspace.label}</span>
            </button>
          ))}
        </nav>
      </aside>

      <main className="workspace-main">
        {activeWorkspace === 'model' ? (
          <div className="config-workspace">
            <ModelConfigPanel onClose={() => setActiveWorkspace('idea')} />
          </div>
        ) : activeWorkspace === 'knowledge' ? (
          <KnowledgeBaseManager
            knowledgeBases={knowledgeBases}
            selectedKnowledgeBaseId={selectedKnowledgeBaseId}
            onKnowledgeBasesChange={setKnowledgeBases}
            onSelectKnowledgeBase={setSelectedKnowledgeBaseId}
          />
        ) : (
          <ChatInterface
            key={activeWorkspace}
            title={currentWorkspace.title}
            activeLabel={currentWorkspace.label}
            inputPlaceholder={currentWorkspace.placeholder}
            taskTypeHint={currentWorkspace.taskType}
            selectedKnowledgeBaseId={selectedKnowledgeBaseId}
            knowledgeBases={knowledgeBases}
            onKnowledgeBaseChange={setSelectedKnowledgeBaseId}
            chatState={currentChatState}
            setChatState={setCurrentChatState}
          />
        )}
      </main>
    </div>
  );
}

export default App;
