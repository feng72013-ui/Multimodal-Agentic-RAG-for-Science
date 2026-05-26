interface SidebarProps {
  activeTab: 'chat' | 'config';
  onTabChange: (tab: 'chat' | 'config') => void;
}

export function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h1>MARS Scholar</h1>
        <p className="sidebar-subtitle">多模态科研助手</p>
      </div>

      <nav className="sidebar-nav">
        <button
          className={`sidebar-nav-item ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => onTabChange('chat')}
        >
          <span className="nav-icon">💬</span>
          <span>对话</span>
        </button>
        <button
          className={`sidebar-nav-item ${activeTab === 'config' ? 'active' : ''}`}
          onClick={() => onTabChange('config')}
        >
          <span className="nav-icon">⚙️</span>
          <span>模型配置</span>
        </button>
      </nav>

      <div className="sidebar-footer">
        <p className="sidebar-footer-text">多模态多智能体 RAG 科研助手</p>
        <p className="sidebar-footer-version">v1.0.0</p>
      </div>
    </div>
  );
}
