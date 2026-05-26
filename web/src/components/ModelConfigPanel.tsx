import { useState, useEffect } from 'react';
import { apiClient } from '../lib/api';
import type { ModelConfig } from '../types';

interface ModelConfigPanelProps {
  onClose: () => void;
}

export function ModelConfigPanel({ onClose }: ModelConfigPanelProps) {
  const [config, setConfig] = useState<ModelConfig>({
    multimodal_llm: {
      model_name: 'glm-4v-flash',
      base_url: 'https://open.bigmodel.cn/api/paas/v4/',
      api_key: '',
      temperature: 0.3,
    },
    multimodal_embedding: {
      model_name: 'qwen3-vl-embedding',
      base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
      api_key: '',
      embedding_dim: 2560,
    },
  });
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' } | null>(null);

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      setIsLoading(true);
      // 先尝试从服务器加载
      const response = await apiClient.getServerConfig();
      setConfig(response.config);
    } catch {
      // 服务器加载失败，尝试从本地加载
      const localConfig = apiClient.getConfig();
      if (localConfig) {
        setConfig(localConfig);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSave = async () => {
    try {
      setIsSaving(true);
      setMessage(null);
      const response = await apiClient.updateServerConfig(config);
      setMessage({ text: response.message, type: 'success' });
      // 3秒后隐藏成功消息
      setTimeout(() => setMessage(null), 3000);
    } catch (error) {
      setMessage({ text: `保存失败: ${error instanceof Error ? error.message : '未知错误'}`, type: 'error' });
    } finally {
      setIsSaving(false);
    }
  };

  const updateLLMConfig = (field: keyof typeof config.multimodal_llm, value: any) => {
    setConfig(prev => ({
      ...prev,
      multimodal_llm: {
        ...prev.multimodal_llm,
        [field]: value,
      },
    }));
  };

  const updateEmbeddingConfig = (field: keyof typeof config.multimodal_embedding, value: any) => {
    setConfig(prev => ({
      ...prev,
      multimodal_embedding: {
        ...prev.multimodal_embedding,
        [field]: value,
      },
    }));
  };

  if (isLoading) {
    return (
      <div className="config-panel">
        <div className="config-loading">加载中...</div>
      </div>
    );
  }

  return (
    <div className="config-panel">
      <div className="config-header">
        <h2>模型配置</h2>
        <button className="close-btn" onClick={onClose}>×</button>
      </div>

      {message && (
        <div className={`config-message ${message.type}`}>
          {message.text}
        </div>
      )}

      <div className="config-content">
        {/* 多模态 LLM 配置 */}
        <div className="config-section">
          <h3 className="config-section-title">
            <span className="config-icon">🤖</span>
            多模态 LLM 模型
          </h3>
          <div className="config-field">
            <label>模型名称</label>
            <input
              type="text"
              value={config.multimodal_llm.model_name}
              onChange={(e) => updateLLMConfig('model_name', e.target.value)}
              placeholder="例如: glm-4v-flash"
            />
          </div>
          <div className="config-field">
            <label>API 基础 URL</label>
            <input
              type="text"
              value={config.multimodal_llm.base_url}
              onChange={(e) => updateLLMConfig('base_url', e.target.value)}
              placeholder="例如: https://open.bigmodel.cn/api/paas/v4/"
            />
          </div>
          <div className="config-field">
            <label>API Key</label>
            <input
              type="password"
              value={config.multimodal_llm.api_key}
              onChange={(e) => updateLLMConfig('api_key', e.target.value)}
              placeholder="输入你的 API Key"
            />
          </div>
          <div className="config-field">
            <label>温度 (Temperature): {config.multimodal_llm.temperature.toFixed(2)}</label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={config.multimodal_llm.temperature}
              onChange={(e) => updateLLMConfig('temperature', parseFloat(e.target.value))}
            />
          </div>
        </div>

        {/* 多模态向量模型配置 */}
        <div className="config-section">
          <h3 className="config-section-title">
            <span className="config-icon">🧠</span>
            多模态向量模型
          </h3>
          <div className="config-field">
            <label>模型名称</label>
            <input
              type="text"
              value={config.multimodal_embedding.model_name}
              onChange={(e) => updateEmbeddingConfig('model_name', e.target.value)}
              placeholder="例如: qwen3-vl-embedding"
            />
          </div>
          <div className="config-field">
            <label>API 基础 URL</label>
            <input
              type="text"
              value={config.multimodal_embedding.base_url}
              onChange={(e) => updateEmbeddingConfig('base_url', e.target.value)}
              placeholder="例如: https://dashscope.aliyuncs.com/compatible-mode/v1"
            />
          </div>
          <div className="config-field">
            <label>API Key</label>
            <input
              type="password"
              value={config.multimodal_embedding.api_key}
              onChange={(e) => updateEmbeddingConfig('api_key', e.target.value)}
              placeholder="输入你的 API Key"
            />
          </div>
          <div className="config-field">
            <label>向量维度</label>
            <input
              type="number"
              value={config.multimodal_embedding.embedding_dim}
              onChange={(e) => updateEmbeddingConfig('embedding_dim', parseInt(e.target.value))}
              placeholder="例如: 2560"
            />
          </div>
        </div>
      </div>

      <div className="config-footer">
        <button className="btn btn-secondary" onClick={loadConfig}>
          重置
        </button>
        <button
          className="btn btn-primary"
          onClick={handleSave}
          disabled={isSaving}
        >
          {isSaving ? '保存中...' : '保存配置'}
        </button>
      </div>
    </div>
  );
}