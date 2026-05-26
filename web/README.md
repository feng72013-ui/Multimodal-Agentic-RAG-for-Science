# 推荐系统论文助手 - 前端

基于 React + Vite + TypeScript 构建的前端应用。

## 前置条件

- Node.js 18+
- npm 或 yarn

## 安装依赖

```bash
cd web
npm install
```

## 启动开发服务器

```bash
npm run dev
```

开发服务器将在 `http://localhost:3000` 启动。

如果后端不是默认的 `8000` 端口，比如你用了 `PORT=8001 bash scripts/run_fastapi.sh`，请这样启动前端：

```bash
VITE_BACKEND_TARGET=http://localhost:8001 npm run dev
```

## 构建生产版本

```bash
npm run build
```

## 功能特性

- ✅ 文本对话
- ✅ 图片上传和问答
- ✅ Markdown 渲染支持
- ✅ 会话保持
- ✅ 人工审批流程
- ✅ 评估分数展示

## API 代理配置

Vite 默认将 `/api`、`/assets`、`/processed_ocr` 转发到 `http://localhost:8000`。如果后端运行在其他端口，可以通过 `VITE_BACKEND_TARGET` 覆盖，例如 `VITE_BACKEND_TARGET=http://localhost:8001 npm run dev`。

## 项目结构

```
web/
├── src/
│   ├── App.tsx                # 主应用
│   ├── main.tsx               # 入口文件
│   ├── types.ts               # TypeScript 类型
│   └── index.css              # 样式
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```
