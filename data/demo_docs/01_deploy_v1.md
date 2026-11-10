# AtlasDesk v1 部署指南

## 环境与启动
v1 使用 Python 3.10 和 PostgreSQL 14，Redis 6 是可选缓存。仅支持单个 API 实例；进程内保存会话，因此不能直接横向扩容。API 默认监听 8080 端口。
启动命令为 `python -m atlasdesk.server --config app.ini`。Worker 使用 `python -m atlasdesk.worker --config app.ini` 单独启动。

## 配置
`app.ini` 中的 `[database] url` 是数据库地址，`[cache] redis_url` 是缓存地址，`[server] port` 控制监听端口。`[auth] session_secret` 为会话签名密钥，禁止写入版本库。
v1 健康检查路径是 `/health`，成功返回 HTTP 200。v1 不提供 `/readyz`。应用日志默认写入本地 `logs/app.log`。

## 发布与限制
发布前停止 Worker 接单，等待已领取任务完成，再停止 API。执行数据库备份后替换程序并启动 API，最后启动 Worker。v1 发布期间需要维护窗口。
上传附件上限为 10 MiB，单页工单列表最多返回 50 条。v1 不支持 OAuth2 客户端凭据认证，也不支持 v2 的 `/api/v2/tickets` 路径。
