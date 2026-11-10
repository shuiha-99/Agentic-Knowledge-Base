# AtlasDesk v2 部署指南

## 环境与启动
v2 使用 Python 3.11、PostgreSQL 15 和 Redis 7。Redis 保存分布式会话及任务协调状态，因此是必需依赖。API 默认监听 8088 端口，支持多个无状态副本；Worker 独立扩缩容。
API 启动命令是 `uvicorn atlasdesk.api:app --host 0.0.0.0 --port 8088`。Worker 启动命令是 `python -m atlasdesk.worker`。

## 环境变量
`ATLAS_DATABASE_URL` 对应数据库连接地址；`ATLAS_REDIS_URL` 对应 Redis 连接地址；`ATLAS_PORT` 默认 8088；`ATLAS_SIGNING_KEY` 为令牌签名密钥。生产环境必须从受控密钥配置注入签名密钥。
连接池默认大小为 10，最多额外允许 5 个溢出连接，配置名分别是 `ATLAS_DB_POOL_SIZE` 和 `ATLAS_DB_MAX_OVERFLOW`。每个 API 副本单独创建连接池。

## 探针与发布
`/healthz` 是存活探针，正常返回 HTTP 200；`/readyz` 检查数据库与 Redis，任意依赖不可用时返回 HTTP 503。负载均衡只向 readiness 成功的实例发流量。
滚动发布最多允许一个 API 副本不可用。新副本连续三次 readiness 成功后才接收流量。退出时设置 30 秒优雅关闭窗口。
v2 日志输出为标准输出中的 JSON。上传附件上限为 25 MiB，列表单页上限为 100 条。版本升级和回退必须同时检查数据库迁移是否可逆。
