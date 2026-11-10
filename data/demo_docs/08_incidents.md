# AtlasDesk v2 常见故障

## Redis 故障
Redis 不可用会导致 `/readyz` 返回503，新请求不应继续路由到该副本。检查 `ATLAS_REDIS_URL`、网络和认证配置，恢复后观察队列积压。v2 不能按 v1 的可选缓存模式绕过 Redis。

## 数据库连接耗尽
每个 API 副本最多使用 pool_size 加 max_overflow 个连接。默认10加5，共15个；四个副本的API连接峰值为60，不包含 Worker、运维和监控连接。
PostgreSQL 演示环境 max_connections 为100，至少预留20个给 Worker、运维与监控。因此按默认连接池配置，API 副本数不能只按100除以10计算；需要按单副本最多15个连接并考虑80个API预算。

## 重复回调
通知采用至少一次投递。超时、连接中断或非2xx响应会重试，最大5次；退避间隔为1、2、4、8、16分钟。消费者应按 event_id 去重。
工单创建接口的 Idempotency-Key 不会自动为回调消费者去重，不能据此宣称通知恰好一次。
