# AtlasDesk v1 到 v2 迁移清单

## 升级前提
只支持从 v1.8 及以上版本直接升级到 v2.0。低于 v1.8 时先升级到 v1.8。正式迁移前必须验证最近一次全量备份可恢复，并在测试环境演练。
确认 PostgreSQL 已从 14 升级到 15、Redis 已从可选 Redis 6 切换到必需 Redis 7。保留 v1 配置和程序包，记录迁移前数据库快照标识。

## 配置映射
`app.ini [database] url` 迁移为 `ATLAS_DATABASE_URL`；`[cache] redis_url` 迁移为 `ATLAS_REDIS_URL`；`[server] port` 迁移为 `ATLAS_PORT`，默认值从 8080 变为 8088。
v1 的 `session_secret` 不能直接作为 v2 OAuth2 客户端密钥。v2 要分别配置令牌签名密钥与客户端凭据，已有客户端需要重新签发凭据。

## 执行顺序
暂停外部写入并让 v1 Worker 排空任务；执行全量备份；运行 `atlasdesk migrate --to 2.0`；检查迁移结果；配置并启动 v2 API；通过 `/readyz` 验证数据库和 Redis；启动 v2 Worker；进行创建工单、查询工单与回调冒烟检查；最后恢复外部写入。
v2.0 新增的 `ticket_events` 表允许可逆迁移，但在 v2 已接收业务写入后，不允许直接用旧版程序连接数据库。是否回退必须参考回滚手册的写入边界。
