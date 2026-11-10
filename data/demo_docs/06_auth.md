# AtlasDesk v2 认证与授权

## 客户端凭据
服务间调用采用 OAuth2 client_credentials。客户端向 `POST /oauth/token` 提交 client_id 和 client_secret 获取令牌。访问令牌有效期为 3600 秒，client_credentials 不签发 refresh_token。
client_secret 存放在受控密钥配置中，禁止写入代码、工单正文和日志。轮换时允许新旧客户端凭据并存15分钟，窗口结束后撤销旧凭据。

## 权限与租户
`tickets:read` 用于查询，`tickets:write` 用于创建或修改。仅有 read 权限的客户端不能创建工单。服务端从已验证令牌中读取 tenant_id，不信任请求体自报的租户。
令牌缺失、过期或签名不合法返回401。令牌有效但缺少操作权限返回403；对无权访问的具体工单资源返回404。

## 旧版区别
v1 使用服务端会话，不支持 v2 的 OAuth2 客户端凭据流程。升级时需要为调用方重新创建客户端并授权，不能复用 v1 的 session_secret 作为客户端密码。
