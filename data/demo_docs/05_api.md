# AtlasDesk v2 工单 API

## 创建工单
`POST /api/v2/tickets` 创建工单，需要 Bearer 访问令牌与 `tickets:write` 权限。请求体包含 `title`、`description` 和 `priority`。title 长度为 1 到 120 字符，priority 只允许 low、normal、high。
创建成功返回 HTTP 201，响应包含工单 `id` 和 `status`。缺少必填字段或字段校验失败返回 422。

## 幂等约定
客户端应提供 `Idempotency-Key` 请求头。相同租户下，同一幂等键在 24 小时内重复提交相同请求体，返回首次创建结果，不重复创建工单。相同键对应不同请求体时返回 HTTP 409。
幂等键只能约束创建 API 的重复请求，不能据此认定通知回调恰好执行一次。

## 查询与分页
`GET /api/v2/tickets/{id}` 查询工单，需要 `tickets:read` 权限。无权读取当前租户工单时返回 404，以减少跨租户资源枚举。
`GET /api/v2/tickets` 使用游标分页，默认20条，每页最多100条；下一页游标在 `next_cursor` 字段。不要将 v1 的50条上限套用到 v2。
