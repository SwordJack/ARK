# Live Tests

本目录存放知识库模块的手动 live test 脚本，不会被 pytest 收集，永远不会在 CI 中运行。

## 脚本说明

| 脚本                     | 用途                                                                  |
| ------------------------ | --------------------------------------------------------------------- |
| `run_knowledge_basic.py` | 对接真实 Embedding API（OpenAI / DashScope / AIMLAPI）的冒烟测试      |
| `run_lifecycle_sql.py`   | 全生命周期测试：对接真实 DashScope Embedding API，**PostgreSQL** 存储 |
| `run_lifecycle_mongo.py` | 全生命周期测试：对接真实 DashScope Embedding API，**MongoDB** 存储    |

## `run_lifecycle_sql.py` — 全生命周期测试（SQL / PostgreSQL）

### 前置条件

1. 确保 PostgreSQL 运行且目标数据库存在：

   ```bash
   createdb isobase
   ```

2. 确保 DashScope API key 已配置在 `.env` 文件中。

### 运行

```bash
# 在仓库根目录执行：
python -m test.knowledge.live.run_lifecycle_sql
```

### 测试流程

每一步都会暂停，按 **Enter** 继续，**Ctrl-C** 可随时中断。

| Step | 操作                                                    | 可在此步用 psql 查看的表                                          |
| ---- | ------------------------------------------------------- | ----------------------------------------------------------------- |
| 1    | 创建两个知识库（一个放文档，一个空的）                  | `knowledge_bases`                                                 |
| 2    | `list_knowledge_bases()` 列出所有                       | `knowledge_bases`                                                 |
| 3    | 索引 3 篇文档（分块 + 嵌入）                            | `knowledge_documents`、`knowledge_chunks`、`knowledge_embeddings` |
| 4    | `list_documents` + `get_document` 逐条查看              | `knowledge_documents`                                             |
| 5    | 3 次检索 + 空库检索 + `retrieve_as_context`             | 无新写入                                                          |
| 6    | `delete_document` 删除第 1 篇文档 + 验证 chunk 级联清除 | 观察 `knowledge_chunks` / `knowledge_embeddings` 中对应行消失     |
| 7    | `delete_knowledge_base` 删除空库 + 验证                 | 观察 `knowledge_bases` 中对应行消失                               |
| 8    | 清理剩余 kb1，库清空                                    | 所有表为空                                                        |

### 在暂停期间用 psql 查看数据

```bash
# 查看所有知识库
psql "postgresql://postgres:postgres@localhost:5432/isobase" \
  -c "SELECT id, name, description FROM knowledge_bases;"

# 查看所有文档
psql "postgresql://postgres:postgres@localhost:5432/isobase" \
  -c "SELECT id, title, knowledge_base_id FROM knowledge_documents;"

# 查看所有 chunk 及其 embedding 长度
psql "postgresql://postgres:postgres@localhost:5432/isobase" \
  -c "SELECT c.id, c.content, c.document_id, length(e.embedding) AS emb_len
      FROM knowledge_chunks c
      JOIN knowledge_embeddings e ON e.chunk_id = c.id;"

# 统计各表行数
psql "postgresql://postgres:postgres@localhost:5432/isobase" \
  -c "SELECT 'knowledge_bases' AS tbl, count(*) FROM knowledge_bases
      UNION ALL SELECT 'knowledge_documents', count(*) FROM knowledge_documents
      UNION ALL SELECT 'knowledge_chunks', count(*) FROM knowledge_chunks
      UNION ALL SELECT 'knowledge_embeddings', count(*) FROM knowledge_embeddings;"
```

### 测试覆盖的 API

| API                     | Step |
| ----------------------- | ---- |
| `create_knowledge_base` | 1    |
| `list_knowledge_bases`  | 2    |
| `index_text`            | 3    |
| `list_documents`        | 4    |
| `get_document`          | 4    |
| `retrieve`              | 5    |
| `retrieve_as_context`   | 5    |
| `delete_document`       | 6    |
| `delete_knowledge_base` | 7    |

## `run_lifecycle_mongo.py` — 全生命周期测试（MongoDB）

### 前置条件

1. 确保 MongoDB 运行在 `localhost:27017`。

2. 确保 DashScope API key 已配置在 `.env` 文件中。

### 运行

```bash
# 在仓库根目录执行：
python -m test.knowledge.live.run_lifecycle_mongo
```

### 测试流程

与 `run_lifecycle_sql.py` 完全一致（8 个 Step），但使用 MongoDB 作为存储后端。

| Step | 操作                                             | 可在此步用 mongosh 查看的集合                                     |
| ---- | ------------------------------------------------ | ----------------------------------------------------------------- |
| 1    | 创建两个知识库（一个放文档，一个空的）           | `knowledge_bases`                                                 |
| 2    | `list_knowledge_bases()` 列出所有                | `knowledge_bases`                                                 |
| 3    | 索引 1 篇文档（分块 + 嵌入）                     | `knowledge_documents`、`knowledge_chunks`、`knowledge_embeddings` |
| 4    | `list_documents` + `get_document` 逐条查看       | `knowledge_documents`                                             |
| 5    | 6 次检索 + 空库检索 + `retrieve_as_context`      | 无新写入                                                          |
| 6    | `delete_document` 删除文档 + 验证 chunk 级联清除 | 观察 `knowledge_chunks` / `knowledge_embeddings` 中对应文档消失   |
| 7    | `delete_knowledge_base` 删除空库 + 验证          | 观察 `knowledge_bases` 中对应文档消失                             |
| 8    | 清理剩余 kb1，库清空                             | 所有集合为空                                                      |

### 在暂停期间用 mongosh 查看数据

```bash
# 查看所有知识库
mongosh "mongodb://localhost:27017/isobase" \
  --eval "db.knowledge_bases.find().pretty()"

# 查看所有文档
mongosh "mongodb://localhost:27017/isobase" \
  --eval "db.knowledge_documents.find().pretty()"

# 查看所有 chunk
mongosh "mongodb://localhost:27017/isobase" \
  --eval "db.knowledge_chunks.find().pretty()"

# 统计各集合文档数
mongosh "mongodb://localhost:27017/isobase" \
  --eval "db.knowledge_bases.countDocuments()" \
  --eval "db.knowledge_documents.countDocuments()" \
  --eval "db.knowledge_chunks.countDocuments()" \
  --eval "db.knowledge_embeddings.countDocuments()"
```

## `run_knowledge_basic.py` — Embedding API 冒烟测试

### 前置条件

1. 复制 `.env.example` 为 `.env` 并填入真实 API key：

   ```bash
   cp test/knowledge/live/.env.example test/knowledge/live/.env
   ```

2. 编辑 `.env`，填入至少一个 provider 的 API key。

### 运行

```bash
python -m test.knowledge.live.run_knowledge_basic           # 所有已配置的 provider
python -m test.knowledge.live.run_knowledge_basic openai    # 仅 OpenAI
python -m test.knowledge.live.run_knowledge_basic dashscope # 仅 DashScope
python -m test.knowledge.live.run_knowledge_basic aimlapi   # 仅 AIMLAPI
```
