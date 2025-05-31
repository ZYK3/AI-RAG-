# 智能诗歌问答系统

一个基于 LLamaIndex 和 OpenAI 的智能诗歌问答系统，以李清照的视角回答诗歌相关问题。

## 功能特点

- 支持 PDF 格式诗歌文档的批量导入
- 智能语义分割和文本清理
- 混合检索策略(向量检索 + BM25)
- 流式输出回答
- 角色扮演(李清照)风格回答
- Web 界面交互

## 技术栈

- Python 3.8+
- LlamaIndex
- OpenAI API
- ChromaDB
- Gradio

## 环境配置

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 配置环境变量
创建 `.env` 文件，添加以下配置：
```env
OPENAI_API_KEY=your_api_key
DEFAULT_MODEL_NAME=your_model_name
```

## 项目结构

```
.
├── OnlyOne/             # PDF文档目录
├── data/                # 向量数据库存储
│   └── chroma_db/
├── multi_version/       
│   └── second_version.py    # 主程序
├── requirements.txt     # 依赖包
└── README.md
```

## 核心功能实现

1. **文档处理**
   - PDF文档加载
   - 文本清理
   - 智能分块

2. **知识库构建**
   - ChromaDB向量存储
   - 混合检索策略
   - 语义理解

3. **问答系统**
   - 流式输出
   - 角色定制
   - 上下文管理

## 使用方法

1. 准备数据
```bash
# 将PDF文件放入OnlyOne目录
```

2. 运行程序
```bash
python multi_version/second_version.py
```

3. 访问Web界面
```
默认地址: http://127.0.0.1:7860
```

## 系统特色

- **混合检索**: 结合向量检索(60%)和BM25检索(40%)提高准确率
- **语义分割**: 使用语义感知的文本分割保持文本完整性
- **个性化回答**: 以李清照视角回答，增加趣味性
- **实时反馈**: 流式输出提供即时响应

## 注意事项

- 确保 PDF 文件编码正确
- 需要稳定的网络连接访问 OpenAI API
- 建议使用较大的语言模型以获得更好效果

## 未来计划

- [ ] 添加更多角色选择
- [ ] 优化检索策略
- [ ] 支持更多文档格式
- [ ] 添加文档预处理选项

## License

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request
