# bd-chat-2020
2020语言与智能技术竞赛：面向推荐的对话任务      
第二名 强行跳大 团队

# 介绍
 * 最终只使用了集成Goal预测、文本回复的单向注意力Bert模型。此项目还附带了一些抛弃的尝试方案，包含了基础的阅读理解模型（model/model_rc.py）、分类模型（model/model_context.py,model/model_goal.py）、QA召回（检索）模型（model/model_recall.py）的实现。    
 * 训练的数据存放位置和输出位置可以参考 cfg.py。训练的顺序是先运行 data_deal/pre_trans.py，再运行train/train_bert_lm.py。
 * 预测test集的代码是code/predict/predict_lm.py，人工评估阶段使用的是code/predict/predict_final.py里的预测

# 数据
https://aistudio.baidu.com/aistudio/competition/detail/48
里面的推荐对话任务数据即是，格式一致

# 依赖
```text
keras==2.3.1
tensorflow-gpu==1.14.0
easydict
nltk==3.5
```

# 文件介绍：
 * conversation_client.py: 类似官方的那种client
 * test_client.py: 交互式的client
 * conversation_server.py: 类似官方的服务端。

# 参考
 * bert4keras：https://github.com/bojone/bert4keras

# 注意
 * 不要覆盖 data/roberta 里的 vocab.txt 和 config
