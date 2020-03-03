## 运行前执行(在fudan局域网下)
```bash
cd /HotSpotServer/component/event_analyze
wget http://10.132.141.239:9000/file/data_path.tar
wget http://10.132.141.239:9000/file/data_path_save.tar
wget http://10.132.141.239:9000/file/ltp_data_v3.4.0.zip

tar -x data_path.tar
tar -x data_path_save.tar
unzip ltp_data_v3.4.0.zip

# 切换到pyhanlp安装包路径static目录下
cd venv/lib/python3.6/site-packages/pyhanlp/static/
wget http://10.132.141.239:9000/file/data-for-1.7.3.zip

```


## 各个文件
- ner_processor.py

    对句子进行命名体提取，提取Person，Location和Organization，并统计权重
    
- parse_processor.py

    对句子进行依存句法分析，提取句子中所有对主语，谓语动词和宾语，并统计权重
    
- 