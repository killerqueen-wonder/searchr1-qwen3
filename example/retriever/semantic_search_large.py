# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 语义相似文本检索
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
import sys

sys.path.append('..')
import os
from text2vec import SentenceModel, cos_sim, semantic_search, BM25
import torch
import time
from datetime import datetime, timedelta


import json

def read_from_jsonl(filename="D:/学习资料/DS 课程/法律大模型/data/法律条文/law.jsonl"):
    """
    从.jsonl文件中读取所有内容，每行作为一个字典对象，组成列表并返回。
    
    参数:
        filename (str): .jsonl文件路径
    
    返回:
        list: 包含所有JSON对象的列表
    """
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 确保不是空行
                    try:
                        content = json.loads(line)
                        data.append(content)
                    except json.JSONDecodeError as e:
                        print(f"跳过无法解析的行: {line[:50]}... 错误: {e}")
    except FileNotFoundError:
        print(f"文件未找到: {filename}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    
    return data

def format_time_delta(seconds):
    """将秒数转换为小时:分钟:秒的格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_embedding_filename(jsonl_filename):
    """
    根据jsonl文件名生成对应的embedding文件名
    例如: law_pure.jsonl -> law_pure_embeddings.pt
    """
    base_name = os.path.splitext(os.path.basename(jsonl_filename))[0]
    dir_name = os.path.dirname(jsonl_filename)
    embedding_filename = os.path.join(dir_name, f"{base_name}_embeddings.pt")
    return embedding_filename

def save_corpus_embeddings(embedder, filename):
    """
    加载语料库并计算句子嵌入，然后保存为与语料库同名的 .pt 文件。
    
    参数:
        embedder: SentenceModel 实例
        filename: 原始语料库路径 (.jsonl)
    """
    print(f"[INFO] 开始加载语料库: {filename}")
    corpus = read_from_jsonl(filename)
    
    # 生成对应的embedding文件名
    embedding_file = get_embedding_filename(filename)
    
    print("[INFO] 正在计算语义向量...")
    if torch.cuda.is_available():
        print(f"[INFO] 使用 GPU: {torch.cuda.get_device_name(0)}")
        pool = embedder.start_multi_process_pool()
        corpus_embeddings = embedder.encode_multi_process(corpus, pool, normalize_embeddings=True)
        embedder.stop_multi_process_pool(pool)
    else:
        print("[INFO] 使用 CPU 计算")
        corpus_embeddings = embedder.encode(corpus)

    print(f"[INFO] 保存 embedding 到: {embedding_file}")
    torch.save(corpus_embeddings, embedding_file)
    print(f"[INFO] 已保存 shape={corpus_embeddings.shape}")
    return embedding_file


def process_retriever(
    queries=None,
    output_path="output_results.txt", 
    filename="/mnt/nvme1n1/legal_LLM/dataset/law/law_pure.jsonl", 
    embedder=SentenceModel("shibing624/text2vec-base-chinese-paraphrase")
):
    """
    处理检索任务，自动检测并使用embedding文件
    
    参数:
        embedder: SentenceModel 实例
        filename: 原始语料库路径 (.jsonl)
        queries: 查询列表
        output_path: 输出结果文件路径
    """
    t1 = time.time()
    
    # 生成对应的embedding文件名
    embedding_file = get_embedding_filename(filename)
    
    # 检查embedding文件是否存在
    if os.path.exists(embedding_file):
        print(f"[INFO] 发现预计算的embedding文件: {embedding_file}")
        print("[INFO] 从本地 embedding 文件加载...")
        corpus_embeddings = torch.load(embedding_file)
        corpus = read_from_jsonl(filename)
    else:
        print(f"[INFO] 未找到预计算的embedding文件: {embedding_file}")
        print("[INFO] 开始计算并保存embedding...")
        # 计算并保存embedding
        save_corpus_embeddings(embedder, filename)
        # 重新加载
        corpus_embeddings = torch.load(embedding_file)
        corpus = read_from_jsonl(filename)

    t2 = time.time()
    print(f"[DEBUG] embedding 运行时间: {format_time_delta(t2 - t1)}")

    top_k = min(5, len(corpus))

    # 初始化 BM25 模型
    bm25_model = BM25(corpus=corpus)

    def cos_sim_retrieval(query):
        query_embedding = embedder.encode(query)
        cos_scores = cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = []
        scores_list = cos_scores.tolist()
        indexed_scores = [(i, score) for i, score in enumerate(scores_list)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        for i in range(min(top_k, len(indexed_scores))):
            idx, score = indexed_scores[i]
            top_results.append((corpus[idx], score))
        return top_results

    def semantic_search_retrieval(query):
        query_embedding = embedder.encode(query)
        hits = semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
        results = []
        for hit in hits:
            results.append((corpus[hit['corpus_id']], hit['score']))
        return results

    def bm25_retrieval(query):
        bm25_scores = bm25_model.get_scores(query, top_k=top_k)
        results = []
        for text, score in bm25_scores:
            results.append((text, score))
        return results

    # 重定向输出到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for query in queries:
            f.write("\n" + "="*50 + "\n")
            f.write(f"Query: {query}\n")
            f.write("="*50 + "\n")

            f.write("\n--- 使用余弦相似度 (Cosine Similarity) ---\n")
            cos_results = cos_sim_retrieval(query)
            for text, score in cos_results:
                f.write(f"{text} (Score: {score:.4f})\n")

            f.write("\n--- 使用 semantic_search ---\n")
            sem_results = semantic_search_retrieval(query)
            for text, score in sem_results:
                f.write(f"{text} (Score: {score:.4f})\n")

            f.write("\n--- 使用 BM25 ---\n")
            bm25_results = bm25_retrieval(query)
            for text, score in bm25_results:
                f.write(f"{text} (Score: {score:.4f})\n")

    print(f"结果已保存到 {output_path}")
    t3 = time.time()
    print(f"[DEBUG] 检索运行时间: {format_time_delta(t3 - t2)}")

# 调用函数
if __name__ == "__main__":
    start_time = time.time()

    model="shibing624/text2vec-base-chinese-paraphrase"
    embedder = SentenceModel(model)
    filename="/mnt/nvme1n1/legal_LLM/dataset/law/law_pure.jsonl"
    output_path="/mnt/nvme1n1/legal_LLM/dataset/law/output_results.txt"
    # Query sentences:
    queries = [
        '2025年中华人民共和国宪法第三十四条',
        '抢劫罪',
        '2003年故意伤人罪',
        "\"李开祥侵犯公民个人信息刑事附带民事公益诉讼案 刑事/侵犯公民个人信息/刑事附带民事公益诉讼/人脸识别/人脸信息\"\n【裁判要点】\n  使用人脸识别技术处理的人脸信息以及基于人脸识别技术生成的人脸信息均具有高度的可识别性，能够单独或者与其他信息结合识别特定自然人身份或者反映特定自然人活动情况，属于刑法规定的公民个人信息。行为人未经公民本人同意，未具备获得法律、相关部门授权等个人信息保护法规定的处理个人信息的合法事由，利用软件程序等方式窃取或者以其他方法非法获取上述信息，情节严重的，应依照《最高人民法院、最高人民检察院关于办理侵犯公民个人信息刑事案件适用法律若干问题的解释》第五条第一款第四项等规定定罪处罚。\n  【相关法条】\n  《中华人民共和国刑法》第253条之一\n  【基本案情】\n  2020年6月至9月间，被告人李开祥制作一款具有非法窃取安装者相册照片功能的手机“黑客软件”，打包成安卓手机端的“APK安装包”，发布于暗网“茶马古道”论坛售卖，并伪装成“颜值检测”软件发布于“芥子论坛”(后更名为“快猫社区”)提供访客免费下载。用户下载安装“颜值检测”软件使用时，“颜值检测”软件会自动在后台获取手机相册里的照片，并自动上传到被告人搭建的腾讯云服务器后台，从而窃取安装者相册照片共计1751张，其中部分照片含有人脸信息、自然人姓名、身份号码、联系方式、家庭住址等公民个人信息100余条。\n  2020年9月，被告人李开祥在暗网“茶马古道”论坛看到“黑客资料”帖子，后用其此前在暗网售卖“APK安装包”部分所得购买、下载标题为“社工库资料”数据转存于“MEGA”网盘，经其本人查看，确认含有个人真实信息。2021年2月，被告人李开祥明知“社工库资料”中含有户籍信息、QQ账号注册信息、京东账号注册信息、车主信息、借贷信息等，仍将网盘链接分享至其担任管理员的“翠湖庄园业主交流”QQ群，提供给群成员免费下载。经鉴定，“社工库资料”经去除无效数据并进行合并去重后，包含各类公民个人信息共计8100万余条。\n  上海市奉贤区人民检察院以社会公共利益受到损害为由，向上海市奉贤区人民法院提起刑事附带民事公益诉讼。\n  被告人李开祥对起诉指控的基本犯罪事实及定性无异议，且自愿认罪认罚。\n  辩护人提出被告人李开祥系初犯，到案后如实供述所犯罪行，且自愿认罪认罚等辩护意见，建议对被告人李开祥从轻处罚，请求法庭对其适用缓刑。辩护人另辩称，检察机关未对涉案8100万余条数据信息的真实性核实确认。\n  【裁判结果】\n  上海市奉贤区人民法院于2021年8月23日以（2021）沪0120刑初828号刑事判决，认定被告人李开祥犯侵犯公民个人信息罪，判处有期徒刑三年，宣告缓刑三年，并处罚金人民币一万元；扣押在案的犯罪工具予以没收。判决李开祥在国家级新闻媒体上对其侵犯公民个人信息的行为公开赔礼道歉、删除“颜值检测”软件及相关代码、删除腾讯云网盘上存储的涉案照片、删除存储在“MＥＧＡ”网盘上相关公民个人信息，并注销侵权所用ＱＱ号码。一审判决后，没有抗诉、上诉，判决现已生效。\n  【裁判理由】\n  法院生效裁判认为：本案争议焦点为利用涉案“颜值检测”软件窃取的“人脸信息”是否属于刑法规制范畴的“公民个人信息”。法院经审理认为，“人脸信息”属于刑法第二百五十三条之一规定的公民个人信息，利用“颜值检测”黑客软件窃取软件使用者“人脸信息”等公民个人信息的行为，属于刑法中“窃取或者以其他方法非法获取公民个人信息”的行为，依法应予惩处。主要理由如下：第一，“人脸信息”与其他明确列举的个人信息种类均具有明显的“可识别性”特征。《最高人民法院、最高人民检察院关于办理侵犯公民个人信息刑事案件适用法律若干问题的解释》（以下简称《解释》）中列举了公民个人信息种类，虽未对“人脸信息”单独列举，但允许依法在列举之外认定其他形式的个人信息。《解释》中对公民个人信息的定义及明确列举与民法典等法律规定中有关公民个人信息的认定标准一致，即将“可识别性”作为个人信息的认定标准，强调信息与信息主体之间被直接或间接识别出来的可能性。“人脸信息”属于生物识别信息，其具有不可更改性和唯一性，人脸与自然人个体一一对应，无需结合其他信息即可直接识别到特定自然人身份，具有极高的“可识别性”。第二，将“人脸信息”认定为公民个人信息遵循了法秩序统一性原理。民法等前置法将“人脸信息”作为公民个人信息予以保护。民法典第一千零三十四条规定了个人信息的定义和具体种类，个人信息保护法进一步将“人脸信息”纳入个人信息的保护范畴，侵犯“人脸信息”的行为构成侵犯自然人人格权益等侵权行为的，须承担相应的民事责任或行政、刑事责任。第三，采用“颜值检测”黑客软件窃取“人脸信息”具有较大的社会危害性和刑事可罚性。因“人脸信息”是识别特定个人的敏感信息，亦是社交属性较强、采集方便的个人信息，极易被他人直接利用或制作合成，从而破解人脸识别验证程序，引发侵害隐私权、名誉权等违法行为，甚至盗窃、诈骗等犯罪行为，社会危害较大。被告人李开祥操纵黑客软件伪装的“颜值检测”软件窃取用户自拍照片和手机相册中的存储照片，利用了互联网平台的开放性，以不特定公众为目标，手段隐蔽、欺骗性强、窃取面广，具有明显的社会危害性，需用刑法加以规制。\n  关于辩护人提出本案公民个人信息数量认定依据不足的辩护意见，法院经审理认为，公安机关侦查过程中采用了抽样验证的方法，随机挑选部分个人信息进行核实，能够确认涉案个人信息的真实性，被告人、辩护人亦未提出涉案信息不真实的线索或证据。司法鉴定机构通过去除无效信息，并采用合并去重的方法进行鉴定，检出有效个人信息8100万余条，公诉机关指控的公民个人信息数量客观、真实，且符合《解释》中确立的对批量公民个人信息具体数量的认定规则，故对辩护人的辩护意见不予采纳。\n  综上，被告人李开祥违反国家有关规定，非法获取并向他人提供公民个人信息，情节特别严重，其行为已构成侵犯公民个人信息罪。被告人李开祥到案后能如实供述自己的罪行，依法可以从轻处罚，且自愿认罪认罚，依法可以从宽处理。李开祥非法获取并向他人提供公民个人信息的侵权行为，侵害了众多公民个人信息安全，损害社会公共利益，应当承担相应的民事责任。故依法作出上述判决。\n  （生效裁判审判人员：李晓杰、管玉洁、高晔涛）        \n",
        '''在2021年，根据《宪法》和法律的规定，关于我国行政区 划变更的法律程序，下列选项正确的是？
    A.	特别行政区的设立和建置由全国人大审批 决定
    B.	甲省行政区域界线的变更，应由全国人大审 议决定
    C.	全国人大决定乙直辖市的设立，国务院决定乙直辖市的区域划分
    D.	丙市部分行政区域界线的变更，由国务院 审批
    ''',
    '''2022年，第一百二十条　除本法第一百一十九条所列情形外，收货人于货物到达目的地点，并在缴付应付款项和履行航空货运单上所列运输条件后，有权要求承运人移交航空货运单并交付货物。
    　　除另有约定外，承运人应当在货物到达后立即通知收货人。
    　　承运人承认货物已经遗失，或者货物在应当到达之日起七日后仍未到达的，收货人有权向承运人行使航空货物运输合同所赋予的权利。''',
        
    ]
    process_retriever(queries,output_path, filename, embedder )

    end_time = time.time()
    print(f"程序运行时间: {format_time_delta(end_time - start_time)}")