import gensim
import numpy
from gensim import corpora, models, matutils
from gensim.models import TfidfModel


class VSM:
    def __init__(self):
        self.tfidf_model: TfidfModel = None

    def build_model(self, docs_tokens):
        print("Building VSM model...")
        dictionary = corpora.Dictionary(docs_tokens)
        corpus = [dictionary.doc2bow(x) for x in docs_tokens]
        self.tfidf_model = models.TfidfModel(corpus, id2word=dictionary)
        print("Finish building VSM model")

    def _get_doc_similarity(self, doc1_tk, doc2_tk):
        doc1_vec = self.tfidf_model[self.tfidf_model.id2word.doc2bow(doc1_tk)]
        doc2_vec = self.tfidf_model[self.tfidf_model.id2word.doc2bow(doc2_tk)]
        return matutils.cossim(doc1_vec, doc2_vec)

    def get_link_scores(self, source, target):
        s_tokens = source['tokens'].split()
        t_tokens = target['tokens'].split()
        score = self._get_doc_similarity(s_tokens, t_tokens)
        return score


class LDA:
    def __init__(self):
        self.ldamodel = None

    def build_model(self, docs_tokens, num_topics=200, passes=1000):
        dictionary = corpora.Dictionary(docs_tokens)
        corpus = [dictionary.doc2bow(x) for x in docs_tokens]
        self.ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary,
                                                        passes=passes, alpha='auto',
                                                        random_state=numpy.random.RandomState(1))

    def get_topic_distrb(self, doc):
        bow_doc = self.ldamodel.id2word.doc2bow(doc)
        return self.ldamodel.get_document_topics(bow_doc)

    def get_link_scores(self, source, target):
        """
        :param doc1_tk: Preprocessed documents as tokens
        :param doc2_tk: Preprocessed documents as tokens
        :return:
        """
        doc1_tk = source['tokens'].split()
        doc2_tk = target['tokens'].split()
        dis1 = self.get_topic_distrb(doc1_tk)
        dis2 = self.get_topic_distrb(doc2_tk)
        # return 1 - matutils.hellinger(dis1, dis2)
        return matutils.cossim(dis1, dis2)


class LSI:
    def __init__(self):
        self.lsi = None

    def build_model(self, docs_tokens, num_topics=200):
        dictionary = corpora.Dictionary(docs_tokens)
        corpus = [dictionary.doc2bow(x) for x in docs_tokens]
        self.lsi = gensim.models.LsiModel(corpus, num_topics=num_topics, id2word=dictionary)

    def get_topic_distrb(self, doc):
        bow_doc = self.lsi.id2word.doc2bow(doc)
        return self.lsi[bow_doc]

    def get_link_scores(self,  source, target):
        doc1_tk = source['tokens'].split()
        doc2_tk = target['tokens'].split()
        dis1 = self.get_topic_distrb(doc1_tk)
        dis2 = self.get_topic_distrb(doc2_tk)
        return matutils.cossim(dis1, dis2)



class JaccardSimilarity:
    def __init__(self):
        self.documents = []
    
    def build_model(self, docs_tokens):
        print("Building Jaccard model...")
        self.documents = docs_tokens
        print("Finish building Jaccard model")
        
    def _get_doc_similarity(self, doc1_tk, doc2_tk):
        # 将文档转换为集合
        set1 = set(doc1_tk)
        set2 = set(doc2_tk)
        
        # 计算交集和并集
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        # 避免除零错误
        if union == 0:
            return 0
            
        return intersection / union
        
    def get_link_scores(self, source, target):
        s_tokens = source['tokens'].split()
        t_tokens = target['tokens'].split()
        score = self._get_doc_similarity(s_tokens, t_tokens)
        return score

class LevenshteinDistance:
    def __init__(self):
        self.documents = []
    
    def build_model(self, docs_tokens):
        print("Building Levenshtein model...")
        self.documents = docs_tokens
        print("Finish building Levenshtein model")
    
    def _get_doc_similarity(self, doc1_tk, doc2_tk):
        # 将token列表转换为字符串，以空格分隔
        str1 = ' '.join(doc1_tk)
        str2 = ' '.join(doc2_tk)
        
        # 获取字符串长度
        m = len(str1)
        n = len(str2)
        
        # 创建动态规划矩阵
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化第一行和第一列
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        # 填充矩阵
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # 删除
                        dp[i][j-1] + 1,    # 插入
                        dp[i-1][j-1] + 1   # 替换
                    )
        
        # 将编辑距离转换为相似度得分（归一化）
        max_len = max(m, n)
        if max_len == 0:
            return 1.0
        return 1 - (dp[m][n] / max_len)
        
    def get_link_scores(self, source, target):
        s_tokens = source['tokens'].split()
        t_tokens = target['tokens'].split()
        score = self._get_doc_similarity(s_tokens, t_tokens)
        return score
    

import numpy as np
import math
from typing import List, Dict

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.avgdl = sum([len(doc) for doc in corpus]) / self.corpus_size
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        
        # 计算文档频率和文档长度
        for doc in corpus:
            self.doc_len.append(len(doc))
            freq = {}
            for word in doc:
                freq[word] = freq.get(word, 0) + 1
            self.doc_freqs.append(freq)
            
            # 更新词的文档频率
            for word, _ in freq.items():
                self.idf[word] = self.idf.get(word, 0) + 1

        # 计算IDF值
        for word, freq in self.idf.items():
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_score(self, query: List[str], doc_id: int) -> float:
        score = 0.0
        doc_freq = self.doc_freqs[doc_id]
        doc_len = self.doc_len[doc_id]
        
        for word in query:
            if word not in doc_freq:
                continue
                
            score += (self.idf.get(word, 0) * doc_freq[word] * (self.k1 + 1) /
                     (doc_freq[word] + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_scores(self, query: List[str]) -> List[float]:
        scores = []
        for doc_id in range(self.corpus_size):
            score = self.get_score(query, doc_id)
            scores.append(score)
        return scores

class BM25Similarity:
    def __init__(self):
        self.bm25_model = None
        self.corpus = None

    def build_model(self, docs_tokens):
        print("Building BM25 model...")
        self.corpus = docs_tokens
        self.bm25_model = BM25(docs_tokens)
        print("Finish building BM25 model")

    def _calculate_bm25_score(self, query_tokens, doc_tokens):
        """计算单向的BM25分数"""
        score = 0.0
        doc_len = len(doc_tokens)
        
        # 计算文档中词频
        doc_freq = {}
        for word in doc_tokens:
            doc_freq[word] = doc_freq.get(word, 0) + 1
            
        for word in query_tokens:
            if word not in doc_freq:
                continue
                
            # 使用已有的IDF，如果词不在训练集中，则给予一个默认值
            idf = self.bm25_model.idf.get(word, math.log(self.bm25_model.corpus_size + 0.5))
            
            score += (idf * doc_freq[word] * (self.bm25_model.k1 + 1) /
                     (doc_freq[word] + self.bm25_model.k1 * 
                      (1 - self.bm25_model.b + self.bm25_model.b * doc_len / self.bm25_model.avgdl)))
        
        return score

    def _get_doc_similarity(self, doc1_tk, doc2_tk):
        # 双向计算BM25分数
        score1 = self._calculate_bm25_score(doc1_tk, doc2_tk)
        score2 = self._calculate_bm25_score(doc2_tk, doc1_tk)
        
        # 取平均作为最终相似度
        return (score1 + score2) / 2

    def get_link_scores(self, source, target):
        s_tokens = source['tokens'].split()
        t_tokens = target['tokens'].split()
        score = self._get_doc_similarity(s_tokens, t_tokens)
        return score





import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import numpy as np




class Word2VecSimilarity:
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.model = None
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count

    def build_model(self, docs_tokens):
        print("Building Word2Vec model...")
        self.model = Word2Vec(
            sentences=docs_tokens,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4
        )
        print("Finish building Word2Vec model")

    def _get_doc_vector(self, tokens):
        vectors = []
        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.model.wv[token])
        if not vectors:
            return np.zeros(self.vector_size)
        return np.mean(vectors, axis=0)

    def _get_doc_similarity(self, doc1_tk, doc2_tk):
        # 计算文档向量
        vec1 = self._get_doc_vector(doc1_tk)
        vec2 = self._get_doc_vector(doc2_tk)
        
        # 计算余弦相似度
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def get_link_scores(self, source, target):
        s_tokens = source['tokens'].split()
        t_tokens = target['tokens'].split()
        score = self._get_doc_similarity(s_tokens, t_tokens)
        return score

class GloVeSimilarity:
    def __init__(self, pretrained_path=None):
        self.model = None
        self.vector_size = None
        self.pretrained_path = pretrained_path

    def build_model(self, docs_tokens):
        print("Loading GloVe model...")
        if self.pretrained_path:
            # 加载预训练的GloVe模型
            self.model = KeyedVectors.load_word2vec_format(
                self.pretrained_path, 
                binary=False
            )
            self.vector_size = self.model.vector_size
        else:
            raise ValueError("GloVe requires pre-trained vectors")
        print("Finish loading GloVe model")

    def _get_doc_vector(self, tokens):
        vectors = []
        for token in tokens:
            if token in self.model:
                vectors.append(self.model[token])
        if not vectors:
            return np.zeros(self.vector_size)
        return np.mean(vectors, axis=0)

    def _get_doc_similarity(self, doc1_tk, doc2_tk):
        vec1 = self._get_doc_vector(doc1_tk)
        vec2 = self._get_doc_vector(doc2_tk)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def get_link_scores(self, source, target):
        s_tokens = source['tokens'].split()
        t_tokens = target['tokens'].split()
        score = self._get_doc_similarity(s_tokens, t_tokens)
        return score
    


import torch
from transformers import (
    BertTokenizer, 
    BertModel,
    RobertaTokenizer, 
    RobertaModel,
    AlbertTokenizer,
    AlbertModel,
    DistilBertTokenizer,
    DistilBertModel
)
import numpy as np
from typing import List, Dict, Optional

class BERTBaseSimilarity:
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 512,
        use_cuda: bool = True
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.max_length = max_length
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def build_model(self, docs_tokens: List[List[str]]):
        """初始化模型，这里docs_tokens参数保持接口一致，但BERT类并不需要训练"""
        print(f"Loading {self.model_name} model...")
        self._init_model()
        print(f"Finish loading {self.model_name} model")

    def _init_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def _get_embedding(self, text: str) -> np.ndarray:
        # 对文本进行编码
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(self.device)

        # 获取BERT输出
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 使用[CLS]标记的输出作为文档表示
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings[0]

    def _get_doc_similarity(self, doc1_tk: List[str], doc2_tk: List[str]) -> float:
        # 将token列表转换为文本
        text1 = ' '.join(doc1_tk)
        text2 = ' '.join(doc2_tk)
        
        # 获取文档嵌入
        embed1 = self._get_embedding(text1)
        embed2 = self._get_embedding(text2)
        
        # 计算余弦相似度
        cos_sim = np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
        return float(cos_sim)

    def get_link_scores(self, source: Dict[str, str], target: Dict[str, str]) -> float:
        s_tokens = source['tokens'].split()
        t_tokens = target['tokens'].split()
        score = self._get_doc_similarity(s_tokens, t_tokens)
        return score

class BERTSimilarity(BERTBaseSimilarity):
    def _init_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

class RoBERTaSimilarity(BERTBaseSimilarity):
    def _init_model(self):
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

class AlBERTSimilarity(BERTBaseSimilarity):
    def _init_model(self):
        self.tokenizer = AlbertTokenizer.from_pretrained(self.model_name)
        self.model = AlbertModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

class DistilBERTSimilarity(BERTBaseSimilarity):
    def _init_model(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

# 支持句子级别的BERT相似度计算
class SentenceBERTSimilarity(BERTBaseSimilarity):
    def __init__(
        self,
        model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        max_length: int = 512,
        use_cuda: bool = True,
        pooling_strategy: str = 'mean'
    ):
        super().__init__(model_name, max_length, use_cuda)
        self.pooling_strategy = pooling_strategy

    def _get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 不同的池化策略
        if self.pooling_strategy == 'cls':
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif self.pooling_strategy == 'mean':
            attention_mask = inputs['attention_mask']
            masked_output = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            embeddings = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        return embeddings.cpu().numpy()[0]
    





