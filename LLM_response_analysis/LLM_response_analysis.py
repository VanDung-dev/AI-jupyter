# PHẦN 1: Imports
import os
import re
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import plotly.express as px

# Optional: sentence-transformers for good embeddings. Nếu không có, sẽ fallback sang TF-IDF
try:
    from sentence_transformers import SentenceTransformer
    USE_SENTENCE_TRANSFORMER = True
except Exception:
    USE_SENTENCE_TRANSFORMER = False

# PHẦN 2: Các hàm tiện ích

def load_logs(path: str) -> pd.DataFrame:
    """Load CSV/JSON logs. Trả về DataFrame chuẩn hóa cột: id, prompt, response, timestamp"""
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.json') or path.endswith('.jsonl'):
        df = pd.read_json(path, lines=path.endswith('.jsonl'))
    else:
        raise ValueError('Chỉ hỗ trợ CSV hoặc JSON(.jsonl)')

    # chuẩn hóa tên cột phổ biến
    mapping = {}
    for c in df.columns:
        low = c.lower()
        if 'prompt' in low or 'question' in low or 'input' in low:
            mapping[c] = 'prompt'
        if 'response' in low or 'answer' in low or 'reply' in low:
            mapping[c] = 'response'
        if 'time' in low or 'timestamp' in low:
            mapping[c] = 'timestamp'
        if 'id' == low or low.endswith('_id'):
            mapping[c] = 'id'

    df = df.rename(columns=mapping)
    if 'prompt' not in df.columns or 'response' not in df.columns:
        raise ValueError('File log phải có cột prompt và response (hoặc tên tương tự)')

    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            pass
    else:
        df['timestamp'] = pd.Timestamp.now()

    if 'id' not in df.columns:
        df['id'] = range(len(df))

    return df


def normalize_text(s: str) -> str:
    s = s or ''
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# PHẦN 3: Embeddings/Vectorization

class Embedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.use_st = USE_SENTENCE_TRANSFORMER
        if self.use_st:
            try:
                print('Loading SentenceTransformer:', model_name)
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print('Không load được sentence-transformer, fallback TF-IDF.', e)
                self.use_st = False
                self.model = None
        if not self.use_st:
            self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
            self.model = None

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        texts = [normalize_text(t) for t in texts]
        if self.use_st and self.model is not None:
            embs = np.array(self.model.encode(texts, show_progress_bar=True))
            return embs
        else:
            X = self.vectorizer.fit_transform(texts)
            return X.toarray()

    def transform(self, texts: List[str]) -> np.ndarray:
        texts = [normalize_text(t) for t in texts]
        if self.use_st and self.model is not None:
            return np.array(self.model.encode(texts, show_progress_bar=False))
        else:
            X = self.vectorizer.transform(texts)
            return X.toarray()


# PHẦN 4: Metrics - similarity, hallucination proxies, structure parsing

def pairwise_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cosine_similarity(a, b)


def response_prompt_similarity(emb_prompts: np.ndarray, emb_responses: np.ndarray) -> np.ndarray:
    # cosine similarity giữa prompt và response từng cặp (diagonal)
    sims = np.array([cosine_similarity([ep], [er])[0,0] for ep, er in zip(emb_prompts, emb_responses)])
    return sims


def estimate_confidence_proxy(text: str) -> float:
    """
    Một proxy đơn giản cho 'độ tự tin' dựa trên ngôn ngữ:
    - câu khẳng định (contains 'definitely', 'certainly', 'always', 'will') tăng điểm
    - từ mơ hồ ('maybe','could','might','possibly') giảm điểm
    Trả về 0..1
    """
    text = text.lower()
    strong = len(re.findall(r"\b(definitely|certainly|always|will|must|absolutely|surely)\b", text))
    weak = len(re.findall(r"\b(maybe|could|might|possibly|seems|appear)\b", text))
    tokens = max(1, len(text.split()))
    score = (strong - 0.5*weak) / (np.log(tokens+1) + 1)
    # Kẹp đến 0..1 thông qua sigmoid-like
    score = 1/(1+np.exp(-score))
    return float(score)


def detect_factual_claims(text: str, max_claim_tokens: int = 20) -> List[str]:
    # heuristic: câu chứa số, ngày, tên riêng + động từ khẳng định
    claims = []
    sentences = re.split(r'[\.\n]+', text)
    for s in sentences:
        if len(s.split()) < 3:
            continue
        if re.search(r'\b\d{2,4}\b', s) or re.search(r'\b(in|on|at) \b', s):
            claims.append(s.strip())
        # phát hiện sự tồn tại của danh từ riêng (từ viết hoa) - ngây thơ
        if re.search(r'\b[A-Z][a-z]{2,}\b', s) and len(s.split()) <= max_claim_tokens:
            claims.append(s.strip())
    return list(dict.fromkeys(claims))


def estimate_hallucination_score(response: str, prompt: str = None) -> Dict[str, Any]:
    """
    Tính toán điểm hallucination dựa trên nhiều yếu tố:
    1. Độ tự tin ngôn ngữ (như trước)
    2. Sự không nhất quán nội tại
    3.Mức độ chi tiết không chắc chắn
    4. Phân tích cấu trúc phủ định
    """
    response= response.lower()
    
    # 1. Confidence scoring (như trước nhưng cải tiến)
    strong_assertions = len(re.findall(r"\b(definitely|certainly|always|will|must|absolutely|surely|undoubtedly|clearly)\b", response))
    uncertain_terms = len(re.findall(r"\b(maybe|could|might|possibly|seems|appear|likely|probably|perhaps|potentially)\b", response))
    disclaimer_phrases = len(re.findall(r"\b(as far as i know|to my knowledge|i believe|i think|in my opinion|supposedly|reportedly|allegedly)\b", response))
    
    confidence_score = (strong_assertions - 0.5 * uncertain_terms - 0.7 * disclaimer_phrases) / max(1, len(response.split()))
    confidence_score = 1 / (1 + np.exp(-confidence_score))  #Normalize to 0-1
    
    # 2. Self-contradiction detection
    contradiction_indicators = len(re.findall(r"\b(but|however|although|though|yet|nevertheless|nonetheless|on the other hand)\b", response))
    contradiction_score = min(1.0, contradiction_indicators / max(1, len(response.split()) / 20))
    
    # 3. Over-specificity detection (potentially made-up details)
    digits_count = len(re.findall(r'\d+', response))
    specific_detail_patterns = len(re.findall(r"(in \d{4}|since\d{4}|for \d+ years|costs? \$?\d+|population (of )?\d+|length \d+)", response))
    over_specificity_score = min(1.0, (digits_count + specific_detail_patterns) / max(1, len(response.split()) / 10))
    
    # 4. Negation structure analysis
    negations = len(re.findall(r"\b(not|no|never|neither|nowhere|nothing|nobody|none)\b", response))
    negation_score = min(1.0, negations / max(1,len(response.split()) / 15))
    
    # Tổng hợp điểm hallucination (cao hơn = khả năng hallucination cao hơn)
    hallucination_score = (
        0.3 * confidence_score +           # Tự tin quá mức có thể là dấu hiệu của hallucination
        0.25 * contradiction_score +       # Tự mâu thuẫn
        0.25 * over_specificity_score +    # Chi tiết cụ thể nghi ngờ
        0.2 * negation_score               # Phủ nhận phức tạp có thể che giấu sự không chắc chắn
    )
    
    return {
        'hallucination_score': float(hallucination_score),
        'confidence_component': float(confidence_score),
        'contradiction_component': float(contradiction_score),
        'specificity_component': float(over_specificity_score),
        'negation_component': float(negation_score)
    }


# PHẦN 5: Structural parsing - tách thành mở bài, giải thích, ví dụ, kết luận

def split_structure(text: str) -> Dict[str, str]:
    # heuristic: dùng các dấu hiệu từ khoá
    parts = {'intro': '', 'explain': '', 'example': '', 'conclusion': ''}
    text = text.strip()
    # tìm example keywords
    ex_pos = re.search(r"\b(example|ví dụ|for example|e\.g\.|e.g\.)\b", text, flags=re.I)
    concl_pos = re.search(r"\b(in summary|tóm lại|kết luận|to conclude|therefore)\b", text, flags=re.I)

    if ex_pos:
        parts['example'] = text[ex_pos.start():]
        text = text[:ex_pos.start()]
    if concl_pos:
        parts['conclusion'] = text[concl_pos.start():]
        text = text[:concl_pos.start()]

    # Tách phần còn lại thành phần giới thiệu (first sentence/2) và explain(rest)
    sents = re.split(r'(?<=[\.!?])\s+', text)
    if len(sents) <= 2:
        parts['intro'] = ' '.join(sents)
    else:
        parts['intro'] = sents[0]
        parts['explain'] = ' '.join(sents[1:])

    # dọn dẹp
    for k in parts:
        parts[k] = parts[k].strip()
    return parts


# PHẦN 6: Analysis pipeline

def run_analysis(df: pd.DataFrame, top_k_clusters: int = 6) -> Dict[str, Any]:
    df = df.copy()
    df['prompt'] = df['prompt'].astype(str).apply(normalize_text)
    df['response'] = df['response'].astype(str).apply(normalize_text)

    emb = Embedder()
    print('Fitting prompt embeddings...')
    emb_prompts = emb.fit_transform(df['prompt'].tolist())
    print('Fitting response embeddings...')
    emb_responses = emb.transform(df['response'].tolist())

    # Sự giống nhau trên mỗi cặp
    sims = response_prompt_similarity(emb_prompts, emb_responses)
    df['prompt_response_sim'] = sims

    # Phân tích ảo giác (Hallucination)
    hallucination_results = df.apply(lambda row: estimate_hallucination_score(row['response'], row['prompt']), axis=1)
    df['hallucination_score'] = [r['hallucination_score'] for r in hallucination_results]
    df['confidence_component'] =[r['confidence_component'] for r in hallucination_results]
    df['contradiction_component'] = [r['contradiction_component'] for r in hallucination_results]
    df['specificity_component'] = [r['specificity_component'] for r in hallucination_results]
    df['negation_component'] = [r['negation_component'] for r in hallucination_results]

    # Tuyên bố thực tế (giữ nguyên bản gốc để tương thích ngược)
    df['claims'] = df['response'].apply(detect_factual_claims)
    df['num_claims'] = df['claims'].apply(len)

    # cấu trúc
    struct = df['response'].apply(split_structure)
    df['intro'] = struct.apply(lambda x: x['intro'])
    df['explain'] = struct.apply(lambda x: x['explain'])
    df['example'] = struct.apply(lambda x: x['example'])
    df['conclusion'] = struct.apply(lambda x: x['conclusion'])

    # Phân cụm phản hồi để tìm kiểu
    print('Clustering responses...')
    # bình thường hoá
    X = emb_responses
    Xn = normalize(X)
    k = min(top_k_clusters, max(2, len(df)//10))
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(Xn)
    df['cluster'] = labels

    # PCA để trực quan hóa
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(Xn)
    df['pc1'] = pcs[:,0]
    df['pc2'] = pcs[:,1]

    results = {
        'df': df,
        'emb_prompts': emb_prompts,
        'emb_responses': emb_responses,
        'kmeans': kmeans,
        'pca': pca,
    }
    return results


# PHẦN 7: Visualization helpers

def plot_similarity_hist(df: pd.DataFrame):
    plt.figure(figsize=(6,4))
    plt.hist(df['prompt_response_sim'].dropna(), bins=30)
    plt.title('Prompt-Response similarity distribution')
    plt.xlabel('cosine similarity')
    plt.ylabel('count')
    plt.tight_layout()
    plt.show()


def plot_clusters_interactive(df: pd.DataFrame):
    fig = px.scatter(df, x='pc1', y='pc2', color=df['cluster'].astype(str),
                     hover_data=['id','prompt','response','prompt_response_sim','confidence_component','num_claims'])
    fig.update_layout(title='Response clusters (interactive)')
    fig.show()


def top_terms_per_cluster(df: pd.DataFrame, n_terms: int = 8):
    # ước tính các thuật ngữ TF-IDF hàng đầu trên mỗi cụm (phỏng đoán nhanh)
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['response'].tolist())
    terms = np.array(vectorizer.get_feature_names_out())
    clusters = df['cluster'].unique()
    out = {}
    for c in clusters:
        idx = df['cluster'] == c
        avg = X[idx.values].mean(axis=0).A1
        topi = np.argsort(avg)[-n_terms:][::-1]
        out[c] = terms[topi].tolist()
    return out


# PHẦN 8: Example usage

if __name__ == '__main__':
    # ví dụ giả sử có file logs.csv
    example_path = 'logs.csv'
    if not os.path.exists(example_path):
        print('Không tìm thấy logs.csv. Tạo sample nhỏ để demo...')
        sample = [
            {'id':0, 'prompt':'Làm thế nào để sắp xếp một list trong Python?', 'response':'Bạn có thể dùng sorted(list) hoặc list.sort(). Ví dụ: sorted([3,1,2]).'},
            {'id':1, 'prompt':'Giải thích time complexity của quicksort', 'response':'Quicksort trung bình O(n log n), worst-case O(n^2). Sử dụng pivot tốt để tránh worst-case.'},
            {'id':2, 'prompt':'Viết ví dụ SQL join', 'response':'Ví dụ: SELECT * FROM A JOIN B ON A.id = B.a_id;'},
        ]
        df = pd.DataFrame(sample)
    else:
        df = load_logs(example_path)

    res = run_analysis(df, top_k_clusters=4)
    df_out = res['df']

    print('\n--- Summary ---')
    print('Số bản ghi:', len(df_out))
    print('Sim mean:', df_out['prompt_response_sim'].mean())
    print('Hallucination score mean:', df_out['hallucination_score'].mean())
    print('Confidence component mean:', df_out['confidence_component'].mean())
    
    # Hiển thị phân tích ảo giác cho các mục hàng đầu
    print('\n--- Top potential hallucinations ---')
    top_hallucinations = df_out.nlargest(3, 'hallucination_score')[['prompt', 'response', 'hallucination_score']]
    for _, row in top_hallucinations.iterrows():
        print(f"Hallucination Score: {row['hallucination_score']:.3f}")
        print(f"Prompt: {row['prompt']}")
        print(f"Response: {row['response'][:100]}...")
        print("-" * 50)

    plot_similarity_hist(df_out)
    plot_clusters_interactive(df_out)

    print('\nTop terms per cluster:')
    print(top_terms_per_cluster(df_out))

    # Xuất kết quả
    df_out.to_csv('analysis_results.csv', index=False)
    print('Kết quả đã lưu: analysis_results.csv')
