from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from vocab import Vocab
import torch
import utils
from models.bilstm_crf import BiLSTMCRF
from datetime import datetime
import re

app = Flask(__name__)
CORS(app)

# 配置
class Config:
    SENT_VOCAB = './vocab/sent_vocab.json'
    TAG_VOCAB = './vocab/tag_vocab.json'
    MODEL = './trained_model/BiLSTMCRF/model.pth'
    CUDA = False  # 根据你的环境调整
    BATCH_SIZE = 32


config = Config()

# 全局变量存储加载的模型和词汇表
_model, _device, _sent_vocab, _tag_vocab = None, None, None, None


def init_components():
    global _model, _device, _sent_vocab, _tag_vocab
    if _model is None:
        try:
            _sent_vocab = Vocab.load(config.SENT_VOCAB)
            _tag_vocab = Vocab.load(config.TAG_VOCAB)

            device = torch.device('cuda' if config.CUDA else 'cpu')
            _model = BiLSTMCRF.load(config.MODEL, device)

            #_model = TransformerCRF.load(config.MODEL, device)

            _model.eval()
            _device = device
            print("模型和词汇表加载成功!")
        except Exception as e:
            print(f"初始化失败: {e}")
            raise

def get_special_tokens():
    """简化获取特殊标记"""
    special_tokens = {
        'pad': _sent_vocab.get_word2id().get('<PAD>', 0),
        'bos': _sent_vocab.get_word2id().get('<START>', 1),
        'eos': _sent_vocab.get_word2id().get('<END>', 2)
    }
    return special_tokens

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        init_components()

        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': '没有提供文本'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'error': '文本不能为空'}), 400

        if len(text) > 1000:
            return jsonify({'error': '文本长度不能超过1000个字符'}), 400

        # 处理文本
        sentences = split_into_sentences(text)  # 先分割句子
        results = process_text(text)  # 处理文本

        # 计算句子总数
        total_sentences = len(sentences)

        return jsonify({
            'success': True,
            'results': results,
            'process_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'text_length': len(text),
            'total_sentences': total_sentences  # 添加句子总数
        })

    except Exception as e:
        return jsonify({'error': f'处理文本时出错: {str(e)}'}), 500




def process_text(text):
    """处理文本并返回NER结果"""
    sentences = split_into_sentences(text)
    test_data = prepare_test_data(sentences)

    results = []
    special_tokens = get_special_tokens()

    with torch.no_grad():
        test_iterator = utils.batch_iter(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

        for batch_sentences, _ in test_iterator:
            padded_sentences, sent_lengths = utils.pad(batch_sentences, special_tokens['pad'], _device)
            predicted_tags = _model.predict(padded_sentences, sent_lengths)

            for orig_sentence, sentence, pred_tags in zip(batch_sentences, batch_sentences, predicted_tags):
                filtered_tokens = []
                filtered_tags = []

                for token, tag in zip(sentence, pred_tags):
                    word = _sent_vocab.id2word(token)  # 通过id2word获取词
                    if word in ['<PAD>', '<START>', '<END>']:
                        continue
                    filtered_tokens.append(word)
                    filtered_tags.append(tag)

                sentence_results = []
                for word, pred_tag in zip(filtered_tokens, filtered_tags):
                    tag = _tag_vocab.id2word(pred_tag)  # 使用id2word获取tag
                    sentence_results.append({
                        'word': word,
                        'tag': tag,
                        'tag_class': get_tag_class(tag)
                    })

                original_text = ''.join(filtered_tokens)
                if original_text.strip():
                    results.append({
                        'sentence': original_text,
                        'tokens': sentence_results
                    })

    return results



def split_into_sentences(text):
    """将文本分割成句子"""
    sentences = re.split(r'([。！？!?；;])', text)
    combined = []
    for i in range(0, len(sentences) - 1, 2):
        combined.append(sentences[i] + sentences[i + 1])
    if len(sentences) % 2 == 1:
        combined.append(sentences[-1])
    return [s.strip() for s in combined if s.strip()]


def prepare_test_data(sentences):
    """准备测试数据"""
    test_data = []
    special_tokens = get_special_tokens()

    for sentence in sentences:
        if not sentence:
            continue

        chars = list(sentence)
        # 使用get_word2id()获取词汇表映射
        char_indices = [ _sent_vocab.get_word2id().get(char, 3) for char in chars]  # 默认使用3作为未知字符的索引

        if 'bos' in special_tokens:
            char_indices = [special_tokens['bos']] + char_indices
        if 'eos' in special_tokens:
            char_indices = char_indices + [special_tokens['eos']]

        tag_indices = [special_tokens['pad']] * len(char_indices)
        test_data.append((char_indices, tag_indices))

    return test_data

def get_tag_class(tag):
    """根据标签返回CSS类名"""
    print(tag)
    if tag.startswith('B-') or tag.startswith('I-'):
        entity_type = tag[2:]  # 去掉 B- 或 I- 前缀，获取实体类型
        if entity_type in ['PER', 'ORG', 'LOC']:
            return f'entity-{entity_type.lower()}'
    return 'entity-other'  # 对于不匹配的标签，返回通用的其他类


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
