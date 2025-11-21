from docopt import docopt
from vocab import Vocab
import time
import torch
import torch.nn as nn
from models.bilstm_crf import BiLSTMCRF
import utils
import random
from tqdm import tqdm
from models.transformer_crf import TransformerCRF
import gc

def get_args():
    """创建并返回参数字典，模拟命令行参数"""
    args = {
        'train': False,  # 设置为True表示训练模式
        'test': True,  # 设置为False
        'TRAIN': './data/NER-train-utf8.txt',  # 训练数据路径
        'TEST': './data/NER-test-utf8.txt',  # 测试数据路径
        'RESULT': './result.txt',  # 结果保存路径
        'SENT_VOCAB': './vocab/sent_vocab.json',  # 句子词典路径
        'TAG_VOCAB': './vocab/tag_vocab.json',  # 标签词典路径
        'MODEL': './trained_model/T/model.pth',  # 模型路径
        '--dropout-rate': '0.3',
        '--embed-size': '256',
        '--hidden-size': '256',
        '--batch-size': '32',
        '--max-epoch': '100',
        '--clip_max_norm': '5.0',
        '--lr': '1e-3',
        '--log-every': '10',
        '--max-patience': '2',
        '--max-decay': '4',
        '--lr-decay': '0.5',
        '--model-save-path': './trained_model/T/model.pth',
        '--optimizer-save-path': './trained_model/T/optimizer.pth',
        '--cuda': True,
        '--debug-train': False,              # 是否在训练时打印预测（默认 True）
        '--debug-train-samples': '2'  
    }
    return args

def train(args):
    """ Training BiLSTMCRF model
    Args:
        args: dict that contains options in command
    """
    sent_vocab = Vocab.load(args['SENT_VOCAB'])
    tag_vocab = Vocab.load(args['TAG_VOCAB'])
    train_data, dev_data = utils.generate_train_dev_dataset(args['TRAIN'], sent_vocab, tag_vocab)
    print('num of training examples: %d' % (len(train_data)))
    print('num of development examples: %d' % (len(dev_data)))

    max_epoch = int(args['--max-epoch'])
    model_save_path = args['--model-save-path']
    optimizer_save_path = args['--optimizer-save-path']
    min_dev_loss = float('inf')
    device = torch.device('cuda' if args['--cuda'] else 'cpu')
    patience, decay_num = 0, 0

    #model = BiLSTMCRF(sent_vocab, tag_vocab, float(args['--dropout-rate']), int(args['--embed-size']),int(args['--hidden-size'])).to(device)
    model = TransformerCRF.load(args['MODEL'], device)
    '''
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, 0, 0.01)
        else:
            nn.init.constant_(param.data, 0)
    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args['--lr']))

    print('start training...')

    # 训练统计信息
    train_history = {
        'train_loss': [],
        'dev_loss': [],
        'learning_rate': []
    }

    debug_train = bool(args['--debug-train'])
    debug_train_samples = int(args['--debug-train-samples'])

    for epoch in range(max_epoch):
        # 训练阶段 - 使用更详细的进度条
        model.train()
        epoch_loss = 0
        total_samples = 0
        total_batches = 0

        # 计算总batch数用于进度条
        total_batches_estimate = len(train_data) // int(args['--batch-size']) + 1

        # 创建更详细的进度条
        train_iterator = utils.batch_iter(train_data, batch_size=int(args['--batch-size']))
        pbar = tqdm(train_iterator,
                    desc=f'🚀 Epoch {epoch + 1}/{max_epoch}',
                    total=total_batches_estimate,
                    unit='batch',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                    ncols=180)

        batch_start_time = time.time()

        for batch_idx, (raw_sentences, raw_tags) in enumerate(pbar):
            current_batch_size = len(raw_sentences)

            # pad inputs and tags (padded tensors on device)
            padded_sentences, sent_lengths = utils.pad(raw_sentences, sent_vocab[sent_vocab.PAD], device)
            padded_tags, _ = utils.pad(raw_tags, tag_vocab[tag_vocab.PAD], device)

            # back propagation
            optimizer.zero_grad()
            batch_loss = model(padded_sentences, padded_tags, sent_lengths)
            loss = batch_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args['--clip_max_norm']))
            optimizer.step()

            batch_loss_value = batch_loss.mean().item()
            epoch_loss += batch_loss.sum().item()
            total_samples += current_batch_size
            total_batches += 1

            # ======== 新增：训练时按 batch 打印若干样本的预测信息（受 debug 控制） ========
            if debug_train:
                # 在训练中临时切换到 eval 模式进行预测，然后恢复 train
                model.eval()
                with torch.no_grad():
                    try:
                        predicted_tags = model.predict(padded_sentences, sent_lengths)
                    except Exception as e:
                        predicted_tags = [[] for _ in range(current_batch_size)]
                        print(f"[WARN] model.predict failed during training debug: {e}")

                    n_print = min(debug_train_samples, current_batch_size)
                    for i in range(n_print):
                        # raw_sentences[i], raw_tags[i] 是原始 index 列表（含首尾标记），predicted_tags[i] 是对应的 id 列表
                        sent_ids = raw_sentences[i]
                        true_ids = raw_tags[i]
                        pred_ids = predicted_tags[i] if i < len(predicted_tags) else []

                        # 去掉首尾（跟测试脚本保持一致）
                        sent_ids_trim = sent_ids[1:-1]
                        true_ids_trim = true_ids[1:-1]
                        # predicted 可能长度和 true 不完全一样，尽量对齐
                        if len(pred_ids) >= 2:
                            pred_ids_trim = pred_ids[1:-1]
                        else:
                            pred_ids_trim = pred_ids

                        sent_words = [sent_vocab.id2word(x) for x in sent_ids_trim]
                        true_tags_words = [tag_vocab.id2word(x) for x in true_ids_trim]
                        pred_tags_words = [tag_vocab.id2word(x) for x in pred_ids_trim]

                        gold_entities = extract_entities(true_tags_words)
                        pred_entities = extract_entities(pred_tags_words)

                        print(f"[Train Debug] Epoch {epoch+1} Batch {batch_idx} Loss:{batch_loss_value:.4f} Sample:{i}")
                        print(" Sentence: ", " ".join(sent_words))
                        print(" True tags:", " ".join(true_tags_words))
                        print(" Pred tags:", " ".join(pred_tags_words))
                        print(" Gold entities:", gold_entities)
                        print(" Pred entities:", pred_entities)
                        print("-" * 40)
                model.train()
            # =====================================================================

            del padded_sentences, padded_tags, sent_lengths, batch_loss, loss
            torch.cuda.empty_cache() if args['--cuda'] else None
            gc.collect()
            # 计算处理速度
            batch_time = time.time() - batch_start_time
            samples_per_sec = current_batch_size / batch_time if batch_time > 0 else 0

            # 更新进度条描述 - 更详细的信息
            avg_epoch_loss = epoch_loss / total_samples
            current_lr = optimizer.param_groups[0]['lr']

            pbar.set_postfix({
                'Batch_Loss': f'{batch_loss_value:.4f}',
                'Epoch_Loss': f'{avg_epoch_loss:.4f}',
                'LR': f'{current_lr:.2e}',
                'Samples/Sec': f'{samples_per_sec:.1f}',
                'Patience': f'{patience}/{args["--max-patience"]}'
            })

            batch_start_time = time.time()

        # 计算epoch平均损失
        epoch_avg_loss = epoch_loss / total_samples
        train_history['train_loss'].append(epoch_avg_loss)
        train_history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # 每个epoch结束后进行验证
        print(f'\n📊 Epoch {epoch + 1} 训练完成, 开始验证...')
        print(f'训练损失: {epoch_avg_loss:.4f}')

        dev_loss = cal_dev_loss(model, dev_data, 64, sent_vocab, tag_vocab, device)
        train_history['dev_loss'].append(dev_loss)
        print("本轮验证损失",dev_loss)
        print("之前最佳验证损失", min_dev_loss)
        if dev_loss < min_dev_loss * 0.98:
            improvement = min_dev_loss - dev_loss
            min_dev_loss = dev_loss
            model.save(model_save_path)
            torch.save(optimizer.state_dict(), optimizer_save_path)
            patience = 0
            print(f'🎉 模型有改进! 损失下降: {improvement:.4f}')
            print(f'💾 模型已保存至: {model_save_path}')
        else:
            patience += 1
            print(f'😐 暂无改进，耐心计数: {patience}/{args["--max-patience"]}')

            if patience == int(args['--max-patience']):
                decay_num += 1
                if decay_num == int(args['--max-decay']):
                    print('🛑 提前停止触发! 训练结束。')
                    break

                # 学习率衰减
                old_lr = optimizer.param_groups[0]['lr']
                lr = old_lr * float(args['--lr-decay'])

                # 加载之前保存的最佳模型
                print('🔄 加载最佳模型并衰减学习率...')

                model = BiLSTMCRF.load(model_save_path, device)
                #model = TransformerCRF.load(model_save_path,device)

                optimizer.load_state_dict(torch.load(optimizer_save_path))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                patience = 0
                print(f'📉 学习率从 {old_lr:.2e} 衰减至: {lr:.2e}')

        # 输出验证结果 - 更直观的显示
        print('-' * 70)
        print(f'✅ 验证结果 - Epoch {epoch + 1}')
        print(f'   训练损失: {epoch_avg_loss:.4f}')
        print(f'   验证损失: {dev_loss:.4f}')
        print(f'   最佳验证损失: {min_dev_loss:.4f}')
        print(f'   学习率: {optimizer.param_groups[0]["lr"]:.2e}')
        print(f'   耐心计数: {patience}/{args["--max-patience"]}')
        print(f'   衰减次数: {decay_num}/{args["--max-decay"]}')
        print('-' * 70)

        print('\n' + '=' * 70 + '\n')

    # 训练结束总结
    print('🎊 训练完成!')
    print(f'📁 最佳模型保存在: {model_save_path}')
    print(f'📈 最终验证损失: {min_dev_loss:.4f}')
    print(f'🔄 总学习率衰减次数: {decay_num}')

def extract_entities(tag_seq):
    """
    将BIO标签序列转为实体 span 列表
    tag_seq: ["B-ORG", "I-ORG", "N", "B-PER"...]
    返回 [(start, end, type), ...]
    """
    entities = []
    start, ent_type = None, None

    for i, tag in enumerate(tag_seq):

        if tag.startswith('B-'):
            # 若前一个实体未结束，先关闭
            if start is not None:
                entities.append((start, i - 1, ent_type))
            start = i
            ent_type = tag[2:]

        elif tag.startswith('I-'):
            # 同一实体继续
            continue

        else:  # N
            if start is not None:
                entities.append((start, i - 1, ent_type))
                start, ent_type = None, None

    if start is not None:
        entities.append((start, len(tag_seq) - 1, ent_type))

    return entities


def tst(args):
    """ Testing the model with P/R/F1 + 每类实体的P/R/F1 """

    sent_vocab = Vocab.load(args['SENT_VOCAB'])
    tag_vocab = Vocab.load(args['TAG_VOCAB'])

    sentences, tags = utils.read_corpus(args['TEST'])
    sentences = utils.words2indices(sentences, sent_vocab)
    tags = utils.words2indices(tags, tag_vocab)
    test_data = list(zip(sentences, tags))
    print('num of test samples: %d' % (len(test_data)))

    device = torch.device('cuda' if args['--cuda'] else 'cpu')
    #model = BiLSTMCRF.load(args['MODEL'], device)

    model = TransformerCRF.load(args['MODEL'], device)

    print('start testing...')

    result_file = open(args['RESULT'], 'w')
    model.eval()

    # ==== 总指标 ====
    total_gold = 0
    total_pred = 0
    total_correct = 0

    # ==== 分类别指标 ====
    types = ["ORG", "LOC", "PER"]
    gold_per_type = {t: 0 for t in types}
    pred_per_type = {t: 0 for t in types}
    correct_per_type = {t: 0 for t in types}

    total_batches = len(test_data) // int(args['--batch-size']) + 1

    with torch.no_grad():
        test_iterator = utils.batch_iter(test_data, batch_size=int(args['--batch-size']), shuffle=False)

        for sentences, tags in tqdm(test_iterator, desc="🧪 测试中",
                                    total=total_batches, unit='batch'):

            padded_sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)

            predicted_tags = model.predict(padded_sentences, sent_lengths)

            for sent, true_tags, pred_tags in zip(sentences, tags, predicted_tags):

                sent = sent[1:-1]
                true_tags = true_tags[1:-1]
                pred_tags = pred_tags[1:-1]

                # 写 result.txt
                for tok, t, p in zip(sent, true_tags, pred_tags):
                    result_file.write(f"{sent_vocab.id2word(tok)} "
                                      f"{tag_vocab.id2word(t)} "
                                      f"{tag_vocab.id2word(p)}\n")
                result_file.write("\n")

                # BIO 标签转文字
                true_text = [tag_vocab.id2word(x) for x in true_tags]
                pred_text = [tag_vocab.id2word(x) for x in pred_tags]

                # N -> O
                true_text = [t for t in true_text]
                pred_text = [t for t in pred_text]

                gold_entities = extract_entities(true_text)
                pred_entities = extract_entities(pred_text)

                # -------------- 新增：在控制台打印每个样本的详细信息 --------------
                # 打印句子、真实标签、预测标签、实体列表
                sent_words = [sent_vocab.id2word(x) for x in sent]
                true_tags_words = [tag_vocab.id2word(x) for x in true_tags]
                pred_tags_words = [tag_vocab.id2word(x) for x in pred_tags]

                print("Sentence: ", " ".join(sent_words))
                print("True tags:", " ".join(true_tags_words))
                print("Pred tags:", " ".join(pred_tags_words))
                print("Gold entities:", gold_entities)
                print("Pred entities:", pred_entities)
                print("-" * 40)
                # -----------------------------------------------------------------

                total_gold += len(gold_entities)
                total_pred += len(pred_entities)

                # 按类型统计
                for (s, e, t) in gold_entities:
                    if t in types:
                        gold_per_type[t] += 1

                for (s, e, t) in pred_entities:
                    if t in types:
                        pred_per_type[t] += 1

                # 严格匹配
                for ent in pred_entities:
                    if ent in gold_entities:
                        total_correct += 1
                        if ent[2] in types:
                            correct_per_type[ent[2]] += 1

    # ==== 总指标 ====
    precision = total_correct / total_pred if total_pred else 0
    recall = total_correct / total_gold if total_gold else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n📊 Overall NER Results:")
    print(f"   Gold:      {total_gold}")
    print(f"   Predicted: {total_pred}")
    print(f"   Correct:   {total_correct}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1:        {f1:.4f}\n")

    # ==== 输出每类实体的结果 ====
    print("📌 Per-Entity-Type Results:")
    print(f"{'Type':<6} {'P':<8} {'R':<8} {'F1':<8} {'Gold':<6} {'Pred':<6} {'Correct'}")
    print("-" * 60)

    for t in types:
        g = gold_per_type[t]
        p = pred_per_type[t]
        c = correct_per_type[t]

        P = c / p if p else 0
        R = c / g if g else 0
        F = 2 * P * R / (P + R) if (P + R) else 0

        print(f"{t:<6} {P:.4f}   {R:.4f}   {F:.4f}   {g:<6} {p:<6} {c}")

    print("\n✅ 测试完成！")




def cal_dev_loss(model, dev_data, batch_size, sent_vocab, tag_vocab, device):
    """ Calculate loss on the development data
    Args:
        model: the model being trained
        dev_data: development data
        batch_size: batch size
        sent_vocab: sentence vocab
        tag_vocab: tag vocab
        device: torch.device on which the model is trained
    Returns:
        the average loss on the dev data
    """
    is_training = model.training
    model.eval()
    loss, n_sentences = 0, 0

    # 计算总batch数
    total_batches = len(dev_data) // batch_size + 1

    with torch.no_grad():
        dev_iterator = utils.batch_iter(dev_data, batch_size, shuffle=False)
        for sentences, tags in tqdm(dev_iterator,
                                    desc='🔍 验证中',
                                    total=total_batches,
                                    leave=False,
                                    unit='batch'):
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            tags, _ = utils.pad(tags, tag_vocab[tag_vocab.PAD], device)

            batch_loss = model(sentences, tags, sent_lengths)
            loss += batch_loss.sum().item()
            n_sentences += len(sentences)

    model.train(is_training)
    return loss / n_sentences


def main():
    # 使用我们自定义的参数获取函数，而不是docopt
    args = get_args()

    random.seed(0)
    torch.manual_seed(0)
    if args['--cuda']:
        torch.cuda.manual_seed(0)

    if args['train']:
        train(args)
    elif args['test']:
        tst(args)


if __name__ == '__main__':
    main()
