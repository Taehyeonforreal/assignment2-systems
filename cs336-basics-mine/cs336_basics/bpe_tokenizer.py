import os
import regex as re
from collections import defaultdict

# GPT-2 pre-tokenization Pattern
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# Before merging, draw boundary. Contraction, Letter, Number, Punctuation, Space Units.
# Given by Assignment 1 Instruction

# pretokenization_example에 있는 코드 복붙
def find_chunk_boundaries(file, desired_num_chunks, split_special_token):
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))

def run_train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    # input_path : BPE를 학습할 텍스트 파일 경로
    # vocab_size : 최종 vocab 크기 목표치
    # special_token : BPE merge에서 제외할 특수 토큰
    # **kawrgs : 추가 인자 자유롭게 받을 수 있게 하는 것. Keyword Arguments

    # 1. 초기 vocab 설정 (0~255: 기본 byte + special tokens). dictionary 형태로.
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    
    # 2. 파일 읽고 pre-tokenization → 각 word의 byte sequence 빈도 계산
    # defaultdict : 존재하지 않는 Key 여도 에러 없이 바로 새 Key 만들어줌
    # int : 새로 만들어진 Key의 Value=0으로.
    word_counts = defaultdict(int)  # (byte1, byte2, ...) → 등장 횟수
    
    with open(input_path, 'rb') as f: 
        # input_path 파일 read with binary(byte)
        # 그걸 f라는 변수에 담는다. f는 열린 파일 객체. with 써서, 작업 끝나면 자동으로 파일 닫아

        # special token 기준으로 청크 나누기
        boundaries = find_chunk_boundaries(f, 4, b"<|endoftext|>")
        # 4 : 4개의 chunk로 나눈다. b".."는 bytes type이라는 뜻
        # <|endoftext|>는 문서 사이를 구분하는 특수 토큰

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # zip은 두 list 짝지어 dim+1 리스트 반환. (마지막 뺀 리스트, 첨 뺀 리스트)

            f.seek(start)
            # 파일에서 읽기 시작할 위치 지정. 이거 안하면 항상 처음부터 읽음
            chunk = f.read(end - start).decode('utf-8', errors='ignore')
            # error ignore 이유는. 깨진 byte있을 수 있으니까. 그냥 무시후 건너뛰기
            
            # special token 제거
            # special token은 bpe merge의 대상이 되면 안됨
            for token in special_tokens:
                chunk = chunk.replace(token, '')
            
            # pre-tokenization: 단어 단위로 쪼개기
            # re.findall(패턴, 텍스트) 패턴에 맞는 모든 부분을 리스트로
            words = re.findall(GPT2_PAT, chunk)
            
            # 각 단어를 byte sequence로 변환 후 빈도 카운팅
            for word in words:
                byte_seq = tuple(word.encode('utf-8'))
                # tuple : 변환한 byte들을 카운트
                # tuple은 list와 달리 immutable이라, dict key로 사용 가능.
                word_counts[byte_seq] += 1

    # 3. BPE merge 반복
    # 강의 코드와 동일한 로직, 근데 word_counts 기반으로
    # byte_base 로 하면 띄어쓰기 같은것도 포함해버릴 위험. word_base 기반으로 시작.
    merges = []
    num_merges = vocab_size - len(vocab)
    # BPE merge는 vocab 크기를 늘리는 대신, 시퀀스 길이를 줄인다.
    # 한번 merge 할때마다 len(vocab)++ 이니까, merge 횟수 정하는거임

    for _ in range(num_merges):
        # 3-1. 모든 인접 쌍의 빈도 계산. word 안에서
        pair_counts = defaultdict(int)
        for byte_seq, count in word_counts.items():
            for b1, b2 in zip(byte_seq, byte_seq[1:]):
                pair_counts[(b1, b2)] += count

        if not pair_counts:
            break
        # 극단적으로, 모든 단어들이 단일 글자면, pair가 없을수도

        # 3-2. 가장 많은 쌍 찾기. 동점이면 byte값 기준으로 최댓값
        best_pair = max(
            pair_counts,
            key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]])
        )
        # lambda p: (pair_counts[p], p)는 p를 받아서 pair_counts[p]를 반환하는 함수
        # max(pair_counts, key=lambda p: pair_counts[p]) 그냥 이렇게만 써도 됨
        # 근데 동점일때 대비, 2차원 튜플로, p 값 비교해서 숫자쌍 자체가 큰걸로.


        # 3-3. 새 토큰으로 합치기
        new_id = len(vocab)
        new_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[new_id] = new_bytes
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # 3-4. word_counts에서 해당 쌍 합치기
        new_word_counts = defaultdict(int)
        for byte_seq, count in word_counts.items(): # items()안하면 Key만
            new_seq = []
            i = 0
            while i < len(byte_seq):
                if i < len(byte_seq) - 1 and (byte_seq[i], byte_seq[i+1]) == best_pair:
                    new_seq.append(new_id)
                    i += 2
                    # best pair 발견하면, new_id를 new_seq에
                else:
                    new_seq.append(byte_seq[i])
                    i += 1
                    # 발견 못했으니까 그대로
            new_word_counts[tuple(new_seq)] += count
            # 지금 new_seq는 list라 Key로 사용x, tuple로 바꿔서, count 다 집어넣기
        word_counts = new_word_counts

    return vocab, merges
    # 두개를 따로 주는 이유는, merges에선 합쳐지는 순서를 볼 수 있어

# 실제로 텍스트를 토큰화하는 객체를 만들기
class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab 
        self.merges = merges
        self.special_tokens = special_tokens or [] # None 이면 빈 리스트로
        
        # bytes → int 역방향 lookup (decode에 필요). 역방향 딕셔너리
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        
        # merge 우선순위 (앞에 있을수록 먼저 적용)
        # merge 딕셔너리에 앞에 있을 수록 먼저 학습된거라, 이걸 저장한다고 보는거지
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

    # 텍스트를 토큰 ID 리스트로 변환
    def encode(self, text: str) -> list[int]:
        tokens = []
        
        # 1. special token 먼저 분리
        if self.special_tokens:
            # special token 기준으로 텍스트 분리
            pattern = '(' + '|'.join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)) + ')'
            # 코드 설명
            # 1.  sorted(self.special_tokens, key=len, reverse=True)
            #     special token을 길이 내림차순 정렬. 짧은게 긴것의 prefix일수 있으니
            # 2.  re.escape(t) for t in sorted
            #     re.escape : 특수 의미 가진 문자(|()...)를 일반 문자로 취급해줌
            # 3.  '|'.join
            #     여러 special token을 "또는"으로 연결. regex 패턴 문자열로 기능.
            #     이러면 re.split에서 |로 연결된 많은 special token 중 하나만 만나도 split
            # 4. '(' + ... + ')'
            #     구분자도 split 결과에 포함됨. 즉 special token도 결과에 포함하는 것.
            #     그래야 special token에도 ID를 부여하니까

            parts = re.split(pattern, text)
        else:
            parts = [text]
        # 여기서 parts를 list로 해야 아래 for문 가능. re.split 결과도 list

        for part in parts:
            if part in self.special_tokens:
                # special token은 그대로 ID로 변환
                # part가 special token의 경우일 때, 
                # 이걸 utf-8 규격으로 bytes로 변환하고, 
                # (bytes, id) 딕셔너리를 찾아서, 
                # ID를 tokens에 추가한다.
                tokens.append(self.bytes_to_id[part.encode('utf-8')])
            elif part:
                # 일반 텍스트는 BPE 적용. part 별로 순차적으로.
                tokens.extend(self._encode_chunk(part))
        
        return tokens

    # encode의 내부 헬퍼
    def _encode_chunk(self, text: str) -> list[int]:
        # pre-tokenization
        # re.findall(패턴, 텍스트) 패턴에 맞는 모든 부분을 리스트로
        words = re.findall(GPT2_PAT, text)
        token_ids = []
        
        for word in words:
            # 각 글자를 byte ID로 변환
            word_tokens = [self.bytes_to_id[bytes([b])] for b in word.encode('utf-8')]
            
            # merge 적용
            while len(word_tokens) >= 2: # 토큰이 두개 이상일때만
                # 현재 인접 쌍 중 가장 우선순위 높은 merge 찾기
                best = None
                best_rank = float('inf')
                
                for i in range(len(word_tokens) - 1):
                    pair = (self.vocab[word_tokens[i]], self.vocab[word_tokens[i+1]])
                    rank = self.merge_ranks.get(pair, float('inf'))
                    # get()에서 pair가 있으면 그 Value 반환, 없으면 무한대
                    if rank < best_rank:
                        best_rank = rank
                        best = (i, pair)
                
                if best is None or best_rank == float('inf'):
                    break  # 더 이상 적용할 merge 없음
                
                # merge 적용
                i, pair = best
                new_bytes = pair[0] + pair[1]
                new_id = self.bytes_to_id[new_bytes]
                word_tokens = word_tokens[:i] + [new_id] + word_tokens[i+2:]
            
            token_ids.extend(word_tokens)
            # extend는 append와 다르게, 모든 들어온 거를 1-Dim 의 ID 리스트로
        
        return token_ids


    # 토큰 ID 리스트를 텍스트로
    def decode(self, ids: list[int]) -> str:
        # ID → bytes → 문자열
        byte_string = b''.join(self.vocab[i] for i in ids) # 구분자 없이 합치기
        return byte_string.decode('utf-8', errors='replace')
        # replace로 error 대체

        # iterator 구성하기
    def encode_iterable(self, iterable):
        # iterable: 줄 단위로 나오는 것 (파일 객체 등)
        # 한 줄씩 읽어서 encode하고, ID를 하나씩 yield
        for line in iterable:
            ids = self.encode(line)
            for id in ids:
                yield id


def get_tokenizer(vocab, merges, special_tokens=None):
    return BPETokenizer(vocab, merges, special_tokens)

