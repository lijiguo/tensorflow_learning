# -*- coding: utf-8 -*-
# https://github.com/CitronMan/nmt/blob/master/batch.py
'''
Prepare data
'''

import string
import numpy as np
import random
import os
import pickle

MAX_SENTENCE_LEN = 30
VOCAB_SIZE = 20000

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

DATA_PATH = './data/data/'


class CorpusPair:
    '''
    After initialization, contains a dictionary of most frequently
    occuring VOCAB_SIZE words. Then, batches can be generated.
    '''

    current_offset = 0   # updated after batching
    num_sentences = 0    # updated when word matrices are built

    def __init__(self, source, dst, load_flag = False):

        # open input files, store lines
        if not load_flag:
            print "Parse training data"
            try:
                with open(source) as s, open(dst) as t:
                    self.source = s.readlines()
                    self.dst = t.readlines()
            except:
                print("Could not open input files")
                exit(1)

            # then preprocess
            print " ... shuffle"
            self.shuffle()
            print " ... build lookup maps"
            self.build_dicts()
            self.save_dict(DATA_PATH);
            self.data_to_idx(source, dst);
            self.save_idxs(DATA_PATH);
            #print " ... vectorize data\n"
            #self.vectorize_corpora()
        else:
            self.load_dict(DATA_PATH);
            self.load_idxs(DATA_PATH);


    def shuffle(self):
        '''
        Prior to training, shuffle training data
        '''
        z = list(zip(self.source, self.dst))

        random.seed(1337)
        random.shuffle(z)

        self.source, self.dst = zip(*z)


    def build_dicts(self):
        '''
        Parse input files, create dictionary of most frequently used 30k words
        '''

        def build_count_dict(x):
            # Count frequencies of all words
            words = string.join(x).split(' ')
            freqs = {}
            for w in words:
                if w in freqs:
                    freqs[w] += 1
                else:
                    freqs[w] = 1

            # Sort words by frequency
            aux = [(freqs[key], key) for key in freqs]
            aux.sort()
            aux.reverse()
            aux = [w[1] for w in aux][:VOCAB_SIZE-1]

            # Build bidirectional dictionary to store word mappings
            idx2w = {}
            w2idx = {}
            for idx, w in enumerate(aux):
                idx2w[idx+1] = w
                w2idx[w] = idx+4

            # Store mapping for unknown words, PAD_ID = 0 GO_ID = 1 EOS_ID = 2 UNK_ID = 3
            idx2w[PAD_ID] = _PAD;
            w2idx[_PAD] = PAD_ID;
            idx2w[GO_ID] = _GO;
            w2idx[_GO] = GO_ID;
            idx2w[EOS_ID] = _EOS;
            w2idx[_EOS] = EOS_ID;
            idx2w[UNK_ID] = _UNK;
            w2idx[_UNK] = UNK_ID;

            return (idx2w, w2idx)

        # Create list of top words
        # self.words_xxx[0] = index to word map
        # self.words_xxx[1] = word to index map
        self.words_src = build_count_dict(self.source)
        self.words_dst = build_count_dict(self.dst)


    def vectorize_corpora(self):
        '''
        Convert sentence pairs into vector representations
        '''

        def words_in_s(s): return len(s.split())

        # Count sentences that fit within length limit
        for idx,line in enumerate(self.source):
            if words_in_s(self.source[idx]) <= MAX_SENTENCE_LEN \
                    and words_in_s(self.dst[idx]) <= MAX_SENTENCE_LEN:
                self.num_sentences += 1

        # Initialize numpy matrices to store sentences as vectors
        self.vec_src = np.zeros(shape=(self.num_sentences,MAX_SENTENCE_LEN,VOCAB_SIZE,))
        self.vec_dst = np.zeros(shape=(self.num_sentences,MAX_SENTENCE_LEN,VOCAB_SIZE,))

        # Vectorize all sentences of valid length
        def encode(idx, idy, word, word_dict, sen_arr):
            if word in word_dict[1]:
                sen_arr[idx,idy,word_dict[1][word]] = 1
            else:
                sen_arr[idx,idy,0] = 1

        idx = 0
        for idl, line in enumerate(self.source):
            if words_in_s(self.source[idl]) <= MAX_SENTENCE_LEN \
                    and words_in_s(self.dst[idl]) <= MAX_SENTENCE_LEN:
                for idw, word in enumerate(self.source[idl].split()):
                    encode(idx, idw, word, self.words_src, self.vec_src)
                for idw, word in enumerate(self.dst[idl].split()):
                    encode(idx, idw, word, self.words_dst, self.vec_dst)
                idx += 1


    def decode(self, sentence, src=True):
        '''
        Given an encoded sentence matrix,
        return the represented sentence string (tokenized).
        '''

        words = []

        for word in sentence:
            idxs = np.nonzero(word)[0]
            if len(idxs) > 1:
                raise Exception("Multiple hot bits on word vec")
            elif len(idxs) == 0:
                continue

            if src:
                words.append(self.words_src[0][idxs[0]])
            else:
                words.append(self.words_dst[0][idxs[0]])

        return ' '.join(words)

    def decode_src(self, sentence):
        return self.decode(sentence)

    def decode_dst(self, sentence):
        return self.decode(sentence, src=False)

    def save_dict(self, save_path):
        with open(save_path + 'dict_src.npy', 'wb') as f:
            pickle.dump(self.words_src, f);
        with open(save_path + 'dict_dst.npy', 'wb') as f:
            pickle.dump(self.words_dst, f);

        #np.save(save_path + 'dict_src_0.npy', self.words_src[0]);
        #np.save(save_path + 'dict_src_1.npy', self.words_src[1]);
        #np.save(save_path + 'dict_dst_0.npy', self.words_dst[0]);
        #np.save(save_path + 'dict_dst_1.npy', self.words_dst[1]);

    def load_dict(self, load_path):
        #self.words_src = [np.load(load_path + 'dict_src_0.npy'), np.load(load_path + 'dict_src_1.npy')];
        #self.words_dst = [np.load(load_path + 'dict_dst_0.npy'), np.load(load_path + 'dict_dst_1.npy')];

        with open(load_path + 'dict_src.npy', 'rb') as f:
            self.words_src = pickle.load(f);
        with open(load_path + 'dict_dst.npy', 'rb') as f:
            self.words_dst = pickle.load(f);

    def get_minibatches(self):
        '''
        Shuffle, retrieve 1600 sentence pairs
        Sort by length, split into 20 minibatches
        '''
        print "Generating batch at i =",self.current_offset

        def sentence_len(sentence):
            length = 0
            for word in sentence:
                if len(np.nonzero(word)[0]) == 1:
                    length += 1
                else:
                    break
            return length

        # Select 1600 sentences
        src_pairs = self.vec_src[self.current_offset:self.current_offset+1600]
        dst_pairs = self.vec_dst[self.current_offset:self.current_offset+1600]
        self.current_offset += 1600

        if self.current_offset > self.num_sentences:
            self.current_offset -= self.num_sentences

        # Sort by sentence length
        # TODO Speed up this list comprehension
        lengths = [sentence_len(sentence) for sentence in src_pairs]
        z = list(zip(lengths, src_pairs, dst_pairs))
        z = sorted(z, key=lambda x: x[0])
        _, src_pairs, dst_pairs = zip(*z)

        # Split into 20 minibatches
        batches = []
        for i in range(1,21):
            batches.append((src_pairs[i*0:i*80], dst_pairs[i*0:i*80]))

        return batches

    def data_to_idx(self, source, dst):
        #read data
        try:
            with open(source) as s, open(dst) as t:
                self.source = s.readlines()
                self.dst = t.readlines()
        except:
            print("Could not open input files")
            exit(1)

        #read dict
        if self.words_src.__len__() == 0 or self.words_dst.__len__() == 0:
            self.load_dict(DATA_PATH);

        source_idx = [];
        dst_idx = [];

        #source
        sentences = self.source;
        for sentence in sentences:
            words = sentence.split(' ');
            sentence_idx = [self.words_src[1].get(w , UNK_ID) for w in words];
            #" ".join([str(tok) for tok in token_ids]) + "\n"
            #source_idx.append(" ".join([str(tok) for tok in sentence_idx]));
            source_idx.append(sentence_idx);

        #dst
        sentences = self.dst;
        for sentence in sentences:
            words = sentence.split(' ');
            sentence_idx = [self.words_dst[1].get(w, UNK_ID) for w in words];
            #dst_idx.append(" ".join([str(tok) for tok in sentence_idx]));
            dst_idx.append(sentence_idx);

        self.source_idxs = source_idx;
        self.dst_idxs = dst_idx;
        return source_idx, dst_idx;

    def save_idxs(self, save_path):
        if len(save_path)>0:
            if not os.path.exists(save_path):
                os.makedirs(save_path);
            #else:


            source_idx_str = [];
            for idx in self.source_idxs:
                source_idx_str.append(" ".join([str(tok) for tok in idx]));

            with open(save_path + 'source_idx.npy', 'wb') as f:
                pickle.dump(source_idx_str, f);

            dst_idx_str = [];
            for idx in self.dst_idxs:
                dst_idx_str.append(" ".join([str(tok) for tok in idx]));

            with open(save_path + 'dst_idx.npy', 'wb') as f:
                pickle.dump(dst_idx_str, f);

            #np.save(save_path + 'source_idx.npy', self.source_idxs);
            #np.save(save_path + 'dst_idx.npy', self.dst_idxs);
        else:
            print('save_path is null');

    def load_idxs(self, load_path):
        if len(load_path)>0:
            if not os.path.exists(load_path):
                print('load_path does not exist');
            else:
                if os.path.isfile(load_path+'source_idx.npy'):
                    with open(load_path + 'source_idx.npy', 'rb') as f:
                        source_idxs_str = pickle.load(f);

                    self.source_idxs = [];
                    for sentence in source_idxs_str:
                        self.source_idxs.append([int(x) for x in sentence.split(' ')]);

                    #source_idxs = np.load(load_path+'source_idx.npy').item();
                    #self.source_idxs = source_idxs;
                    print('load source_idx file successfully')
                else:
                    print('load_file does not exist');

                if os.path.isfile(load_path+'dst_idx.npy'):
                    with open(load_path + 'dst_idx.npy', 'rb') as f:
                        dst_idx_str = pickle.load(f);

                    self.dst_idxs = [];
                    for sentence in dst_idx_str:
                        self.dst_idxs.append([int(x) for x in sentence.split(' ')]);

                    #dst_idxs = np.load(load_path+'dst_idx.npy').item();
                    #self.dst_idxs = dst_idxs;
                    print('load dst_idx file successfully')
                else:
                    print('load_file does not exist');

