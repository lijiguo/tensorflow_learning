#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

DATA_PATH = './data/data/';
CORPUS_FILENAME_RAW = 'corpus.txt'
CORPUS_FILENAME_SEG = 'en.txt' #'corpus_seg.txt'

MODEL_FILE_NAME = 'word2vector_en.model'

'''
    :param data_file_name:
    :return:
'''
def read_data(data_file_name, test_flag = False):

    fid = open(data_file_name);

    line = str(fid.readline());
    counter = 0;
    while line is not None and len(line)>4:
        words = line.split(' ');
        yield words;

        line = str(fid.readline());
        counter += 1;
        if test_flag and counter>10:
            break;

    fid.close();


def test():
    sentances = read_data(DATA_PATH+CORPUS_FILENAME_SEG, test_flag=True);

    for sentance in sentances:
        for word in sentance:
            print(unicode(word, encoding='utf-8'));


class TextLoader(object):
    def __int__(self):
        pass;

    def __iter__(self):
        fid = open(DATA_PATH+CORPUS_FILENAME_SEG);

        line = str(fid.readline());
        counter = 0;
        while line is not None and len(line)>4:
            words = line.split(' ');
            yield words;

            line = str(fid.readline());
            counter += 1;
            #if counter>100:
            #    break;

        fid.close();


if __name__ == "__main__":
    #sentences = read_data(DATA_PATH+CORPUS_FILENAME_SEG, test_flag=True);
    sentences = TextLoader();
    print 'train the model...'
    model = Word2Vec((sentences), workers=multiprocessing.cpu_count());
    print 'train end, save the model...'
    model.save(MODEL_FILE_NAME);
    print 'save end, test the model...'
    #print '台湾和中国的相似度'+str(model.similarity('台湾','中国'));
    print '台湾和大学的相似度'+str(model.similarity('I','me'));
