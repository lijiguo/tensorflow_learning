#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import pynlpir


def test():
    pynlpir.open()
    s = "因为比较懒，所以只修改了这句话";
    segments = pynlpir.segment(s);

    for segment in segments:
        print segment[0], '\t', segment[1];
    pynlpir.close();


def word_cut(test_flag=False):
    DATA_PATH = './data/';
    CORPUS_FILENAME_RAW = 'corpus.txt'
    CORPUS_FILENAME_SEG = 'corpus_seg.txt'

    pynlpir.open()

    fid_input = open(DATA_PATH+CORPUS_FILENAME_RAW,'r');
    fid_output = open(DATA_PATH+CORPUS_FILENAME_SEG,'w');

    line = fid_input.readline();

    line_idx = 0;
    while line is not None and len(line)>19:
        line = line[9:-11];#delete <content> and <\content>
        segments = pynlpir.segment(line,pos_tagging=False);
        result = ' '
        for segment in segments:
            result = result+segment+' ';

        if len(result)>4:
            fid_output.write(result+'\n');

        line = fid_input.readline();

        line_idx += 1;

        if test_flag and line_idx>10:
            break;

        print("line:%d\n"%(line_idx));

    print "end\n";

    pynlpir.close()

if __name__ == "__main__":

    word_cut();

