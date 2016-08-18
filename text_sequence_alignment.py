import sys, os, getopt
import string
import numpy
import re
from difflib import SequenceMatcher
import swalign
import textwrap

'''
for the swalign stuff, had to make some changes to the file to fix syntax
like changing xrange to range and including print things in ()
'''


def aligned_query_filtering(original,query):
    y=query.split()
    lengths=[]
    for a in y:
        x=len(a)
        lengths.append(x)
    for i in range(len(y)):
        if y[i] not in original:

            if y[i][0] is '-':
                regex=re.compile('^-+')
                y[i]=regex.sub('',y[i])
                y[i]=y[i].rjust(lengths[i])
            if y[i][-1] is '-':
                regex=re.compile('-+$')
                y[i]=regex.sub('',y[i])
                y[i]=y[i].ljust(lengths[i])

            regex=re.compile('-*')
            y[i]=regex.sub('', y[i])
            y[i]=y[i].ljust(lengths[i])
    return list_to_string(y)

def aligned_ref_filtering(original, ref):
    y=ref.split()
    lengths=[]
    for a in y:
        x=len(a)
        lengths.append(x)
    for i in range(len(y)):
        if y[i] not in original:

            if y[i][0] is '-':
                regex=re.compile('^-+')
                y[i]=regex.sub('',y[i])
                y[i]=y[i].rjust(lengths[i])
            if y[i][-1] is '-':
                regex=re.compile('-+$')
                y[i]=regex.sub('',y[i])
                y[i]=y[i].ljust(lengths[i])

            regex=re.compile('-*')
            y[i]=regex.sub('', y[i])
            y[i]=y[i].ljust(lengths[i])
    return list_to_string(y)


def list_to_string(string_list):
    s=''
    for sr in string_list:
        s+=sr+' '
    return s.lstrip().rstrip()


'''
number of errors (deletion, insertion, substitution)
----
wer code below from https://martin-thoma.com/word-error-rate-calculation/
'''
def wer(r, h): #r and h are lists
    # initialisation
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

def print_alignment(str1,str2, out, width=100):
    l1=len(str1)
    l2=len(str2)

    i = 0
    while i < min(l1,l2):
        line1 = str1[i:min(i+width,l1)]
        line2 = str2[i:min(i+width,l2)]

        i += width

        if min(i,l1-1,l2-1) is not i:
            break

        while str1[i] is not ' ' or str2[i] is not ' ':
            line1 += str1[i]
            line2 += str2[i]
            i += 1
            if min(i,l1-1,l2-1) is not i:
                break

        out.write(line1+'\n')
        out.write(line2+'\n\n')



def print_alignment_withSymbol(str1,str2, symbol, out, width=100):

    reg = [' ','-']

    l1=len(str1)
    l2=len(str2)
    l3=len(symbol)

    i = 0
    while i < min(l1,l2):
        line1 = str1[i:min(i+width,l1)]
        line2 = symbol[i:min(i+width,l3)]
        line3 = str2[i:min(i+width,l2)]

        i += width

        if min(i,l1-1,l2-1) is not i:
            break

        while str1[i] not in reg or str2[i] not in reg:
            line1 += str1[i]
            line2 += symbol[i]
            line3 += str2[i]
            i += 1
            if min(i,l1-1,l2-1) is not i:
                break

        out.write(line1+'\n')
        out.write(line2+'\n')
        out.write(line3+'\n\n')


def remove_ref_punctuation(lines_list,grnd): #lines_list is list of transcript file lines
    reg123=['\'','\"','\.*','\,','\?','\!']
    reg_list=[]
    for r in reg123:
        reg_list.append(re.compile(r))

    paragraphs=lines_list
    filtered=[]
    for p in paragraphs:
        fil=p
        for regex in reg_list:
            fil=regex.sub('', fil)
        filtered.append(fil)
    t=[]
    prev=''
    count=0
    for i in range(len(filtered)):

        f = filtered[i].lstrip().rstrip().rstrip('\n')
        if f == '':
            continue
        seq=SequenceMatcher(None,prev,f)
        a=seq.find_longest_match(0,len(prev),0,len(f))
        x=a[0]
        y=a[1]
        size=a[2]
        str1=prev[x:x+size].lstrip().rstrip()
        str2=f[y:y+size].lstrip().rstrip()
##            print(str1,' ||| ', str2)
##            print(a)
        if y==0:
            if prev[x:x+size] == prev[-size:]:
                c=str1+' '+str2
##                    print(prev[x:x+size])
##                    print(prev[-size:])
##                    print(c, c not in grnd)
                if c not in grnd:
##                        print(a)
##                        print(prev)
##                        print(f)
##                        print(c)
                    t.append(prev[:-size-1])

                else:
                    t.append(prev)

            else:
                t.append(prev)

        else:
            t.append(prev)
        prev=f

    #not all repetition is removed the first time b/c longest match
    #isn't always the beginning/end
    #this below helps get some of the smaller beginnig/end repetitions in transcript
    #lines, but not all.. I'm not sure how to fix it
    prev=''
    s=''
    count=0
    for i in range(len(t)):
        f = t[i].lstrip().rstrip()
##            print('CURR:', f)
##            print('PREV:',prev)
        if f == '':
            continue
        ps=prev.split()
        fs=f.split()
        j=0
        w=fs[j]
        #print(w)
        w=fs.pop(0)
        while w in ps and len(fs) !=0:
            w+=' '+fs[0]

        a=prev.find(w.strip())
        if a>=0:
##                print(prev)
##                print(f)
##                print(prev[a:],' |||| ',w)
##                print(a)
            c=prev[a:]+' '+w
            #print(c)
            if prev[a:].strip()==w.strip() and c not in grnd:
                #print(w,len(w))
                #print(a)
                #print(prev[a:],' |||| ',w)
                #print('repeat')
                #print(s)
                s=s[:-len(w.strip())-1]
                #print(s)
        s+=f+' '
        prev=f
    return s.lower().lstrip().rstrip()

'----------------------------------------------------------------------'
'''
names=[]
grnd_names=[]
with open('filename.txt') as f:
    filelist = f.readlines()

    #print(filelist)
    for name in filelist:
        x=name[-21:-1]
        names.append(x)
        grnd_names.append(x[:-14]+'groundtruth.txt')
#print(grnd_names)
#print(names)

error_sum=0
file=open('child_speech_recognition_evaluation2.txt',"w")
for i in range(len(names)):
    paragraphs=[]
    with open(names[i]) as f:
        paragraphs=f.readlines()
    filtered=[]
    regex=re.compile('\;\d*\.*\d*')
    #remove the ;numbers at the end of each line in transcrip files and save to new file
    for p in paragraphs:
        fil=regex.sub(' ', p)
        fil=re.compile('\n').sub('',fil)
        filtered.append(fil)

    a=remove_grnd_punctuation(grnd_names[i])
    aa=a.split()
    aaa=list_to_string(aa)
    r=remove_ref_punctuation(filtered,aaa)
    rr=r.split()
    rrr=list_to_string(rr)
    x=get_alignment_WER(rrr,aaa) #x[0] total words, x[1] # errors, x[2] WER
    print(grnd_names[i][:-16])
    print('*******************************************************')
    s=grnd_names[i][:-16] +','+str(x[0])+','+str(x[1])+','+str(x[2])+'\n'
    error_sum+=x[2]
    file.write(s)
file.close()
print('average WER: ',error_sum/65)
'''

def trim_query_lines(dir, query_lines, align_parser):
    # create result and directory
    if not os.path.exists(dir):
        os.makedirs(dir)

    for i in range(len(query_lines)-1):
        alignment = align_parser.align(query_lines[i],query_lines[i+1])
        alignment.dump()




def main(argv):

    query_dir = ''     # text to be aligned (e.g., result from speech api)
    ref_dir = ''     # text to be aligned against (e.g., hand transcription)
    result_dir = 'result/'

    # punctuation
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    match = 3
    mismatch = -1
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    sw = swalign.LocalAlignment(scoring,-1.5,-.4)   # you can also choose gap penalties, etc...
                                                    # can play around with values of match, mismatch, and the gaps parameters in localalignment

    try:
        opts, args = getopt.getopt(argv,"hq:r:",["queryDir=","refDir="])
    except getopt.GetoptError:
        print ('text_alignment.py -q <query_dir> -r <ref_dir>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ('text_alignment.py -q <query_dir> -r <ref_dir>')
            sys.exit()
        elif opt in ("-q", "--queryDir"):
            query_dir = arg
        elif opt in ("-r", "--refDir"):
            ref_dir = arg

    print ('Query Directory is ', query_dir)
    print ('Reference Directory is ', ref_dir)

    # create result and directory
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    #
    for root, subdirs, files in os.walk(query_dir):

        for filename in files:

            if filename.endswith('txt'):
                query_file_path = os.path.join(root, filename)
                print (query_file_path)
                ref_file_path = os.path.join(ref_dir, filename.replace('transcript','groundtruth'))
                print (ref_file_path)
                rst_file_path = os.path.join(result_dir, filename.replace('transcript','result'))

                #query_lines=[]
                query_line=''
                ref_line=''
                with open(query_file_path, 'r') as query_file, open(ref_file_path, 'r') as ref_file, open(rst_file_path, 'w+') as rst_file:

                    for line in query_file:
                        line = re.split(';', line)
                        line = regex.sub('', line[0]).rstrip().lower()
                        query_line += line + ' '
                        #query_lines.append(line)

                    #trim_query_lines('query_trimmed', query_lines, sw)


                    for line in ref_file:
                        line = regex.sub('', line).rstrip().lower()
                        ref_line += line + ' '


                    #print query_line
                    #print ref_line

                    alignment=sw.align(query_line,ref_line)
                    x=alignment.match() #x[0] corresponding with ref_line, x[1] corresponding with query_line

                    print('****************************************************************')

                    #print_alignment_withSymbol(x[0],x[1],x[2],sys.stdout)
                    #print_alignment(x[0],x[1],sys.stdout)

                    # changed a and aa to be ref_line_filtered and ref_line_filtered_split
                    # changed b and bb to be query_line_filtered and query_line_filtered_split
                    # changed aligned_act_filtering name to aligned_ref_filtering
                    # changed aligned_rec_filtering name to aligned_query_filtering
                    ref_line_filtered=aligned_ref_filtering(ref_line, x[0])
                    query_line_filtered=aligned_query_filtering(query_line, x[1])
                    print_alignment(ref_line_filtered,query_line_filtered,rst_file)
                    query_line_filtered_split=query_line_filtered.split()
                    ref_line_filtered_split=ref_line_filtered.split()
                    errors=wer(query_line_filtered_split,ref_line_filtered_split)
                    rst_file.write('\n\n\ntotal words: %d' % len(ref_line_filtered_split))
                    rst_file.write('\nERRORS: %d' % errors)
                    rst_file.write('\nWER: %.4f\n' % (errors/float(len(ref_line_filtered_split))))



if __name__ == "__main__":
   main(sys.argv[1:])

