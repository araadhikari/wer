import sys, os, getopt
import string
#from string import whitespace
import numpy
import re
from difflib import SequenceMatcher
import swalign
import textwrap
import distance
import fuzzywuzzy.fuzz
from xlrd import open_workbook
import nltk
from nltk.corpus import stopwords

'''
for the swalign stuff, had to make some changes to the file to fix syntax
like changing xrange to range and including print things in ()
'''


def aligned_rec_filtering(original,rec):
    y=rec.split()
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

def aligned_act_filtering(original, act):
    y=act.split()
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

def wer2(r, h): #r and h are lists
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

def print_alignment2(str1,str2, width=100):
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

        print(line1+'\n')
        print(line2+'\n\n')

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

def print_alignment_withSymbol2(str1,str2, symbol, width=100):

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

        print(line1+'\n')
        print(line2+'\n')
        print(line3+'\n\n')

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


def exact_phrase_matching(child_story,robot_story):
    x=phrases_alignments(child_story,robot_story)
    robot_align_split=x[0]
    child_align_split=x[1]
    #print('$$$$$$$$$$$$$$$$$$$$$$$$$$')
    matches=[]
    substring_matches=[]
    for i in range(len(robot_align_split)):
        #print(robot_align_split[i])
        #print(child_align_split[i])
        #dist= distance.nlevenshtein(child_align_split[i],robot_align_split[i]) #lower score closer to being a better match
        #print('levenshtein distance: ', dist)
        str1=child_align_split[i]
        str2=robot_align_split[i]
        #print(str1) #from robot
        #print(str2) #from child
        x=substringsFinder(str1,str2)
        substring_matches+=x
        #print(x)
        #print('*****')
    return substring_matches


#http://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
def longestSubstringFinder(str1, str2):
    answer = ""

    if len(str1) == len(str2):
        if str1==str2:
            return str1
        else:
            longer=str1
            shorter=str2
    elif (len(str1) == 0 or len(str2) == 0):
        return ""
    elif len(str1)>len(str2):
        longer=str1
        shorter=str2
    else:
        longer=str2
        shorter=str1

    matrix = numpy.zeros((len(shorter), len(longer)))

    for i in range(len(shorter)):
        for j in range(len(longer)):
            if shorter[i]== longer[j]:
                matrix[i][j]=1

    longest=0

    start=[-1,-1]
    end=[-1,-1]
    for i in range(len(shorter)-1, -1, -1):
        for j in range(len(longer)):
            count=0
            begin = [i,j]
            while matrix[i][j]==1:

                finish=[i,j]
                count=count+1
                if j==len(longer)-1 or i==len(shorter)-1:
                    break
                else:
                    j=j+1
                    i=i+1

            i = i-count
            if count>longest:
                longest=count
                start=begin
                end=finish
                break

    answer=shorter[int(start[0]): int(end[0])+1]
    return answer

def substringsFinder(str1,str2,len_min=3):
    substrings = []
    ## rsplit(' ',1)[0] to help hopefully filter out cut off words
    ### example: s taken out of stuck b/c it matched 'his head s' for 'his head so' and 'his head stuck'
    #print('substring finder')
    #print(str1)
    #print(str2)
    x = longestSubstringFinder(str1+' ', str2+' ').rsplit(' ',1)[0]
    while len(x) >= len_min:
        substrings.append(x)
        #print(x)

        str1 = re.sub(x, ' ', str1)
        str2 = re.sub(x, ' ', str2)
        #print(str1)
        #print(str2)
        x = longestSubstringFinder(str1, str2).rsplit(' ',1)[0]
        #print(x)
    output=[]
    for s in substrings:
        ss=re.sub('  +','',s).strip()
        #print(ss)
        if ss!='' and len(ss)>=len_min:
            output.append(ss)
    #print(output)

    ###add filtering of the list to remove single words and things like that
    ##seems like it works out better removing anything less than 2 words...there are sometimes ????
    output2=[]
    for i in range(len(output)):
        tokenized=output[i].split()
        #print(tokenized)
        length=len(tokenized)
        if length>2:
            output2.append(output[i])
    return(output2)

def phrases_alignments(child_story,robot_story):
    storya_filename = robot_story
    # storya_filename = 'cyber4_robot_story_A.txt'
    # storyb_filename = 'cyber4_robot_story_B.txt'
    # storyf_filename = 'cyber4_robot_story_full.txt'
    # storyq_filename = 'cyber4_robot_story_questions.txt'
    storya = []

    with (open(storya_filename, encoding='cp437')) as z:
        x = z.read()
        for c in string.punctuation:
            x = x.replace(c, '')
        s = x.splitlines()
        for ss in s:
            if ss != '':
                storya.append(ss.lower())
    # print(storya)
    # words = stopwords.words('english')
    # print(words)
    robot_story_string1 = list_to_string(storya)
    robot_story_token = robot_story_string1.split()
    filtered_words = [word for word in robot_story_token if word not in stopwords.words('english')]
    #print('robot story: ',list_to_string(filtered_words))
    robot_story_string=list_to_string(filtered_words)
    print('Robot story: ', robot_story_string)

    match = 3
    mismatch = -1
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    sw = swalign.LocalAlignment(scoring, -1.5, -.4)  # you can also choose gap penalties, etc...
    # can play around with values of match, mismatch, and the gaps parameters in localalignment

    ref_lines = []
    with open(child_story, encoding='cp437') as z:
        lines = z.readlines()
        # print(lines)
        for i in range(len(lines)):
            x = lines[i].rstrip('\n')
            if x != '':
                for c in string.punctuation:
                    # if c=='.' or c=='?' or c==',' or c=='!':
                    # x=x.replace(c,'$')
                    x = x.replace(c, '')
                ref_lines.append(x.lower())
    child1 = list_to_string(ref_lines)+' '
    child_story_token = child1.split()
    filtered_words2 = [word for word in child_story_token if word not in stopwords.words('english')]
    child=list_to_string(filtered_words2)
    robot = robot_story_string+' '
    print('Child story: ', child)

    print('------------------------')
    alignment = sw.align(child, robot)
    # x=alignment.dump()
    x = alignment.match()  # x[0] is robot x[1] is child
    # print(x[0])
    robot_align = aligned_act_filtering(robot, x[0])
    child_align = aligned_rec_filtering(child, x[1])
    # print(robot_align)
    # print(child_align)
    print_alignment2(robot_align, child_align)

    # split based on robot align
    robot_align_split = re.split("   +", robot_align)
    # print(robot_align_split)
    # print(robot_align)
    # print(len(robot_align))
    # print(len(child_align))
    robot_split_index = [0]
    prev_index = 0
    for phrase in robot_align_split:
        # print('!@!#@$^$^&#%&*&*')
        # print('phrase ', phrase)
        # print(child_align[prev_index:])
        index = robot_align[prev_index:].find(phrase)
        # print('index ',index)
        # print('length of phrase', len(phrase))
        prev_index += len(phrase) + index
        robot_split_index.append(prev_index)
    # print(split_index)
    # print(child_align)
    # print(child_align[0:split_index[-1]])
    child_align_split = []
    # print(split_index[:-1])
    index_tracker = robot_split_index
    start_ind = index_tracker[0]
    end_ind = index_tracker[1]
    # trying to handle split between sentences with the end_ind_space
    for index in robot_split_index[:-2]:
        # #end_ind_space = child_align.index(" ", end_ind)
        # print(end_ind)
        # #print(end_ind_space)
        # #phrase = child_align[start_ind:end_ind_space]
        # phrase = child_align[start_ind:end_ind]
        #
        # print('phrase ',phrase)
        # child_align_split.append(phrase)
        # # print(phrase)
        # index_tracker.pop(0)
        # start_ind = index_tracker[0]
        # if len(index_tracker) == 1:
        #     end_ind = len(child_align)
        # else:
        #     end_ind = index_tracker[1]
        end_ind_space=child_align.index(" ",end_ind-1)
        #print(end_ind)
        #print(end_ind_space)
        phrase=child_align[start_ind:end_ind_space]
        #print('phrase: ',phrase)
        child_align_split.append(phrase)
        index_tracker.pop(0)
        start_ind=end_ind_space
        end_ind=index_tracker[1]
    child_align_split.append(child_align[end_ind_space:])

    # for i in range(len(robot_align_split)):
    #     print(robot_align_split[i])
    #     print(child_align_split[i])
    return (robot_align_split,child_align_split)

def similar_phrase_matching(child_story,robot_story):
    x=phrases_alignments(child_story,robot_story)
    robot_align_split=x[0]
    child_align_split=x[1]
    # print('$$$$$$$$$$$$$$$$$$$$$$$$$$')
    fuzzy_matches = []
    for i in range(len(robot_align_split)):
        # print(robot_align_split[i])
        # print(child_align_split[i])
        # dist= distance.nlevenshtein(child_align_split[i],robot_align_split[i]) #lower score closer to being a better match
        # print('levenshtein distance: ', dist)
        str1 = child_align_split[i]
        str2 = robot_align_split[i]

        #print(str2)  # from robot
        #print(str1)  # from child

        str1_split=re.split(" ", str1) #child words
        str2_split=re.split(" ", str2) #robot words
        str1_split_filtered=[]
        str2_split_filtered=[]
        for word in str1_split:
            if word!='':
                str1_split_filtered.append(word)
        for word in str2_split:
            if word!='':
                str2_split_filtered.append(word)
        #print(' robot ',str2_split_filtered)
        #print(' child ',str1_split_filtered)


        fuzzy=fuzzywuzzy.fuzz.token_sort_ratio(str1,str2)
        #print('fuzz token ratio: ',fuzzy)



        if fuzzy >=50 :
            ###work on getting the number of similar words between the two also, not just the fuzzywuzzy token ratio????
            match_count=0
            len2=len(str2_split_filtered)
            len1=len(str1_split_filtered)
            #print('len robot phrase ', len(str2_split_filtered))
            #print('len child phrase ', len(str1_split_filtered))
            for word in str2_split_filtered:
                if len(str1_split_filtered)!= 0:
                    if word in str1_split_filtered:
                        str1_split_filtered.remove(word)
                        match_count+=1
            #print('match count: ', match_count)
            #print('*****')
            fuzzy_matches.append((str2,str1,len2,len1,match_count,fuzzy))
            #fuzzy_matches.append((str2, str1, len2, len1, match_count))

    return fuzzy_matches



def main(argv):
    phrase_matching_file=open('matches.csv',"w")
    child_story_file='CYBER4-P003-Y-c_2storytellingChanges.txt'
    robot_story_file='cyber4_robot_story_A.txt'

    wb = open_workbook('Cyber4_Sheet_storyab.xlsx')
    for s in wb.sheets():
        # print 'Sheet:',s.name
        values = []
        for row in range(s.nrows):
            col_value = []
            for col in range(s.ncols):
                value = (s.cell(row, col).value)
                try:
                    value = str(int(value))
                except:
                    pass
                col_value.append(value)
            values.append(col_value)
    values.pop(0)
    #print (values)

    matches=exact_phrase_matching(child_story_file,robot_story_file)
    matches_string=''
    for m in matches:
        print(m)
        matches_string+=m+','
    phrase_matching_file.write(child_story_file+','+robot_story_file+','+matches_string)

    phrase_matching_file.close()
    matches=similar_phrase_matching(child_story_file,robot_story_file)
    for m in matches:
        print(m)

    '''

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

                query_lines=[]
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
                    x=alignment.match() #x[0] act x[1] rec

                    print('****************************************************************')

                    #print_alignment_withSymbol(x[0],x[1],x[2],sys.stdout)
                    #print_alignment(x[0],x[1],sys.stdout)
                    a=aligned_act_filtering(ref_line, x[0])
                    b=aligned_rec_filtering(query_line, x[1])
                    print_alignment(a,b,rst_file)
                    bb=b.split()
                    aa=a.split()
                    x=wer(bb,aa)
                    rst_file.write('\n\n\ntotal words: %d' % len(aa))
                    rst_file.write('\nERRORS: %d' % x)
                    rst_file.write('\nWER: %.4f\n' % (x/float(len(aa))))
                    #return(len(aa),x,x/len(aa))

    '''


if __name__ == "__main__":
   main(sys.argv[1:])

