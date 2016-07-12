import numpy
import re
from difflib import SequenceMatcher
import swalign

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

def print_alignment(str1,str2):
    l1=len(str1)
    l2=len(str2)
    i=0
    while i<min(l1-150,l2-150):
        print(str1[i:i+150])
        print(str2[i:i+150])
        print('')
        i+=150
    print(str1[i:])
    print(str2[i:])
    

def get_alignment_WER(recognized,actual): #recognized and act are strings
    r=recognized.split()
    for i in range(len(r)):
        if '-' in r[i]:
            r[i]=re.compile('.+\-').sub('',r[i])
    recognized=list_to_string(r)
    #print(actual)
    #print(recognized)
    match = 3
    mismatch = -1
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    sw = swalign.LocalAlignment(scoring,-1.5,-.4)  # you can also choose gap penalties, etc...
    #can play around with values of match, mismatch, and the gaps parameters in localalignment
    alignment=sw.align(recognized,actual)
    x=alignment.match() #x[0] act x[1] rec
   
    #print_alignment(x[0],x[1])
    
##    bb=x[1].split()
##    aa=x[0].split()
##               
##    w=wer(x[1],x[0])
##    print('ERRORS: ',w)
##    print('total words: ',len(x[0].split()))
##    print('WER: ', w/len(x[0].split()))
    print('****************************************************************')

    a=aligned_act_filtering(actual, x[0])
    b=aligned_rec_filtering(recognized,x[1])
    print_alignment(a,b)
    bb=b.split()
    aa=a.split()           
    x=wer(bb,aa)
    print('total words: ',len(aa))
    print('ERRORS: ',x)
    print('WER: ', x/len(aa))
    return(len(aa),x,x/len(aa))
    
    
'-------------------------------------------------------------------------'

def remove_grnd_punctuation(filename): #groundtruth file name
    reg123=['\'','\"','\.*','\,','\?','\!','\-$']
    reg_list=[]
    for r in reg123:
        reg_list.append(re.compile(r))
    with open(filename) as file:
        paragraphs=file.readlines()
        filtered=[]
        for p in paragraphs:
            fil=p
            for regex in reg_list:
                fil=regex.sub('', fil)
            regex=re.compile('\- ')
            fil=regex.sub(' ',fil)
            filtered.append(fil)
        s=''
        for f in filtered:
            f = f.lstrip().rstrip().rstrip('\n')
            if f == '':
                continue
            s+=f+' '
        return s.lower().lstrip().rstrip()

    
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
