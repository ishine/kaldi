# coding=utf-8
import codecs
import sys

wdict={}
wmaxlen=0

def loaddict(dictfile):
	global wdict
	global wmaxlen
	INPUT_FILE=dictfile
	file_i=codecs.open(INPUT_FILE,"r","utf-8")

	for line in file_i:
		line=line.strip("\r\n")
		
		line=line.split()
	#	len(line)
		line = line[0]
	#	print line
		
		
		#specific mission code
		wdict[line]=""
		if(len(line)>wmaxlen):
			wmaxlen=len(line)
	file_i.close()
	print "load dict complete: "+str(len(wdict))+"/"+str(wmaxlen)

#Forward Maximum Matching
#可以用后缀树来优化
def fmm(sentence):
	global wmaxlen
	result=[]
	while True:
		slen=len(sentence)
		wlen = len("".join(sentence[0:slen])) 
		#print 'wlen',wlen
		if(wlen>wmaxlen):
		#	slen=wmaxlen
			wlen1 = 0
			slen = 0
			while wlen1 < wmaxlen:
				slen += 1
				wlen1 = len("".join(sentence[0:slen])) 
				#print 'slen',slen
			slen -= 1
		#print slen
		word=sentence[0:slen]
		mergW = "".join(word)
		isw=isword(mergW)
	 	#print slen
		while slen>1 and isw==False:
			slen-=1
			mergW = "".join(word[0:slen])
			isw=isword(mergW)
		result.append(mergW)
		#print mergW,slen
		sentence=sentence[slen:]
		if(slen==0):
			break
	return result

#Reverse Maximum Matching
def rmm(sentence):
	global wmaxlen
	result=[]
	while True:
		slen=len(sentence)
		wlen = len("".join(sentence[0:slen])) 
		if(wlen>wmaxlen):
		#	slen=wmaxlen
			wlen1 = 0
			slen = 0
			while wlen1 < wmaxlen:
				slen += 1
				wlen1 = len("".join(sentence[-slen:])) 
			#	print 'slen',slen
			slen -= 1
		word=sentence[-slen:]

		mergW = "".join(word)
		#print mergW
		isw=isword(mergW)
		while slen>1 and isw==False:
			slen-=1
			mergW = "".join(word[-slen:])
			isw=isword(mergW)
		result.append(mergW)
		sentence=sentence[0:-slen]
		#print sentence
	#	print slen,mergW
		if(slen==0):
			break
	result.reverse()
	return result

def isword(ss):
	global wdict
	if(ss in wdict):
		return True
	else:
		return False

def mergeseg(sentence,osegs,nsegs):
	#为了方便起见，使用数组记录字的类型，之后合并相同类型的字
	result_tags=[]
	i=1
	s=0
	for seg in osegs:
		l=len(seg)
		for j in range(s,s+l):
			result_tags.append(i)
		i+=1
		s+=l
	i=-1
	s=0
	for seg in nsegs:
		l=len(seg)
		if(l==1):
			s+=l
			continue
		for j in range(s,s+l):
			result_tags[j]=i
		i-=1
		s+=l
	#根据result_tags输出分词结果
	lasttag=result_tags[0]
	mergedseg=[]
	word=""
	i=0
	for tag in result_tags:
		if(tag==lasttag):
			word+=sentence[i]
		else:
			mergedseg.append(word)
			word=sentence[i]
			lasttag=tag
		i+=1
	if(word!=""):
		mergedseg.append(word)
	return mergedseg

#返回分词结果的切分点的index
#i位置字的前面为切分位置
def getcutindex(segs):
	cutpots=[]
	i=0
	cutpots.append(i)
	l=len(segs)
	for j in range(l-1):
		seg=segs[j]
		l=len(seg)
		i+=l
		cutpots.append(i)
	return cutpots

#两个集合set1,set2中元素从小到大排列
#计算其中的元素的相同的个数
def countsetdiff(set1,set2):
	i1=0
	i2=0
	l1=len(set1)
	l2=len(set2)
	eq_num=0
	while(True):
		#print "set1/set2: "+str(set1[i1])+"/"+str(set2[i2])
		if(set1[i1]==set2[i2]):
			eq_num+=1
			i1+=1
			i2+=1
			if(i1>=l1):
				break
			if(i2>=l2):
				break
		elif(set1[i1]<set2[i2]):
			i1+=1
			if(i1>=l1):
				break
		else:
			i2+=1
			if(i2>=l2):
				break
	#print eq_num
	return eq_num

#nsegs_f与nsegs_r，越少干涉原有的越好
def chooseseg(osegs,segs_f,segs_r):
	cutpots_o=getcutindex(osegs)
	cutpots_f=getcutindex(segs_f)
	cutpots_r=getcutindex(segs_r)
	eq_f=countsetdiff(cutpots_o,cutpots_f)
	eq_r=countsetdiff(cutpots_o,cutpots_r)
	if(eq_f>eq_r):
		return segs_f
	else:
		return segs_r

#最好的办法是使用得分+解码
#得分要考虑上下文……这不就成了crf了么……
def process(inputfile,outputfile,add,OOV_PERCENT,SINGLE_WORD_PERCENT):
	file_i=codecs.open(inputfile,"r","utf-8")
	file_o=codecs.open(outputfile,"w","utf-8")
	linenum=0 
	for line in file_i:
		count1 = 0;
		count2 = 0;
		oov_percent = 0;
		single_word_percent = 0;
		linenum+=1
		#print linenum
		if(linenum%10000==0):
			print "line processed: "+str(linenum)
		line=line.strip("\r\n")
		line=line.strip()
		if line == '':
			continue
		osegs=line.split(" ")
		sentencemerge="".join(osegs)
		sentence = osegs
		#按照最大匹配算法使用给定的新词词典进行分词
		nsegs_f=fmm(sentence)
		nsegs_r=rmm(sentence)
		segs_f=mergeseg(sentencemerge,osegs,nsegs_f)
		segs_r=mergeseg(sentencemerge,osegs,nsegs_r)
		mergedseg=chooseseg(osegs,segs_f,segs_r)
		if OOV_PERCENT != 1 or SINGLE_WORD_PERCENT != 1:
			for mergword in mergedseg:
				print mergword
				if len(mergword) == 1:
					count2 += 1
					if mergword not in wdict:
						count1 += 1
			oov_percent = float(count1)/float(len(mergedseg))
			single_word_percent = float(count2)/float(len(mergedseg))
	#	print oov_percent,single_word_percent
		if oov_percent <= OOV_PERCENT and single_word_percent <= SINGLE_WORD_PERCENT:

			outputline=" ".join(mergedseg)
			#outputline=" ".join(segs_f)
			#print outputline	
			if  add == "AddMark":
				file_o.write("<s> "+outputline+ " </s>"+"\n")
			else:
				file_o.write(outputline+"\n")
		#break
	#print type(OOV_PERCENT),type(SINGLE_WORD_PERCENT)
	file_i.close()
	file_o.close()

if __name__=="__main__":
	#print len(sys.argv)
	if(len(sys.argv) < 4 ):
		print("USAGE:mergenewword.py wdict inputfile outputfile [AddMark] [oov_percent] [single_word_percent]")
		sys.exit(1)
	print("process begin")
	loaddict(sys.argv[1])
	if (len(sys.argv) == 4):
		process(sys.argv[2],sys.argv[3],"",float(1),float(1))
	if (len(sys.argv) == 5):
		process(sys.argv[2],sys.argv[3],sys.argv[4],float(1),float(1))
	if (len(sys.argv) == 6):
		process(sys.argv[2],sys.argv[3],sys.argv[4],float(sys.argv[5]),float(1))
	if (len(sys.argv) == 7):
		process(sys.argv[2],sys.argv[3],sys.argv[4],float(sys.argv[5]),float(sys.argv[6]))
	#process("test.txt","test.t.txt")
	print("process complete!")