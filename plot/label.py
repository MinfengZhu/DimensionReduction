fr = open("twitter.2d.txt","r")
fo = open("twitter.2d.label.txt","w")
length = fr.readline().split(' ')[0]
print length
for i in range(1,int(length)):
    fo.write('0')
    fo.write('\n')
fo.close()
fr.close()

