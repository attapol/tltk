#-*- coding: UTF-8 -*-

import tltk
import sys


print(tltk.nlp.spell_candidates('สงฆ'))

Input = "นายกรัฐมนตรีกล่าวกับคนขับรถประจำทางหลวงสายสองว่า อยากวิงวอนให้ใช้ความรอบคอบอย่าหลงเชื่อคำชักจูงหรือปลุกระดมของพวกหัวรุนแรงจากทางการไฟฟ้า"
print("syllable segmented",tltk.nlp.syl_segment(Input))
print("max coll",tltk.nlp.word_segment(Input))
print("max match",tltk.nlp.word_segment_mm(Input))
print("\n")

#Input= "ทางการเมืองสมุทรปราการได้สรุปว่า คะแนนเฉลี่ยของพฤติกรรมการใช้ยาตามเกณฑ์การรักษามีค่าสูงส่งผลให้การตีความเป็นไปได้ยาก"
#print("max coll",tltk.nlp.word_segment(Input))
#print("max match",tltk.nlp.word_segment_mm(Input))

#tltk.nlp.reset_thaidict()

#tltk.nlp.read_thaidict('BEST.dict')



#Input=u"สรุปได้ว่า ผลการวิเคราะห์เปรียบเทียบคะแนนเฉลี่ยของพฤติกรรมการใช้ยาตามเกณฑ์การรักษาในกลุ่มทดลองระหว่างก่อนและหลังการทดลอง พบว่าค่าเฉลี่ยของพฤติกรรมการใช้ยาตามเกณฑ์การรักษาสูงกว่าก่อนการทดลองอย่างมีนัยสำคัญทางสถิติที่ระดับ 0.05 ซึ่งเป็นไปตามสมมุติฐานที่ 1 อันเนื่องจากประสิทธิภาพของงานวิจัย ผลของโปรแกรมการให้คำปรึกษาครอบครัวร่วมกับการให้สุขภาพจิตศึกษาต่อพฤติกรรมการใช้ยาตามเกณฑ์การรักษาของผู้ป่วยจิตเภทในชุมชน ซึ่งผู้วิจัยจึงนำแนวคิดการให้คำปรึกษาครอบครัวกลุ่มโครงสร้างครอบครัว (Structural Family Therapy) ของ Minuchin (1974) ร่วมกับสุขภาพจิตศึกษาตามแนวคิดของ Anderson, Hongarty and Reiss (1980) ส่งผลให้สามารถพัฒนาสัมพันธภาพในระบบโครงสร้างครอบครัวที่เหมาะสม สามารถพัฒนาความรู้ความเข้าใจในการปฏิบัติตนของผู้ป่วยจิตเภท และท้ายที่สุ
            
tltk.corpus.load3gram('TNC.3g')

w1= 'จะ'
w2 = 'ไป'

print(tltk.corpus.bigram(w1,w2))


w = 'แดง'
col = tltk.corpus.collocates(w,'mi','both',2,100,1)
for ((x,y),s) in col:
    print(x,y,s)
    

    
sys.exit()
