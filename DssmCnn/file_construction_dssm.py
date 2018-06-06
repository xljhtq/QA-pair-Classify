# -*- encoding:utf-8 -*-
import random

import jieba
import xlrd
import json
import re


def read_excel(file):
    bk = xlrd.open_workbook(file)
    sh = bk.sheet_by_name('sheet1')
    standardData = []
    dic = {}
    count = 0
    row = 0
    for row in range(1, sh.nrows):
        # guangFaFAQ
        # count += 1
        # if count == 1: continue
        standard = sh.cell_value(row, 3)
        extendedQ = sh.cell_value(row, 4)
        if standard and extendedQ:
            extended = extendedQ.split("\n")
            for extend in extended:
                if extend == "": continue
                if standard.encode("utf8") in dic:
                    dic[standard.encode("utf8")].append(extend.encode("utf8"))
                else:
                    standardData.append(standard.encode("utf8"))
                    dic[standard.encode("utf8")] = [extend.encode("utf8")]
        else:
            print(1)
    print(row)
    return standardData, dic


# bankPath = "zhaoHangFAQ/zhaoShangFAQ.xlsx"
# bank_1_0_path = "dssmCnn/zhaoHangFAQ_1_0_cut.txt"
# bank_mix_path = "dssmCnn/zhaoHangFAQ_mix**.txt"
# bankPath = "nanjingFAQ/nanjingFAQ.xlsx"
# bank_1_0_path = "dssmCnn/nanjingFAQ_1_0_cut.txt"
# bank_mix_path = "dssmCnn/nanjingFAQ_mix**.txt"
bankPath = "guangZhouBank.xlsx"
bank_1_0_path = "guangZhouBank_1_0_cut.txt"
bank_mix_path = "guangZhouBank_mix**.txt"
heng_number = 10
shu_number = 15


standardData, dic = read_excel(bankPath)
dic_extract = {}
standardData_extract = []
for standard in standardData:
    standard2 = re.sub(u"[A-Za-z0-9\!\%\[\]\...\，\。\?\-\/\？\,\！\、\：\）\（\“\”\:\+\~\…\·\；\】\◆\�\#\)\"]", "",
                      standard.decode("utf8"))
    standard_1 = ' '.join(jieba.lcut(standard2))
    if standard_1 not in dic_extract:
        dic_extract[standard_1] = [standard_1]
        standardData_extract.append(standard_1)
    for x in dic[standard]:
        xx = re.sub(u"[A-Za-z0-9\!\%\[\]\...\，\。\?\-\/\？\,\！\、\：\）\（\“\”\:\+\~\…\·\；\】\◆\�\#\)\"]", "",
                    x.decode("utf8"))
        xx_1 = ' '.join(jieba.lcut(xx))
        dic_extract[standard_1].append(xx_1)

print(len(standardData_extract))
print(len(dic_extract))

# 生成1_0数据
print("-----------------------生成1_0数据--------------------------")
with open(bank_1_0_path, "w") as out_op:
    for i in range(len(standardData_extract)):
        print(i)
        nameList = dic_extract[standardData_extract[i]]
        for x in range(len(nameList)):
            nameX = nameList[x]
            count = len(nameList)
            if len(nameList) - x >= heng_number:
                count = min(x + heng_number + 1, count)
            for y in range(x + 1, count):
                nameY = nameList[y]
                for j in range(i + 1, len(standardData_extract)):
                    nameList2 = dic_extract[standardData_extract[j]]
                    listz = range(len(nameList2))
                    if len(listz) >= shu_number:
                        elementZ = random.sample(listz, shu_number)
                    else:
                        elementZ = random.sample(listz, len(listz))
                    for z in elementZ:
                        out_op.write(
                            "1 0" + "\t" + nameY.encode("utf8") + "\t" + nameX.encode("utf8") + "\t" + nameList2[z].encode("utf8") + "\n")

# 分词
# print("---------------------------分词-------------------------------------")
# with open(bank_1_0_cut_path, 'w') as out_op:
#     for line2 in open(bank_1_0_path):
#         line = line2.strip().strip('\n').split('\t')
#         if len(line) != 4:
#             print(line2)
#             continue
#         x1 = re.sub(u"[A-Za-z0-9\!\%\[\]\...\，\。\?\-\/\？\,\！\、\：\）\（\“\”\:\+\~\…\·\；\】\◆\�\#\)\"]", "",
#                     line[1].decode("utf8"))
#         x2 = re.sub(u"[A-Za-z0-9\!\%\[\]\...\，\。\?\-\/\？\,\！\、\：\）\（\“\”\:\+\~\…\·\；\】\◆\�\#\)\"]", "",
#                     line[2].decode("utf8"))
#         x3 = re.sub(u"[A-Za-z0-9\!\%\[\]\...\，\。\?\-\/\？\,\！\、\：\）\（\“\”\:\+\~\…\·\；\】\◆\�\#\)\"]", "",
#                     line[3].decode("utf8"))
#         if x1 and x2 and x3:
#             slist_1 = jieba.cut(x1)
#             slist_2 = jieba.cut(x2)
#             slist_3 = jieba.cut(x3)
#             result = line[0] + "\t" + ' '.join(slist_1) + "\t" + ' '.join(slist_2) + "\t" + ' '.join(slist_3) + "\n"
#             out_op.write(result.encode("utf8"))

# 打乱1_0成0_1数据
print("----------------------------打乱1_0成0_1数据-----------------------------")
with open(bank_mix_path, "w") as out_op:
    count = 0
    for line2 in open(bank_1_0_path):
        count += 1
        line = line2.strip().strip("\n").split("\t")
        if len(line) != 4:
            print(line2)
            continue
        left = line[1]
        centre = line[2]
        right = line[3]
        if count % 2 == 0:
            out_op.write(line2)
        else:
            out_op.write("0 1" + "\t" + right + "\t" + centre + "\t" + left + "\n")
