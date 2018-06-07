# -*- encoding:utf-8 -*-
import argparse
import os
import random
import sys

import tensorflow as tf

import jieba
import xlrd
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


def main(_):
    bankPath = FLAGS.input_path
    bank_1_0_path = FLAGS.output_1_0_path
    bank_mix_path = FLAGS.output_mix_path
    heng_number = FLAGS.heng_number
    shu_number = FLAGS.shu_number

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
                                "1 0" + "\t" + nameY.encode("utf8") + "\t" + nameX.encode("utf8") + "\t" + nameList2[
                                    z].encode("utf8") + "\n")

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

    os.system("shuf -n" + str(count) + " " + FLAGS.output_mix_path + " >" + FLAGS.output_shuf_path)


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", default="guangZhouBank.xlsx", help="input_path")
parser.add_argument("--output_1_0_path", default="guangZhouBank_1_0_cut.txt", help="output_1_0_path")
parser.add_argument("--output_mix_path", default="guangZhouBank_mix.txt", help="output_mix_path")
parser.add_argument("--output_shuf_path", default="guangZhouBank_shuf**.txt", help="output_shuf_path")
parser.add_argument("--heng_number", type=int, default=10, help="heng_number")
parser.add_argument("--shu_number", type=int, default=15, help="shu_number")
FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
