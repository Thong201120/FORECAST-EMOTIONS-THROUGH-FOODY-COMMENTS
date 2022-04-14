import requests
from bs4 import BeautifulSoup
import csv
urls = ["https://www.foody.vn/thuong-hieu/phuc-long-coffee-tea-express?c=ho-chi-minh", "https://www.foody.vn/thuong-hieu/tous-les-jours-sg?c=ho-chi-minh", "https://www.foody.vn/thuong-hieu/kfc-ho-chi-minh?c=ho-chi-minh", "https://www.foody.vn/thuong-hieu/tra-sua-bobapop?c=ho-chi-minh", "https://www.foody.vn/thuong-hieu/highlands-coffee?c=ho-chi-minh", "https://www.foody.vn/thuong-hieu/starbucks-coffee-ho-chi-minh?c=ho-chi-minh", "https://www.foody.vn/thuong-hieu/lotteria-ga-ran-sai-gon?c=ho-chi-minh"]
def crawl(url):
    req = requests.get(url).content
    soup = BeautifulSoup(req, "html.parser")
    articles = soup.find("body")
    contenthtml = str(soup.prettify())
    listlink = []
    for div in articles.find_all("h2"):
        a = div.find("a")["href"]
        listlink.append(a)

    listlink = listlink[0:len(listlink)-1]

    all_list_texts = []
    for link in listlink:
        link = "https://www.foody.vn" + str(link) + "/binh-luan"
        req1 = requests.get(link).content
        soup1 = BeautifulSoup(req1, "html.parser")
        articles2 = soup1.find("div", class_="micro-right1000")
        divs = articles2.find_all("div", class_="rd-des toggle-height")
        points = articles2.find_all("div", class_="review-points ng-scope green")
        list_texts = []
        for div in divs:
            text = div.find("span").text
            text = str(text).replace("\n", "")
            text = str(text).replace("\t", "")
            text = str(text).replace("\r", "")
            list_texts.append(text)
        points = articles2.find_all("div", class_="review-points")
        list_points = []
        for point in points:
            pt = point.find("span").text
            if str(pt).startswith('{'):
                continue
            else:
                list_points.append(pt)
        m = []
        for i in range(len(list_texts)):
            m.append((list_points[i]))
            m.append(list_texts[i])
        all_list_texts = all_list_texts + m
    return all_list_texts

datas = []
for url in urls:
    datas = datas + crawl(url)

cmt = []
pts = []
print(len(datas))
for i in range(0, len(datas) - 1, 2):
    pts.append(datas[i])
    cmt.append(datas[i+1])
cmtNegative, cmtNeutral, cmtPositive = [], [], []
ptsNegative, ptsNeutral, ptsPositive = [], [], []
for i in range(len(cmt)):
    if float(pts[i]) <= 5:
        # Negative.append(pts[i])
        cmtNegative.append(cmt[i])
        ptsNegative.append(pts[i])

    elif (float(pts[i]) > 5) and (float(pts[i]) <= 6):
        # Neutral.append(pts[i])
        cmtNeutral.append(cmt[i])
        ptsNeutral.append(pts[i])
    else:
        # Positive.append(pts[i])
        cmtPositive.append(cmt[i])
        ptsPositive.append(pts[i])

LabelNegative = [-1 ]* len(cmtNegative)
LabelNeutral = [0] * len(cmtNeutral)
LabelPositive = [1] * len(cmtPositive)

def JoinList(Listcmt, Listpts):
    List_data = []
    for i in range(len(Listcmt)):
        List_data.append(Listcmt[i])
        List_data.append(Listpts[i])
    return List_data
Full_Clean_Data = []
Full_Clean_Data = JoinList(cmtNegative, LabelNegative) + JoinList(cmtNeutral, LabelNeutral) + JoinList(cmtPositive, LabelPositive)

list_test1=[]
for i in range(int(len(Full_Clean_Data)/2)):
    list_test2=[]
    list_test2.append(Full_Clean_Data[2*i])
    list_test2.append(Full_Clean_Data[2*i+1])
    list_test1.append(list_test2)
import random
list_test=[]
list_test=random.sample(list_test1,len(list_test1))
# print(list_test)

len_data=len(list_test)
# print("\n",len_data)

f_train = open("./Data/train.txt", 'w', encoding='utf-8')

count_train = 0
for i in range(len(list_test)):
    f_train.write("train_"+str(count_train)+"\n")
    f_train.write('"' + str(list_test[i][0]) + '"\n')
    f_train.write(str(list_test[i][1]) + "\n\n")
    count_train+=1
f_train.close()