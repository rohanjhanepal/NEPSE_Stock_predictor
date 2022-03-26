from selenium import webdriver
import time
import csv

loc = "C:\\chromedriver"
link = "https://merolagani.com/CompanyDetail.aspx?symbol="
data = []
driver = webdriver.Chrome(loc)

def initialize(symbol = "NHPC"):
    global link
    lin = link + symbol
    driver.get(lin)
    driver.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_CompanyDetail1_lnkHistoryTab\"]").click()
    

def next(page):
    if(page==2):
        driver.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice\"]/div[1]/div[2]/a[5]").click()
        
    else:
        driver.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice\"]/div[1]/div[2]/a[6]").click()
    
        
def save(data , name = "scrapped"):
    with open(name+'.csv' , 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Date" , "Close"])
        for i in data:
            writer.writerow(i)
        print("-------- SAVED -----")
        
def scrap(pages = 5):
    for i in range(pages):
        for j in range(1,100):
            try:
                date = driver.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice\"]/div[2]/table/tbody/tr[{}]/td[2]".format(j+1)).text
                close = driver.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice\"]/div[2]/table/tbody/tr[{}]/td[3]".format(j+1)).text
                data.append([date , float(close)])
                print("date-{} close-{}".format(date , close))
            except:
                print('err')
        next(i)
        time.sleep(1)
        

        
if __name__ == "__main__":
    stock = input("Enter Ticker and Pages separated by space > ").split(" ")
    stock[0].upper()
    initialize(stock[0])
    time.sleep(5)
    scrap(int(stock[1]))
    data.reverse()
    save(data , stock[0])
    driver.quit()
    
