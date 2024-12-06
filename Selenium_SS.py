from selenium import webdriver
from selenium.webdriver.common.by import By 
from selenium.webdriver import FirefoxOptions

def driver():
    opts = FirefoxOptions()
    opts.add_argument('--no-sandbox')
    opts.add_argument("--headless")
    opts.add_argument("--incognito")
    opts.add_argument("--disable-notifications")
    # opts.set_preference('prefs', profile)
    # opts.add_argument('--kiosk-printing')
    binary='/home/datascience/Downloads/firefox-106.0.3/firefox/firefox'  ########### PATH TO FIREFOX BINARY
    path="/home/datascience/Desktop/url2txt-DiT/stitched_product/gecko32/geckodriver"  ##########  PATH TO GECKODRIVER
    profile = webdriver.FirefoxProfile()
    profile.set_preference("browser.cache.disk.enable", False)
    profile.set_preference("browser.cache.memory.enable", False)
    profile.set_preference("browser.cache.offline.enable", False)
    profile.set_preference("network.http.use-cache", False) 
    wd = webdriver.Firefox(firefox_binary=binary,options=opts,firefox_profile=profile)
    original_size = wd.get_window_size()
    return wd, original_size

def Take_SS(url,out_path):
    wd,original_size=driver()
    wd.get(url)
    wd.delete_all_cookies()
    required_width = wd.execute_script('return document.body.parentNode.scrollWidth')
    required_height = wd.execute_script('return document.body.parentNode.scrollHeight')
    wd.set_window_size(required_width, required_height)
    wd.find_element(By.TAG_NAME,'body').screenshot(out_path)
    wd.set_window_size(original_size['width'], original_size['height'])
    print("checked screenshot code")