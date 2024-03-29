import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.edge.service import Service as EdgeService
from typing import Callable
from multiprocessing import Pool

def get_random_candidates() -> list[str]:
    xpaths = []
    for i in range(161, 180+1):
        if i == 179: continue
        xpaths.append(f'//input[@value="{i}"]')
    random.shuffle(xpaths)
    xpaths = xpaths[:9]
    xpaths.append('//input[@value="179"]')
    return xpaths

def start_driver(infer: Callable[[bytes], str]) -> None:
    EdgeOptions = webdriver.EdgeOptions()
    EdgeOptions.add_argument("--headless")
    EdgeOptions.add_argument('log-level=3')
    with webdriver.Edge(options=EdgeOptions, service=EdgeService(EdgeChromiumDriverManager().install())) as driver:
        driver.get("http://tainangtrevietnam.vn/index.html")
        driver.execute_cdp_cmd('Network.setBlockedURLs', {"urls": ["www.google-analytics.com", "www.googletagmanager.com"]})
        driver.execute_cdp_cmd('Network.enable', {})

        for i in range(100):
            time.sleep(5)
            for xpath in get_random_candidates():
                check_box = driver.find_element(by = By.XPATH, value = xpath)
                check_box.click()
            captcha_image = driver.find_element(by = By.XPATH,
                value = r'//*[@id="thongtin"]/div/div[3]/div[2]/div/img')
            image = captcha_image.screenshot_as_png
            result = infer(image)

            text_box = driver.find_element(by = By.XPATH,
                value = r'//input[@class="inputVote"]')
            text_box.send_keys(result)

            submit_button = driver.find_element(by = By.XPATH,
                value = r'//*[@id="ctl00_webPartManager_wp793523384_wp1892398315_btnSubmit1"]')
            submit_button.click()

            alert = driver.switch_to.alert
            print(f"Result: {alert.text}")
            if "2022" not in alert.text:
                with open(f"fail_log/{result}_FAIL_{i}.png", "wb") as file:
                    file.write(image)
            alert.accept()
            driver.delete_all_cookies()

def start(infer: Callable[[bytes], str]) -> None:
    pool_size = 8
    with Pool(processes=pool_size) as pool:
        pool.map(start_driver, [infer] * pool_size)
