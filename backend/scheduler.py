import schedule
import time
from scraper import run

def job():
    print("Running scraper...")
    run()

schedule.every().day.at("16:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(30)