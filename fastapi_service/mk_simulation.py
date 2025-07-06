import argparse
import time
import joblib
import logging
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects

dock = 4600
local = 1702

session = Session()

TEST_USERS = [992329,125625,
 800456,428642,712443,
 492414,948586,85734,
 820159,1057088,1185234]




def mk_simulation():
    for usr in TEST_USERS:
        url=f"http://127.0.0.1:{dock}/recommendations/{user_id}"
        response = session.post(url, verify=False)
        print(response.text)
        
        time.sleep(2)

if __name__ == "__main__":
    mk_simulation()
