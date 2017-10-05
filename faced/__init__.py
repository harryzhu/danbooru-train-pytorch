import os
import time
import logging

TIMEYMD = time.strftime("%Y%m%d", time.localtime())
LOGFILE = os.path.join("logs","log_faced_" +str(TIMEYMD)+".log")
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename= LOGFILE,
                    filemode='a')
