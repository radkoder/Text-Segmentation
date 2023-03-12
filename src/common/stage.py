import time, functools
from math import floor
from collections import deque
def stostr(sec: int) -> str:
    h = sec/3600
    hfin = floor(h)

    min = (h-hfin)*60
    minfin = floor(min)

    s = (min - minfin)*60
    sfin = floor(s)
    return f'{hfin}h {minfin}min {sfin}s'


class ProgressBar:
    def __init__(self, title, max) -> None:
        self.barlen = 10
        self.title = title
        self.max = max
        self.count = 0
        self.meanTime = 0
        self.totalTime = 0
        self.diffs = deque(maxlen=15)
        self.lastTime = 0
        self.currTime = time.time()

    def update(self,status, currCount = None, count_delta = 1) -> None:
        self.lastTime = self.currTime
        self.currTime = time.time()
        self.totalTime += self.currTime - self.lastTime
        self.diffs.append(self.currTime - self.lastTime)
        self.meanTime = sum(self.diffs)/len(self.diffs)
        if currCount == None:
            self.count += count_delta
        else:
            self.count = currCount
        self.print_bar(status)

    def print_bar(self,status) -> None:
        r = int(self.count*self.barlen/self.max)
        s = '.'*r + ' '*(self.barlen-r)
        outstr = f'{self.title}:[{s}][{self.count}/{self.max}] => {status} [{stostr(int(self.totalTime))}][eta: {stostr(int(self.meanTime*(self.max-self.count)))}]'
        print(outstr+'   ',end='\r')

    def end(self) -> None:
        self.print_bar("DONE")
        print('')


def measure(name):
    def decorator(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            print(name + "...", end=None)
            begin = time.time()
            ret = func(*args, **kwargs)
            end = time.time()
            print(name+f' done in {end-begin:.2f} s')
            return ret
        return inner
    return decorator