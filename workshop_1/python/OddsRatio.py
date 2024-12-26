import math 
import numpy as np
from typing import List

class OddsRatio():
    def __init__(self, event_e: int, n_e: int, event_c: int, n_c):
        '''
        Perform odds ratio analysis. Requires arguments:
        `event_e` : number with event in experimental group
        `n_e` : total number in experimental group
        `event_c` : number with event in control group
        `n_c` : total number in control group
        '''
        self.event_e = event_e
        self.n_e = n_e
        self.event_c = event_c
        self.n_c = n_c
    
    def print_contigency_table(self):
        print("\n              Not affected Affected")
        print("Control      ",self.n_c - self.event_c,"        ",self.event_c)
        print("Experimental ",self.n_e - self.event_e,"        ",self.event_e)

    def calculate_odds(self) -> float:
        odds = (self.event_e / (self.n_e - self.event_e)) / \
            (self.event_c / (self.n_c - self.event_c))
        return odds
    
    def calculate_confidence_interval(self) -> List[float]:
        odds = self.calculate_odds()
        lower_ci = math.e ** (np.log(odds)-1.96*math.sqrt(\
            (1/self.event_e)+(1/(self.n_e - self.event_e))+\
            (1/self.event_c)+(1/(self.n_c - self.event_c))))
        upper_ci = math.e ** (np.log(odds)+1.96*math.sqrt(\
            (1/self.event_e)+(1/(self.n_e - self.event_e))+\
            (1/self.event_c)+(1/(self.n_c - self.event_c))))
        return [lower_ci,upper_ci]
