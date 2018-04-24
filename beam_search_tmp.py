#coding:utf-8
import numpy as np
import os, sys
import torch
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

class Node(object):
    def __init__(self, score, val):
        self.score = score
        self.val = val
        self.children = None

class BeamSearch(object):
    def __init__(self, width=5, max_depth=30):
        self.width = width
        self.max_depth = max_depth
        self.digger = None
        self.end = 2
        self.initialize()
        
    def initialize(self):
        self.depth = 0
        self.result = []
        self.thresh = 0
        self.root = Node(1, [1])
        self.growthnodes = [self.root]
        
    def search(self):
        self.initialize()
        while True:
            self.growth()
            logger.debug(self.depth)
            if len(self.result) > self.width:
                self.thresh = min([node.score for node in self.result])
            if self.depth >= self.max_depth:
                self.result.extend(self.growthnodes)                
                break
            if len(self.growthnodes) == 0:
                break

        self.result = sorted(self.result, key=lambda x:-x.score)
        return self.result[:self.width]
        
    def growth(self):
        next_growth = []
        for node in self.growthnodes:
            self._growth(node)
            next_growth.extend(node.children)

        next_growth = sorted(next_growth, key=lambda x:-x.score)
        next_growth = next_growth[:self.width]
        end_nodes = [node for node in next_growth if node.val[-1] == self.end]
        next_growth = [node for node in next_growth if node.val[-1] != self.end]

        self.result.extend(end_nodes)
        self.growthnodes = next_growth
        self.depth += 1

    def _growth(self, node):
        base_score = node.score
        topi, topv = self.digger(node)
        topv = topv[:self.width] * base_score # + alpha
        topi = topi[:self.width]
        node.children = [Node(v, node.val + [i]) for v, i in zip(topv, topi) if v > self.thresh]
                
if __name__=="__main__":
    beam = BeamSearch()
    beam.end = 5
    def digger(node):
        val = node.val
        children = np.random.randint(0, 2, size=(5,)) + max(val)
        score = np.random.random(size=(5,))
        return children, score
    
    beam.digger = digger
    result = beam.search()

    for node in result:
        print(node.val, node.score)
