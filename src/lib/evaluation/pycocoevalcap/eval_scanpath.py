from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = 'tylin'

from .ciderR.ciderR import CiderR
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice
from .wmd.wmd import WMD
import numpy as np

class ScanpathEval:
    def __init__(self, Gts, Res):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.Gts = Gts
        self.Res = Res
        # self.params = {'image_id': coco.getImgIds()}
        #
        # self.Spice = Spice()

    def evaluate(self):
        # imgIds = self.params['image_id']
        # # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for key in self.Res:
            gts[key] = self.Gts[key]
            res[key] = self.Res[key]

        # =================================================
        # Set up scorers
        # =================================================
        # print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (CiderR(), "CIDEr-R"),
            # (self.Spice, "SPICE"),
            # (WMD(),   "WMD"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, list(gts.keys()), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, list(gts.keys()), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in list(self.imgToEval.items())]