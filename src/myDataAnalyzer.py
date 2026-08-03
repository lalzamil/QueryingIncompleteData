import numpy as np
import pandas as pd
import os
output_dir = "psql_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class myDataAnalyzer:
    def __init__(self,datasetName="", output_dir="", out_file=""):

        self.output_dir= output_dir
        self.datasetName=datasetName
        if self.output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.out_file=out_file
            #self.output_file= f"{self.output_dir}/mnar_Accs_bank_nyc.txt"
            self.output_file= f"{self.output_dir}/{self.out_file}"
            if self.datasetName:
                with open(self.output_file, 'a') as txtfile:
                    txtfile.write("====="+self.datasetName+"=====\n")







    def add_stats(self,acc,qt,jqt,deltaw=None):

        self.save_to_txt("accuracy:", acc)

        if deltaw is not None:
           self.save_to_txt("QT:", qt)
           self.save_to_txt("JQT:", jqt)
           self.save_to_txt("Delta w:", deltaw)
        else:
          self.save_to_txt("mdl:", jqt)
          self.save_to_txt("exe:", qt)

    def add_set_queries_time(self,set_q_time):
            self.save_Set_to_txt("time:", set_q_time)

    def add_dominancePALL_queries_time(self,set_q_time,precesion,recall,TV=0,js_sqr=0):
            self.save_Set_to_txt("time:", set_q_time)
            self.save_Set_to_txt("precesion:", precesion)
            self.save_Set_to_txt("recall:", recall)
            self.save_Set_to_txt("TV:", TV)
            self.save_Set_to_txt("js_sqr:", js_sqr)

    def add_mertics(self,set_q_time,recall,f2score=0):
            self.save_Set_to_txt("time:", set_q_time)
            # self.save_Set_to_txt("precesion:", precesion)
            self.save_Set_to_txt("recall:", recall)
            self.save_Set_to_txt("f2-score:", f2score)
            # self.save_Set_to_txt("TV:", TV)
            # self.save_Set_to_txt("js_sqr:", js_sqr)

    def unweighted_accuracy(self,group_metrics):
        """Macro‐average: simple mean of per‐group accuracies."""
        if not group_metrics:
            return None
        # return sum(g['accuracy'] for g in group_metrics) / len(group_metrics)
        accs = sum(
                    g["accuracy"]
                    for g in group_metrics
                    if g.get("accuracy") is not None
                )
        return accs / len(group_metrics)

    def weighted_accuracy(self,group_metrics):
        """
        Micro‐average: weight each group by n * true_mean.
        err = sum(n * |est-true|) / sum(n * true)
        acc = (1 - err) * 100
        """
        num = 0.0
        den = 0.0
        for g in group_metrics:
            n  = g['n']
            est = g['estimate']
            if est == None: continue
            est = float(est)
            gt  = g['ground_truth']
            if gt == None: continue
            gt = float(gt)
            num += n * abs(est - gt)
            den += n * gt
        if den == 0:
            return None
        return (1 - num/den) * 100

    def average_normalized_width(self,deltaw):

        normalized_widths=[]

        for i in  deltaw.copy():
            # print(i)
            if i[0] != -111 and i[1] != -111 :
                numenator =  (i[1] - i[0])
                # print("ub: "+str(i[1])+" lb: " +str(i[0]))
                denuminator = ((i[0]+ i[1]) / 2)
                print("ub: "+str(i[1])+" lb: " +str(i[0])+"Denu:", denuminator )
                val = numenator/(denuminator + 1e-9)
                normalized_widths.append( val)
                # print(normalized_widths)
        if normalized_widths:
            normlized_widths_mean = np.mean(normalized_widths)
        else:
            normlized_widths_mean = -111
            # self.save_to_txt((self.approachName), "delta: ", normlized_widths_mean)
        return normlized_widths_mean

    def save_to_txt(self, label, value1):
        with open(self.output_file, 'a') as txtfile:
            txtfile.write(f"{label}{value1:.3f}\n")
    def addNewLine(self):
        with open(self.output_file, 'a') as txtfile:
            txtfile.write("--------------------------------------------------------------\n")
    def save_Set_to_txt(self, label, value1):
        with open(self.output_file, 'a') as txtfile:
            txtfile.write(f"{label}{value1:.3f}\n")
    def add_fraction_lable(self,frac):
        with open(self.output_file, 'a') as txtfile:
            txtfile.write(f"\n-----------------------/ / /Prune fraction: {frac:.2f} -------------------------\n")
