
class measure:
    def F(self, gt, my_bin):
        tmp = my_bin - gt
        FP = sum(sum(tmp == 255))
        FN = sum(sum(tmp == -255))
        TP = sum(sum((my_bin == gt) & (gt == 255)))
        p = TP / (TP + FP)
        r = TP / (TP + FN)

        belt2 = 0.1

        Fmeasure = ((1 + belt2 ) * p * r) / (belt2 * (p + r))
        return p, r, Fmeasure


