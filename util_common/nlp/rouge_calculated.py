from rouge import Rouge


def rouge(sys, ref):
    rouge = Rouge()
    return rouge.get_scores(sys, ref, avg=True)
