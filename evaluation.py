import argparse
import collections
import numpy as np
import pandas as pd
# from sklearn import metrics
# from sklearn.preprocessing import label_binarize


def get_y_true():
    # """ 
    # Read file to obtain y_true.
    # All of five tasks of Sentihood use the test set of task-BERT-pair-NLI-M to get true labels.
    # All of five tasks of SemEval-2014 use the test set of task-BERT-pair-NLI-M to get true labels.

    true_data_file = "/content/drive/My Drive/Dataset/true_test.csv"
    

    df = pd.read_csv(true_data_file,sep='\t',header=None).values
    y_true=[]
    for i in range(len(df)):
        label = df[i][1]
        assert label in ['positive', 'neutral', 'negative', 'none'], "error!"
        if label == 'negative':
            n = 0
        elif label == 'positive':
            n = 1
        elif label == 'neutral':
            n = 2
        elif label == 'none':
            n = 3

        y_true.append(n)
    
    return y_true


def get_y_pred(pred_data_dir):
    """ 
    Read file to obtain y_pred and scores.
    """
    pred=[]
    score=[]
    
    count = 0
    tmp = []
    with open(pred_data_dir, "r", encoding="utf-8") as f:
        s = f.readline().strip().split()
        while s:
            tmp.append([float(s[2])])
            count += 1
            if count % 4 == 0:
                tmp_sum = np.sum(tmp)
                t = []
                for i in range(4):
                    t.append(tmp[i] / tmp_sum)
                score.append(t)
                # if t[0] >= t[1] and t[0] >= t[2] and t[0]>=t[3] and t[0]>=t[4]:
                #     pred.append(0)
                # elif t[1] >= t[0] and t[1] >= t[2] and t[1]>=t[3] and t[1]>=t[4]:
                #     pred.append(1)
                # elif t[2] >= t[0] and t[2] >= t[1] and t[2]>=t[3] and t[2]>=t[4]:
                #     pred.append(2)
                # elif t[3] >= t[0] and t[3] >= t[1] and t[3]>=t[2] and t[3]>=t[4]:
                #     pred.append(3)
                # else:
                #     pred.append(4)
                
                pred.append(t.index(max(t)))
                tmp = []
            s = f.readline().strip().split()

    return pred, score



# def sentihood_AUC_Acc(y_true, score):
    # """
    # Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of Sentihood.
    # Calculate "Acc" of sentiment classification task of Sentihood.
    # """
    # # aspect-Macro-AUC
    # aspect_y_true=[]
    # aspect_y_score=[]
    # aspect_y_trues=[[],[],[],[]]
    # aspect_y_scores=[[],[],[],[]]
    # for i in range(len(y_true)):
    #     if y_true[i]>0:
    #         aspect_y_true.append(0)
    #     else:
    #         aspect_y_true.append(1) # "None": 1
    #     tmp_score=score[i][0] # probability of "None"
    #     aspect_y_score.append(tmp_score)
    #     aspect_y_trues[i%4].append(aspect_y_true[-1])
    #     aspect_y_scores[i%4].append(aspect_y_score[-1])

    # aspect_auc=[]
    # for i in range(4):
    #     aspect_auc.append(metrics.roc_auc_score(aspect_y_trues[i], aspect_y_scores[i]))
    # aspect_Macro_AUC = np.mean(aspect_auc)
    
    # # sentiment-Macro-AUC
    # sentiment_y_true=[]
    # sentiment_y_pred=[]
    # sentiment_y_score=[]
    # sentiment_y_trues=[[],[],[],[]]
    # sentiment_y_scores=[[],[],[],[]]
    # for i in range(len(y_true)):
    #     if y_true[i]>0:
    #         sentiment_y_true.append(y_true[i]-1) # "Postive":0, "Negative":1
    #         tmp_score=score[i][2]/(score[i][1]+score[i][2])  # probability of "Negative"
    #         sentiment_y_score.append(tmp_score)
    #         if tmp_score>0.5:
    #             sentiment_y_pred.append(1) # "Negative": 1
    #         else:
    #             sentiment_y_pred.append(0)
    #         sentiment_y_trues[i%4].append(sentiment_y_true[-1])
    #         sentiment_y_scores[i%4].append(sentiment_y_score[-1])

    # sentiment_auc=[]
    # for i in range(4):
    #     sentiment_auc.append(metrics.roc_auc_score(sentiment_y_trues[i], sentiment_y_scores[i]))
    # sentiment_Macro_AUC = np.mean(sentiment_auc)

    # # sentiment Acc
    # sentiment_y_true = np.array(sentiment_y_true)
    # sentiment_y_pred = np.array(sentiment_y_pred)
    # sentiment_Acc = metrics.accuracy_score(sentiment_y_true,sentiment_y_pred)

    # return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC


def semeval_PRF(y_true, y_pred):
    """
    Calculate "Micro P R F" of aspect detection task of SemEval-2014.
    """
    s_all=0
    g_all=0
    s_g_all=0
    for i in range(len(y_pred)//5):
        s=set()
        g=set()
        for j in range(5):
            if y_pred[i*5+j]!=3:
                s.add(j)
            if y_true[i*5+j]!=3:
                g.add(j)
        if len(g)==0:continue
        s_g=s.intersection(g)
        s_all+=len(s)
        g_all+=len(g)
        s_g_all+=len(s_g)

    p=s_g_all/s_all
    r=s_g_all/g_all
    f=2*p*r/(p+r)

    return p,r,f


def semeval_Acc(y_true, y_pred, score, classes=4):
    """
    Calculate "Acc" of sentiment classification task of SemEval-2014.
    """
    assert classes in [2, 3, 4], "classes must be 2 or 3 or 4."

    if classes == 4:
        total=0
        total_right=0
        for i in range(len(y_true)):
            if y_true[i]==4:continue
            total+=1
            tmp=y_pred[i]
            if tmp==4:
                if score[i][0]>=score[i][1] and score[i][0]>=score[i][2] and score[i][0]>=score[i][3]:
                    tmp=0
                elif score[i][1]>=score[i][0] and score[i][1]>=score[i][2] and score[i][1]>=score[i][3]:
                    tmp=1
                elif score[i][2]>=score[i][0] and score[i][2]>=score[i][1] and score[i][2]>=score[i][3]:
                    tmp=2
                else:
                    tmp=3
            if y_true[i]==tmp:
                total_right+=1
        sentiment_Acc = total_right/total
    elif classes == 3:
        total=0
        total_right=0
        for i in range(len(y_true)):
            if y_true[i]>=3:continue
            total+=1
            tmp=y_pred[i]
            if tmp>=3:
                if score[i][0]>=score[i][1] and score[i][0]>=score[i][2]:
                    tmp=0
                elif score[i][1]>=score[i][0] and score[i][1]>=score[i][2]:
                    tmp=1
                else:
                    tmp=2
            if y_true[i]==tmp:
                total_right+=1
        sentiment_Acc = total_right/total
    else:
        total=0
        total_right=0
        for i in range(len(y_true)):
            if y_true[i]>=2:continue #or y_true[i]==1:continue
            total+=1
            tmp=y_pred[i]
            if tmp>=2: #or tmp==1:
                if score[i][0]>=score[i][1]:
                    tmp=0
                else:
                    tmp=1
            if y_true[i]==tmp:
                total_right+=1
        sentiment_Acc = total_right/total

    return sentiment_Acc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_data_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The pred data dir.")
    args = parser.parse_args()


    result = collections.OrderedDict()
  
    y_true = get_y_true()
    y_pred, score = get_y_pred(args.pred_data_dir)
    aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)
    # sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
    sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
    sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
    result = {'aspect_Precision': aspect_P,
            'aspect_Recall': aspect_R,
            'aspect_F1': aspect_F,
            # 'sentiment_Acc_4_classes': sentiment_Acc_4_classes,
            'sentiment_Acc_3_classes': sentiment_Acc_3_classes,
            'sentiment_Acc_2_classes': sentiment_Acc_2_classes}

    for key in result.keys():
        print(key, "=",str(result[key]))
    

if __name__ == "__main__":
    main()
