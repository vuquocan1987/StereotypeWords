import torch
from torch.utils.data import Dataset, DataLoader
import os
import warnings
import copy
import sys
import config as cf
import model, data_process
import argparse

from pathlib import Path


def MAIN():


    # settings
    cf.XMaxLen = 0
    cf.YList = []
    cf.Use_GPU = torch.cuda.is_available()   

    # output configuration
    print('Dataset_Name={}'.format(cf.Dataset_Name))
    print('Base_Model={}'.format(cf.Base_Model))

    # multiple rounds experiments
    for i in range(cf.Round):
        # if i > 1:
        #     cf.Pretrained = True
            
        # random seed setting    
        cf.Seed = i        
        cf.random_setting(i)
        
        # prepare a specific dataset
        TextDataset = data_process.TextDataset(cf.Dataset_Name)
        
        # build model for intialized training
        if cf.Base_Model == 'TextCNN' or cf.Base_Model == 'TextRCNN':

            [cf.embedding, cf.word2id] = cf.Pickle_Read('./w2v/glove.300d.en.txt.pickle')
            cf.embedding.weight.requires_grad = False

        if cf.Base_Model == 'TextCNN':

            model_x = model.TextCNN()
            model_s = model.TextCNN()

        elif cf.Base_Model == 'TextRCNN':

            model_x = model.TextRCNN()
            model_s = model.TextRCNN()


        elif cf.Base_Model == 'RoBERTa':
            model_x = model.RoBERTa()
            model_s = model.RoBERTa()
        else:
            warnings.warn(f'No such model: {cf.Base_Model}!')
            sys.exit(0)
        

        if cf.Use_GPU == True:
            model_x = model_x.cuda()
            model_s = model_s.cuda()

        # prepare dataloader for initialized training
        train_dataset = data_process.TrainDataset(TextDataset.train_examples)
        dev_dataset = data_process.TrainDataset(TextDataset.dev_examples)
        test_dataset = data_process.TrainDataset(TextDataset.test_examples)

        train_loader = DataLoader(train_dataset, batch_size=cf.Train_Batch_Size, shuffle=cf.DataLoader_Shuffle)
        dev_loader = DataLoader(dev_dataset, batch_size=cf.DevTest_Batch_Size, shuffle=cf.DataLoader_Shuffle)
        test_loader = DataLoader(test_dataset, batch_size=cf.DevTest_Batch_Size, shuffle=cf.DataLoader_Shuffle)
        tr = model.Train(model_x, model_s)
        
        f_test_bacc, f_test_bmaf1, init_factual_keyword_fairness = 0,0,0
        try:
            tr.model_x.load_state_dict(torch.load(cf.get_init_model_path()))
        except:
            f_test_bacc, f_test_bmaf1, init_factual_keyword_fairness = tr.Init_Train(train_loader, dev_loader, test_loader)                                      
        # check whether pre-trained model exists
        # if not cf.Pretrained:                        
        #     f_test_bacc, f_test_bmaf1, init_factual_keyword_fairness = tr.Init_Train(train_loader, dev_loader, test_loader)                                      
        # else:                    
        #     # try:
        #     tr.model_x.load_state_dict(torch.load(cf.get_init_model_path()))
            # except:
                # quick hack to load initial model remember to fix this!
                # tr.model_x.load_state_dict(torch.load(cf.Base_Model + cf.Dataset_Name + "Normal" + 'xinit.pt'))         
        
        # mask the stereotype words    
        TextDataset.Read_Data(True,tr)
        # prepare dataloader for biased training
        train_dataset = data_process.TrainDataset(TextDataset.train_examples)
        dev_dataset = data_process.TrainDataset(TextDataset.dev_examples)
        test_dataset = data_process.TrainDataset(TextDataset.test_examples)

        train_loader = DataLoader(train_dataset, batch_size=cf.Train_Batch_Size, shuffle=cf.DataLoader_Shuffle)
        dev_loader = DataLoader(dev_dataset, batch_size=cf.DevTest_Batch_Size, shuffle=cf.DataLoader_Shuffle)
        test_loader = DataLoader(test_dataset, batch_size=cf.DevTest_Batch_Size, shuffle=cf.DataLoader_Shuffle)
        # biased training and debiased prediction
        test_acc, test_bacc, test_maf1, test_bmaf1, f_fairness, f_bfairness = tr.Train(train_loader, dev_loader, test_loader, i+100)

        write_result_to_disk(f_test_bacc, f_test_bmaf1, init_factual_keyword_fairness, test_acc, test_bacc, test_maf1, test_bmaf1, f_fairness, f_bfairness)

def write_result_to_disk(f_test_bacc, f_test_bmaf1, init_factual_keyword_fairness, test_acc, test_bacc, test_maf1, test_bmaf1, f_fairness, f_bfairness):
    
    with open(cf.get_file_prefix() + "Test" if cf.IS_TESTING else "" +'.txt', 'a') as f:
        f.write(cf.Base_Model)
        f.write('\n')
        f.write(cf.Fusion)
        f.write('\n')
        f.write(cf.Dataset_Name)
        f.write('\n')
        f.write(str(cf.Sigma))
        f.write(str(cf.Stereotype.name))
        f.write('\n')
        f.write('Init Acc in {}-Rounds= {}'.format(cf.Round, f_test_bacc))
        f.write('\n')
        f.write('Init F1 in {}-Rounds= {}'.format(cf.Round, f_test_bmaf1))
        f.write('\n')
        f.write('Init Fairness in {}-Rounds= {}'.format(cf.Round, init_factual_keyword_fairness))
        f.write('\n')
        f.write('Final Acc in {}-Rounds= {}'.format(cf.Round, test_acc))
        f.write('\n')
        f.write('Final BaseAcc in {}-Rounds= {}'.format(cf.Round, test_bacc))
        f.write('\n')
        f.write('Final F1 in {}-Rounds= {}'.format(cf.Round, test_maf1))
        f.write('\n')
        f.write('Final BaseF1 in {}-Rounds= {}'.format(cf.Round, test_bmaf1))
        f.write('\n')
        f.write('Final Fair in {}-Rounds= {}'.format(cf.Round, f_fairness))
        f.write('\n')
        f.write('Final BaseFair in {}-Rounds= {}'.format(cf.Round, f_bfairness))
        f.write('\n')
    if cf.IS_TESTING:
        csv_file_path = Path("result/csv/Tests/Result.csv")
    else:
        csv_file_path = Path("result/csv/Result.csv")
    if not csv_file_path.exists():
        with open(csv_file_path, 'w') as f:
            f.write('Base_Model,StereoType,Fusion,Dataset_Name,Sigma,Round,InitAcc,InitF1,InitFairness,FinalAcc,FinalBaseAcc,FinalF1,FinalBaseF1,FinalFairness,FinalBaseFairness,NumTrainEpoch,BatchNo\n')
    with open(csv_file_path, 'a') as f:
        f.write(f"{cf.Base_Model},{cf.Stereotype.name},{cf.Fusion},{cf.Dataset_Name},{cf.Sigma},{cf.Round},{f_test_bacc},{f_test_bmaf1},{init_factual_keyword_fairness},{test_acc},{test_bacc},{test_maf1},{test_bmaf1},{f_fairness},{f_bfairness},{cf.Epoch},{cf.BATCH}\n")
        
def main():
    print('sys.argv={}'.format(sys.argv))
    parser = argparse.ArgumentParser(
        prog='Debias Model',
        description='Test run the models',
    )
    parser.add_argument('--Dataset_Name', type=str, default='Amazon', help='Dataset Name')  
    parser.add_argument('--Base_Model', type=str, default='TextCNN', help='Base Model Name')
    args = parser.parse_args()
    cf.Dataset_Names = [args.Dataset_Name]
    cf.Base_Model = args.Base_Model
    # Run model on datasets
    for name in cf.Dataset_Names:
        cf.Dataset_Name = name
        MAIN()

if __name__ == "__main__":
    main()
