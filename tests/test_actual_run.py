# test run on all dataset using parameterize
import pytest
import config as cf
import my_main

@pytest.mark.parametrize("dataset_name", ["Amazon", "ARC" , "ChemProt", "Economy", "HyperPartisan", "News","Parties", "SCIERC","Twitter","Yelp_Hotel"])
@pytest.mark.parametrize("base_model", ["TextRCNN","TextCNN"])
@pytest.mark.parametrize("stereotype", [cf.StereoType.Normal])
def test_actual_run_init(dataset_name,base_model,stereotype):
    cf.DATA_PATH = './data/'
    cf.Dataset_Name = dataset_name
    cf.Base_Model = base_model
    cf.Stereotype = stereotype

    my_main.MAIN()

@pytest.mark.parametrize("dataset_name", ["Amazon", "ARC" , "ChemProt", "Economy", "HyperPartisan", "News","Parties", "SCIERC","Twitter","Yelp_Hotel"])
@pytest.mark.parametrize("base_model", ["TextRCNN","TextCNN"])
@pytest.mark.parametrize("stereotype", [cf.StereoType.Keyword, cf.StereoType.Imbword, cf.StereoType.RandomMask, cf.StereoType.Noun])
def test_actual_run_RCNN_CNN(dataset_name,base_model,stereotype):
    cf.Round=1
    cf.DATA_PATH = './data/'
    cf.Dataset_Name = dataset_name
    cf.Base_Model = base_model
    cf.Stereotype = stereotype

    my_main.MAIN()


@pytest.mark.parametrize("dataset_name", ["Amazon", "ARC" , "ChemProt", "Economy", "HyperPartisan", "News","Parties", "SCIERC","Twitter","Yelp_Hotel"])
@pytest.mark.parametrize("base_model", ["TextRCNN","TextCNN"])
@pytest.mark.parametrize("stereotype", [cf.StereoType.Normal, cf.StereoType.Keyword, cf.StereoType.Imbword])
def test_smoke_bigram(dataset_name,base_model,stereotype):
    cf.Round = 1
    cf.DATA_PATH = './data/'
    cf.Dataset_Name = dataset_name
    cf.Base_Model = base_model
    cf.Stereotype = stereotype
    cf.N_GRAM = 2
    cf.Alpha = 1
    my_main.MAIN()


@pytest.mark.parametrize("dataset_name", ["Amazon", "ARC" , "ChemProt", "Economy", "HyperPartisan", "News","Parties", "SCIERC","Twitter","Yelp_Hotel"])
@pytest.mark.parametrize("base_model", ["RoBERTa"])
@pytest.mark.parametrize("stereotype", [cf.StereoType.Normal, cf.StereoType.Keyword, cf.StereoType.Imbword, cf.StereoType.RandomMask, cf.StereoType.Noun])
def test_smoke_roberta(dataset_name,base_model,stereotype):
    cf.DATA_PATH = './data/'
    cf.Dataset_Name = dataset_name
    cf.Base_Model = base_model
    cf.Stereotype = stereotype
    my_main.MAIN()


@pytest.mark.parametrize("dataset_name", ["Amazon", "ARC" , "ChemProt", "Economy", "HyperPartisan", "News","Parties", "SCIERC","Twitter","Yelp_Hotel"])
@pytest.mark.parametrize("base_model", ["RoBERTa"])
@pytest.mark.parametrize("stereotype", [cf.StereoType.Normal, cf.StereoType.Keyword, cf.StereoType.Imbword])
def test_smoke_bigram_roberta(dataset_name,base_model,stereotype):
    cf.DATA_PATH = './data/'
    cf.Dataset_Name = dataset_name
    cf.Base_Model = base_model
    cf.Stereotype = stereotype
    cf.N_GRAM = 2
    cf.Alpha = 1
    my_main.MAIN()