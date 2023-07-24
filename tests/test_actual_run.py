# test run on all dataset using parameterize
import pytest
import config as cf
import my_main


@pytest.mark.parametrize("dataset_name", ["Economy","Amazon","Parties","SCIERC", "ARC" , "HyperPartisan", "Twitter"])
@pytest.mark.parametrize("base_model", ["TextRCNN","TextCNN"])
@pytest.mark.parametrize("stereotype", [cf.StereoType.RandomMask])
def test_actual_run_everything_light(dataset_name,base_model,stereotype):
    cf.Round=1
    cf.Batch
    cf.DATA_PATH = './data/'
    cf.Dataset_Name = dataset_name
    cf.Base_Model = base_model
    cf.Stereotype = stereotype

    my_main.MAIN()



@pytest.mark.parametrize("dataset_name", ["Amazon", "ARC" , "ChemProt", "Economy", "HyperPartisan", "News","Parties", "SCIERC","Twitter","Yelp_Hotel"])
@pytest.mark.parametrize("base_model", ["TextRCNN","TextCNN"])
@pytest.mark.parametrize("stereotype", [cf.StereoType.Keyword])
def test_actual_bigram(dataset_name,base_model,stereotype):
    cf.Round = 1
    cf.DATA_PATH = './data/'
    cf.Dataset_Name = dataset_name
    cf.Base_Model = base_model
    cf.Stereotype = stereotype
    cf.N_GRAM = 2
    cf.Alpha = 1
    my_main.MAIN()


@pytest.mark.parametrize("dataset_name", ["Amazon", "ARC" , "ChemProt", "Economy", "HyperPartisan", "Parties", "SCIERC","Twitter",]) # "Yelp_Hotel","News",
@pytest.mark.parametrize("base_model", ["TextRCNN","TextCNN"])
@pytest.mark.parametrize("stereotype", [cf.StereoType.Idiom])
def test_actual_Idiom(dataset_name,base_model,stereotype):
    cf.Round = 1
    cf.DATA_PATH = './data/'
    cf.Dataset_Name = dataset_name
    cf.Base_Model = base_model
    cf.Stereotype = stereotype
    my_main.MAIN()



@pytest.mark.parametrize("dataset_name", ["Yelp_Hotel", "News","Economy","Amazon","Parties", "ChemProt","SCIERC", "ARC" , "HyperPartisan", "Twitter",])
@pytest.mark.parametrize("base_model", ["TextRCNN","TextCNN"])
@pytest.mark.parametrize("stereotype", [cf.StereoType.Normal])
def test_get_init(dataset_name,base_model,stereotype):
    cf.Round = 1
    cf.Init_epoch = 1
    cf.DATA_PATH = './data/'
    cf.Dataset_Name = dataset_name
    cf.Base_Model = base_model
    cf.Stereotype = stereotype
    my_main.MAIN()


