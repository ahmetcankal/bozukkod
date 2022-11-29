#veriler burdan yüklenecek
#df = pd.read_csv('/content/drive/MyDrive/00dosyalar/pyhton/dostumhissedegiskenler.csv')
# data = pd.read_csv('/content/drive/MyDrive/00dosyalar/pyhton/dostumhissedegiskenler.csv', sep=";")   #on_bad_lines='skip'
# data.head()
# data_y=data.loc[:,"hkapanis"]
# data_y.head()
# data_x = data.drop(columns=data.columns[-1], axis=1)
# data_x = data.drop(columns=data.columns[0], axis=1)

# data_x.head()

# x = data[data['hisseadi'] == 'ARCLK']
# x.info()


from dataclasses import dataclass
from fileinput import filename
from msilib.schema import CreateFolder
from unittest import result
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

import os, time, cmath, math 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, test 
from sklearn.feature_selection import SelectKBest, chi2, f_regression,mutual_info_classif,f_classif
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import azmodels
import xlsxwriter 
import os.path
from os import path

########################################
##########################
all_results={}
total_tests = 0
isPlotting = False
#initial = azutils.start_timer()
results=[]
tablo1=[]
<<<<<<< HEAD
resultst2=[]
tablo2=[]
_countries=["ireland","greece","spain","italy","portugal"]
=======
_countries=["ireland","italy","portugal"]
>>>>>>> 4b8d63f77688c0b7e502206cb81cf9f167a0c3ee


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print("New directory for results created.") 
    return None


def start_timer():
    return time.time()
    

def stop_timer(initial, method_name):
    end = time.time() - initial
    print( "Time elapsed for method "+method_name+": " + str(format(end, '.0f'))+ " sec.")
    return end




def azload_excel_data(file_name,_page, method_name,_test):
    
    # tüm veri yükleniyor
    #my_array = np.array([[11,22,33],[44,55,66]])
    # df = pd.DataFrame(my_array, columns = ['Column_A','Column_B','Column_C'])

    alldata = pd.read_excel('data/data_uclu.xlsx',sheet_name='all')
    ##allfeature_names = list(alldata.columns)
    alldata_y=alldata.loc[:,"gdp_ratio"]
    alldata_x=alldata.loc[:,"netlendbor_gdp":"unemp_rate"]
    allfeature_names = list(alldata_x.columns)
    alldata_ulke=alldata.loc[:,"ulke"]
   

    alldata_x = alldata_x.to_numpy()
    alldata_y = alldata_y.to_numpy()
    alldata_ulke=alldata_ulke.to_numpy()



    #normalize 
    alldata_xn,alldata_yn=aznormalize_data(alldata_x, alldata_y)

    # no normalize
    #alldata_xn,alldata_yn=alldata_x, alldata_y
    alldata_n=np.concatenate((alldata_xn,alldata_y[:,None]),axis=1)
    alldata_n=np.concatenate((alldata_n,alldata_ulke[:,None]),axis=1)
    #alldata_n=np.concatenate((alldata_xn,alldata_yn[:,None]),axis=1)
    


    ## convert your array into a dataframe
    data = pd.DataFrame (alldata_n)

    ## save to xlsx file

    filepath = 'alldata_n.xlsx'

    data.to_excel(filepath, index=False)


    #data = pd.read_excel(file_name,sheet_name=sheet_name) 
    # train data load except test 
    #data = pd.read_excel('data/data.xlsx',sheet_name='all')

    p1=[0,1,2,3]
    p2=[4,5,6,7,8]
    p3=[9,10,11,12]
    

    for testpage in _countries:
        
        #test ulke verisi ayrı diziye aktarılıyor
        vtestdata=data.loc[data[13] == testpage]
        #test veri rastgele paketlere ayrılıyor düzenleniyor
        vtestdata=azrandpart(vtestdata,testpage)
       
        #train veriler test ülkesi hariç kısmın tümü alınıyor
        vtraindata=data.loc[data[13] != testpage]
        filepath = "results/traintest_"+testpage+".xlsx"
        writer = pd.ExcelWriter(filepath,  engine = 'xlsxwriter')

        ortp0,ortp1,ortp2,ortp3=0,0,0,0
        ortp0train,ortp1train,ortp2train,ortp3train=0,0,0,0
        #ortp3train=0
        #ortp1train=0
        #ortp2train=0
        t2traindata,t2testdata=azgettable2(vtraindata,vtestdata,testpage)

        for package in range(14):
            traindata,testdata=azgettraintest(vtraindata,vtestdata,testpage,package)
            trainpage="train"+testpage+str(package)
            traindata.to_excel(writer, sheet_name = trainpage)
            testpagename = 'test'+testpage+str(package)+'.xlsx'
            testdata.to_excel(writer, sheet_name = testpagename)
    
        #data=data.loc[data[13] != "portugal"]

        #datatrain=data.loc[data[13] != _test]
        #datatrain=data[70:]
            traindata_x, traindata_y,testdata_x,testdata_y=azgetsplit(traindata,testdata)   
            

            #veriler modellere gönderiliyor
            test,train = azmodels.azsupport_vector_regresyon_linear(traindata_x, traindata_y,testdata_x,testdata_y,
                                                _test, isPlotting)
            if package in p1:
                foraverage=1
                ortp1=ortp1+test
                ortp1train=ortp1train+train
            elif package in p2:
                foraverage=2
                ortp2=ortp2+test
                ortp2train=ortp2train+train
            elif package in p3:
                foraverage=3
                ortp3=ortp3+test
                ortp3train=ortp3train+train
            else:
                foraverage=0
                ortp0=ortp0+test
                ortp0train=ortp0train+train
            results.append([testpage,package,test,train,foraverage])
            df = pd.DataFrame(results, columns=["country","package","test","train","average"])
            _file='Results/'+'countryfold.csv'
            df.to_csv(_file, index=False)
            #print(results)
        ortp1=ortp1/4
        ortp2=ortp2/5
        ortp3=ortp3/4
        ortp1train=ortp1train/4
        ortp2train=ortp2train/5
        ortp3train=ortp3train/4


<<<<<<< HEAD
        tablo1.append([testpage,"R0",ortp0,ortp0train])
        tablo1.append([testpage,"R1",ortp1,ortp1train])
        tablo1.append([testpage,"R2",ortp2,ortp2train])
        tablo1.append([testpage,"R3",ortp3,ortp3train]) 
        dftablo1 = pd.DataFrame(tablo1, columns=["country","Paketno","testaverage","train"])
        #_file='Results/'+'tablo1'+testpage+'.csv'
        _file='Results/'+'tablo1.csv'
        dftablo1.to_csv(_file, index=False)
=======
        tablo1.append([testpage,"R0",ortp0,train,foraverage])
        tablo1.append([testpage,"R1",ortp1,train,foraverage])
        tablo1.append([testpage,"R2",ortp2,train,foraverage])
        tablo1.append([testpage,"R3",ortp3,train,foraverage]) 
        dftablo1 = pd.DataFrame(tablo1, columns=["country","Paketno","testaverage","train","average"])
        _file='Results/'+'tablo1'+testpage+'.xlsx'
        dftablo1.to_excel(_file, index=False)
>>>>>>> 4b8d63f77688c0b7e502206cb81cf9f167a0c3ee
        
        
        writer.save()
        writer.close()
    #tablo1,tablo2=azozettablo(results)

    return traindata_x,traindata_y,testdata_x,testdata_y
    # return train_x,train_y, test_x,test_y,selected feature names
    #düzenle


def azrandpart(testdata,_test):
    path = "results/"+_test+"parts.xlsx"
    

    writer = pd.ExcelWriter(path, engine = 'xlsxwriter')
    
    testdata.to_excel(writer, sheet_name = 'hamveri')
    train,part1=train_test_split(testdata,test_size=22,random_state=True)
    #part1 ayrılıyor
    prt1indexes=part1.index
    kalan=testdata.drop(prt1indexes)
    #part1 ve geriye kalan ayrıldı dosyaya yazılıyor
    
   

    part1.to_excel(writer, sheet_name = 'part1')
    kalan.to_excel(writer, sheet_name = 'part1kalan')
    #part1 ve part1kalan excele yazıldı
    
    #part2 olusturuluyor ve excele kaydediliyor
    train,part2=train_test_split(kalan,test_size=22,random_state=True)
    prt2indexes=part2.index
    kalan=kalan.drop(prt2indexes)
    part2.to_excel(writer, sheet_name = 'part2')
    kalan.to_excel(writer, sheet_name = 'part2kalan')   
    
    #part 3 olusturuluyor
    train,part3=train_test_split(kalan,test_size=22,random_state=True)
    prt3indexes=part3.index
    part4=kalan.drop(prt3indexes)
    part3.to_excel(writer, sheet_name = 'part3')
    
    #part4 geriye kalan rastgele oluşturulmuş oldu
    part4.to_excel(writer, sheet_name = 'part4')   





    vdata=pd.concat([part1,part2],axis=0,ignore_index=True)
    vdata=pd.concat([vdata,part3],axis=0,ignore_index=True)
    vdata=pd.concat([vdata,part4],axis=0,ignore_index=True)

    vdata.to_excel(writer, sheet_name = 'tumpart')   
    writer.save()
    writer.close()


    return vdata

def azgetsplit(traindata,testdata):
        traindata_y=traindata.loc[:,12]
        traindata_x=traindata.loc[:,0:11]
        #feature_names = list(traindata_x.columns)
        
    #Selected_feature_names=azsave_k_highest_scores(traindata_x,traindata_y, method_name,_page)
        #data_x = data_x[data_x.columns(Selected_feature_names)]
        #traindata_x = traindata_x.loc[:,Selected_feature_names]
        
        traindata_x = traindata_x.to_numpy()
        traindata_y = traindata_y.to_numpy()

        ## test data load 
        #testdata = pd.read_excel('data/data.xlsx',sheet_name=_page)
        #testdata=data.loc[data[13] == _test]

        #testdata=data[:70]
        testdata_y=testdata.loc[:,12]
        testdata_x=testdata.loc[:,0:11]
        #column ordering
        #testdata_x = testdata_x.loc[:,Selected_feature_names]

        testdata_x = testdata_x.to_numpy()
        testdata_y = testdata_y.to_numpy()

        return traindata_x, traindata_y,testdata_x,testdata_y


def azgettraintest(vtraindata,vtestdata,_test,package):

    #test ülkesindeki ilk Paket ayrıştırılıyor
    if package==0:
        traindataek=vtestdata[:22]
        prtindexes=traindataek.index
        testdata=vtestdata.drop(prtindexes)

    elif package==1:
        traindataek=vtestdata[22:44]
        prtindexes=traindataek.index
        testdata=vtestdata.drop(prtindexes)
    elif package==2:
        traindataek=vtestdata[44:66]
        prtindexes=traindataek.index
        testdata=vtestdata.drop(prtindexes)
    elif package==3:
        traindataek=vtestdata[66:88]
        prtindexes=traindataek.index
        testdata=vtestdata.drop(prtindexes)
   #ilk 44 paket
    elif package==4:
        traindataek=vtestdata[:44]
        prtindexes=traindataek.index
        testdata=vtestdata.drop(prtindexes)
    #ikinci 44 paket
    elif package==5:
        traindataek=vtestdata[22:66]
        prtindexes=traindataek.index
        testdata=vtestdata.drop(prtindexes)
    #ucuncu 44 paket
    elif package==6:
        traindataek=vtestdata[44:88]
        prtindexes=traindataek.index
        testdata=vtestdata.drop(prtindexes)   
    #dorduncu 44 paket (1. ve 4. )
    elif package==7:
        traindataek=pd.concat([vtestdata[:22],vtestdata[66:88]],axis=0,ignore_index=True)
        prtindexes=traindataek.index
        testdata=vtestdata.drop(prtindexes)
    #besinci 44 paket (2 ve 4)
    elif package==8:
        traindataek=pd.concat([vtestdata[22:44],vtestdata[66:88]],axis=0,ignore_index=True)
        prtindexes=traindataek.index
        testdata=vtestdata.drop(prtindexes)
    #ilk 66 paket (1,2,3)
    elif package==9:
        traindataek=vtestdata[:66]
        prtindexes=traindataek.index
        testdata=vtestdata.drop(prtindexes)
    #ikinci 66 paket(2,3,4)
    elif package==10:
        traindataek=vtestdata[22:88]
        prtindexes=traindataek.index
        testdata=vtestdata.drop(prtindexes)
    #ucuncu 66 paket (1,3 ve 4)
    elif package==11:
        traindataek=pd.concat([vtestdata[:22],vtestdata[44:88]],axis=0,ignore_index=True)
        prtindexes=traindataek.index
        testdata=vtestdata.drop(prtindexes)
    #dorduncu 66 paket (1,2 ve 4)
    elif package==12:
        traindataek=pd.concat([vtestdata[:44],vtestdata[66:88]],axis=0,ignore_index=True)
        prtindexes=traindataek.index
        testdata=vtestdata.drop(prtindexes)
    #test verisi hiç eklenmiyor
    elif package==13:
        testdata=vtestdata
        traindata=vtraindata
            
    #test ülkesindeki paket train verisine ekleniyor
    if package!=13:
        traindata=pd.concat([vtraindata,traindataek],axis=0,ignore_index=True)

    
   
    #filepath = 'traindata_'+_test+str(package)+'.xlsx'

    #traindata.to_excel(filepath, index=False)

    #traine dataya eklenen kısmı ülke test verisinden çıkarıyoruz
    #testdata=np.delete(vtestdata,traindataek,axis=0)
    #testdata=vtestdata.drop([0,21], axis=0, inplace=True)
    
    #testdata.to_excel(filepath, index=False)

    testdata=testdata
    return traindata,testdata

def azgettable2(vtraindata,vtestdata,testpage):
        
        t2traindata=vtraindata
        t2testdata=vtestdata
        for paket in range(4):
              #veriler modellere gönderiliyor
            #test ülkesindeki ilk Paket ayrıştırılıyor P1 test kalanı eğtim verisi
           
            if paket==0:
                t2testdata=vtestdata[:22]
                prtindexes=t2testdata.index
                t2traindata=vtestdata.drop(prtindexes)
            #test ülkesindeki ilk Paket ayrıştırılıyor P2 test kalanı eğtim verisi

            elif paket==1:
                t2testdata=vtestdata[22:44]
                prtindexes=t2testdata.index
                t2traindata=vtestdata.drop(prtindexes)

            #test ülkesindeki ilk Paket ayrıştırılıyor P3 test kalanı eğtim verisi    
            elif paket==2:
                t2testdata=vtestdata[44:66]
                prtindexes=t2testdata.index
                t2traindata=vtestdata.drop(prtindexes)

            #test ülkesindeki ilk Paket ayrıştırılıyor P4 test kalanı eğtim verisi        
            elif paket==3:
                t2testdata=vtestdata[66:88]
                prtindexes=t2testdata.index
                t2traindata=vtestdata.drop(prtindexes)
            traindata_x, traindata_y,testdata_x,testdata_y=azgetsplit(t2traindata,t2testdata)   
            t2test,t2train = azmodels.azsupport_vector_regresyon_linear(traindata_x, traindata_y,testdata_x,testdata_y,
                                                testpage, isPlotting)
            
           # resultst2.append([testpage,paket,t2test,t2train])
            #df = pd.DataFrame(results, columns=["country","package","test","train"])
            #_file='Results/'+testpage+'countrytable2fold.csv'
            #df.to_csv(_file, index=False)                     
            tablo2.append([testpage,paket,t2test,t2train])
            
            dftablo2 = pd.DataFrame(tablo2, columns=["country","Paketno","test","train"])
        #_file='Results/'+'tablo1'+testpage+'.csv'
            _file='Results/'+'tablo2.csv'
            dftablo2.to_csv(_file, index=False)
        return t2traindata,t2testdata
def azozettablo(results):
    #for say in range(70):
    #for ind in results.index:
    #    if results['average'][ind]==1:




    tablo1=results

    tablo2= results
    return tablo1,tablo2

def aznormalize_data(data_x, data_y):
    scalerx = MinMaxScaler().fit(data_x)
    X_all = scalerx.fit_transform(data_x)
    scalery = MinMaxScaler() 
    Y_all = data_y.reshape(-1,1)
    #X_train, X_test, y_train, y_test = train_test_split(X_std, Y_std, 
                                        #train_size = 0.80,random_state=0)

    return X_all,Y_all





def azsave_k_highest_scores(data_x,data_y, method_name,_page):
    selector=SelectKBest(score_func=f_regression, k=12)
    model= selector.fit(data_x,data_y)
    selected_feature_names=data_x.columns[model.get_support()]
    scores=model.scores_
    pvalues=model.pvalues_
    
    zipped=zip(selected_feature_names,scores,pvalues)
    zipped=sorted(zipped, key=lambda x: x[1],reverse=True)
    
    df = pd.DataFrame(zipped, columns=["variables","score","pvalue"])
    create_folder("Results/"+method_name+'/')
    df.to_csv('Results/'+method_name+'/'+_page+'_allvariables.csv', index=False)
    # for feature,score in zipped:
    #     print(feature,score)
    selected_feature_names=[]
    for f,s,pv in zipped:
        selected_feature_names.append(f)

    return selected_feature_names

  
def azplot_f_importance(coef, names,_k, method_name,_page):
    imp = coef[0]
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
   
    fig = plt.gcf()
    fig.set_size_inches((15, 15), forward=False)
    create_folder("Results/"+method_name)
    plt.savefig('Results/'+method_name+'/'+_page+str(_k)+'_variables.png', dpi=300)
    plt.close()
    
    return None




def plot_results(results, show_plot, method_name,_page):
    plt.figure(figsize=(5, 2.7), layout='constrained')
    np_results = np.array(results)
    #variables = np_results[:,0].astype(int)
    test_results = np_results[:,0]
    train_results = np_results[:,1]

    rlen = len(test_results)-1
    variables_order = -1
    max_score = 0
    for i in range(0, rlen):
        score = test_results[i]
        if score > max_score:
            max_score = score
            variables_order = i

  #  plt.scatter(variables[variables_order], 
  #              test_results[variables_order], c="red")
  #  plt.annotate(str(format(test_results[variables_order], '.4f')),
  #              ( variables[variables_order], test_results[variables_order]))
     
    #plt.scatter(variables[variables_order], 
    #           train_results[variables_order], c="blue")
 

    # plt.plot(variables, test_results, label='Test')  
    # plt.plot(variables, train_results, label='Train')   
    # plt.xlabel('Variables')
    # plt.ylabel('Score')
    # plt.title("Variable Score Table for "+method_name)
    # plt.legend()

    # fig = plt.gcf()
    # fig.set_size_inches((5, 5), forward=False)
    # create_folder("Results")
    # plt.savefig('Results/'+method_name+_page+'_variable_score_table.png', dpi=300)

    # if show_plot:
    #     plt.show()

    # plt.close()

    return None


def plot_all_results(all_results):
 
    axesSize = math.ceil(math.sqrt(len(all_results)))
    figure, axis = plt.subplots(axesSize, axesSize)

    axis_x = 0
    axis_y = 0
    for model_name, rslt in all_results.items(): 

        np_results = np.array(rslt)
        variables = np_results[:,0].astype(int)
        test_results = np_results[:,1]
        train_results = np_results[:,2]
     
        rlen = len(test_results)-1
        variables_order = -1
        max_score = 0
        for i in range(0, rlen):
            score = test_results[i]
            if score > max_score:
                max_score = score
                variables_order = i
        
        axis[axis_x, axis_y].scatter(variables[variables_order], 
                                test_results[variables_order], c="red")
        axis[axis_x, axis_y].annotate(str(format(test_results[variables_order], 
                 '.4f')),
                (variables[variables_order], test_results[variables_order]))
 
        axis[axis_x, axis_y].plot(variables, test_results, label='Test')  
        axis[axis_x, axis_y].plot(variables, train_results, label='Train')   
        # axis[axis_x, axis_y].xlabel('Variables')
        # axis[axis_x, axis_y].ylabel('Score')
        axis[axis_x, axis_y].set_title(model_name)
        # axis[axis_x, axis_y].legend()
        axis_y = axis_y + 1
        if(axis_y==axesSize):
            axis_y = 0
            axis_x = axis_x + 1

   
    figure.set_size_inches((axesSize*3, axesSize*3), forward=False)
    create_folder("Results")
    plt.savefig('Results/all_score_tables.png', dpi=300)
    plt.close()

    return None


def save_results(results, method_name,_page):
    df = pd.DataFrame(results, columns=["test","train"])
    df.to_csv(r'Results/'+method_name+_page+'_results.csv', index=False)

    return None
    
def save_all_results(all_results):
    df = pd.DataFrame(all_results, columns=["test","train","model"])
    df.to_csv(r'Results/all_results.csv', index=False)

    return None
