import pandas as pd
sonuclar=[]
import azmodels
import azutils
#_countries=["italy"]
_countries=["ireland","greece","spain","italy","portugal"]



# x = data[data['hisseadi'] == 'ARCLK']
#sayfa='italy','spain','portugal','greece','ireland'
#sayfa="ireland"
all_results={}
total_tests = 0
isPlotting = False
azutils.create_folder("Results") 
#######DEBT RATIO tahmini

##          1. METHOD
#############SVR  ################

initial = azutils.start_timer()
results=[]

#for page in _countries:
Xtrain, Ytrain,Xtest,Ytest = azutils.azload_excel_data(
        'data/data.xlsx',"all",  "linear_svr","ireland")   
    #for k in range(3,14,1):
    #   data_x=Xtrain[:,:k]
    #    yenifeatures=feature_names[0:k]  
    #test,train = azmodels.azsupport_vector_regresyon_linear(Xtrain, Ytrain,Xtest,Ytest,
    #                                        page, isPlotting)
    #results.append([test,train])
        #df = pd.DataFrame(results, columns=["degisken","test","train"])
        #_file='Results/'+page+'degtestsonuc.csv'
        #df.to_csv(_file, index=False)
    #print(results)   
    # all_results["linear_svr"] = results
    # azutils.stop_timer(initial, "linear_svr")
    # azutils.save_results(results, "_linear_svr_",page)
    # azutils.plot_results(results, False, "_linear_svr_",page)
    # results=[] '''
    
