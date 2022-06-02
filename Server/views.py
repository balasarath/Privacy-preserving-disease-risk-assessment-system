from django.shortcuts import render
import json
from django.http import JsonResponse
from django.http import HttpResponse
from .models import Symptoms
from django.views.decorators.csrf import csrf_exempt
from .forms import UploadFileForm
from .models import FileModel
from sklearn.tree import DecisionTreeClassifier                            
from sklearn.model_selection import train_test_split
import joblib                                               
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from django.conf import settings
from io import BytesIO
from io import StringIO
from PIL import Image
import matplotlib.pyplot as plt
import math
import csv
from collections import Counter
from imblearn.over_sampling import SMOTE

def hospital(request):
    return render(request,"hospital.html")

def hospitalResult(request):
    return render(request,"hospitalResult.html")

def patient(request):
    SymptomsList = getSymptoms()
    SymptomsList.sort()
    return render(request,"patient.html",{'SymptomsList' : SymptomsList})

def getSymptoms():
    q = Symptoms.objects.all().values('Sym_name')
    response = []
    for i in q:
        response.append(i['Sym_name'])
    return response

def reset(request):
    SymptomsList = getSymptoms()
    SymptomsList.sort()
    return render(request,"patient.html",{'SymptomsList' : SymptomsList})

def uploadFile(request):
    print("Executing...")
    if request.method == 'POST':
        print("\n\n*********** RSA Encrypted Text ***********\n\n")
        ciphertext = request.POST['reqdata']
        for i in range(math.floor(len(ciphertext)*0.01)):
            if i%133 == 0:
                print()
            print(ciphertext[i],end="")
        print("......")
        print("\n\n*********** RSA Decrypted text ***********\n")
        plaintxt = decrypt(ciphertext)
        for i in range(math.floor(len(plaintxt)*0.01)):
            if i%133 == 0:
                print()
            print(plaintxt[i],end="")
        print("......\n")
        csvText = ""
        for i in range(len(plaintxt)):
            csvText+=plaintxt[i]
        newdf = pd.read_csv(StringIO(csvText))
        print("\n\n\n")
        print(newdf)
        attr = newdf.drop('prognosis', axis=1)
        target = newdf['prognosis']  
        d = {}
        for key in target:
            d[key] = d.get(key, 0) + 1

        #print count of each disease
        for i in d:
            print(i, d[i])

        smote = SMOTE()
        x_smote, y_smote = smote.fit_resample(attr, target)
        x_smote['prognosis'] = y_smote
        
        print('\n\nOriginal dataset shape', Counter(target))
        print('\n\nResample dataset shape', Counter(y_smote))
        f = open(settings.MEDIA_ROOT+"\\test\\UploadTest.csv", 'a+', newline='')
        f.seek(0) # Ensure you're at the start of the file..
        first_char = f.read(1) # Get the first character
        exist = False
        if not first_char:
            print("file is empty") 
        else:
            exist = True
        writer = csv.writer(f)
        if exist == False:
            writer.writerow(list(x_smote.columns.values))
            
        for index,row in x_smote.iterrows():
                writer.writerow(row)
        f.close()
        Acc = trainModel()
        Acc*=100
        return render(request,"hospitalResult.html",{"Accuracy" : str(Acc)})
    else:
        print("return exec")
        return render(request,"hospital.html")

def decrypt(ciphertext):
    ct = ciphertext.split(',')
    key = 61
    n = 187
    plain = [chr(pow(int(char) ,key) % n) for char in ct]
    return plain
# def uploadFile(request):
#     print("Executing...")
#     if request.method == 'POST':
#         inpFile = request.FILES.get('file',False)
#         if inpFile == False:
#             return JsonResponse({'Upload Status': 'No files found, please ensure whether file is selected properly!'})
#         print(inpFile.name)
#         if inpFile.name.endswith(".csv") == False:
#             print("Not csv")
#             return JsonResponse({'Upload Status': 'Invalid file format, Please upload .csv file in specified structure!'})
#         df = pd.read_csv(inpFile)
#         if isValidCsv(df) == False:
#             print("Invalid Structure of given csv")
#             return JsonResponse({'Upload Status': 'Invalid structure of given .csv file, please check for correct Structure!'})
#         inpFile.name = 'Dataset.csv'
#         FileModel.objects.all().delete()
#         Symptoms.objects.all().delete()
#         putSymptoms(df)
#         form = FileModel(upload = inpFile)
#         form.save()
#         Acc = trainModel()
#         Acc*=100
#         res = 'Model building is completed with accuracy of ' + str(Acc) +"%"
#         return JsonResponse({'Upload Status': res})
#     else:
#         form = UploadFileForm()
#         print("Else exec")
#     print("return exec")
#     return render(request,"hospital.html")

def isValidCsv(df):
    valid = True
    r = df.shape[0]
    c = df.shape[1]
    print(df.shape)
    col = 0
    for i in df.items():
        if type(i[0]) != str: #i[0] gives column value
            valid = False
        if col != (c-1):
            val = [0,1]
            for j in i[1]:    #i[1] gives row values of a column
                if j not in val:
                    valid = False
        else:
            for j in i[1]:   #Checks row values of last column are strings 
                if type(j) != str:
                    valid = False  
        col+=1
    return valid

def putSymptoms(df):
    c = df.shape[1]
    col = 0
    for i in df.items():
        if col != (c-1):
            S = Symptoms(Sym_name = i[0])
            S.save()
        col+=1
    return

def trainModel():
    # q = FileModel.objects.all()
    # dataset = pd.read_csv(q[0].upload)
    dataset = pd.read_csv(settings.MEDIA_ROOT+"\\test\\UploadTest.csv")
    attr = dataset.drop('prognosis', axis=1)    
    # print(attr.iloc[[0]])                                              
    target = dataset['prognosis']                                                                   
    X_train, X_test, Y_train, Y_test = train_test_split(attr, target, test_size=0.15)  
    rf_model = RandomForestClassifier(n_estimators=1000, n_jobs=-1)              
    rf_model.fit(X_train, Y_train)
    rf_prediction = rf_model.predict(X_test)
    # df = pd.DataFrame({'Actual': Y_test, 'Predicted': rf_prediction})
    # print(df)
    Acc = accuracy_score(Y_test, rf_prediction)
    print(Acc)
    # print(classification_report(Y_test, rf_prediction))
    # print(Y_test)
    url = settings.MEDIA_ROOT + "\models\model1.pkl"
    joblib.dump(rf_model, url)    
    return Acc

def getFile(request):
    # q = FileModel.objects.all()
    # print(q)
    # df = pd.read_csv(q[0].upload)
    return JsonResponse({'GET Status ': 'Success' })

def sendFile(request):
    url = settings.MEDIA_ROOT + "\downloads\Sample_Input_CSV_Structure.csv"
    outFile = open(url, 'rb')
    response = HttpResponse(outFile,content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename =Sample_Input_Structure.csv' 
    outFile.close()
    return response

def result(request):
        Symptoms = getSymptoms()
        SymptomsDict = {}
        for i in Symptoms:
            SymptomsDict[i] = 0
        noOfSymptomsFound = 6
        syms = []
        for i in range(noOfSymptomsFound):
            syms.append(request.POST.get("sym" + str(i+1)))
        print(syms)
        symsFound = set(syms)
        none = "None"
        if none in symsFound:
            symsFound.remove('None')
        print(len(symsFound))
        if len(symsFound) == 0:
            return patient(request)
        for  i in symsFound :
            SymptomsDict[i] = 1
        print(SymptomsDict)
        InpSeries = pd.Series(SymptomsDict)
        FinInpData = pd.DataFrame([InpSeries])
        url = settings.MEDIA_ROOT + "\models\model1.pkl"
        model = joblib.load(url)            
        rf_prediction =  model.predict_proba(FinInpData)
        cls = model.classes_ # Getting Output classes(i.e., Diseases)
        res = [] #2d array that contains elements in given form ([disease, probablity to get that disease])
        mean = np.mean(rf_prediction[0])
        for i in range(len(cls)):
            if rf_prediction[0][i] >= mean:
                l = []
                l.append(cls[i])
                l.append(rf_prediction[0][i])
                res.append(l)
        print("Mean =>", mean)
        #sorting in reverse order to get high risk disease first
        def takeSecond(elem):
            return elem[1]
        res.sort(key=takeSecond,reverse=True) 
        print("Input symptoms ->", symsFound)
        print("\nResult in form of figure:\n")
        x = []
        y = []
        for i in res:
            percent = i[1] * 100 #Converting to percentage
            if percent > 0:
                outText = "   Risk of having {} is {:0.1f} %".format(i[0],percent)
                x.append(i[0])
                y.append(percent)
                print(outText)
        print("\n\nResult in form of figure : ")
        font1 = {'family':'serif','color':'Khaki','size':12}
        font2 = {'family':'serif','color':'Khaki','size':12}
        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use("dark_background")
        fig , ax = plt.subplots()
        plt.bar(x,y,width=0.5,color = "Gold")
        ax.set_xticks(x)
        ax.set_xticklabels(x, rotation=45, horizontalalignment="right")
        plt.title("Disease Risk Assessment Result",fontsize=22,fontdict = font1)
        plt.ylabel("Percentage",fontdict = font2)
        plt.xlabel("Disease",fontdict = font2)
        target = settings.MEDIA_ROOT + "\Result\\res.jpg"
        print(target)
        plt.savefig(target,bbox_inches ="tight",pad_inches = 0.3,transparent = True)
        return render(request,"result.html")




#Structure of Symptoms
        # Symptoms = [
        #                 "abdominal_pain",
        #                 "abnormal_menstruation",
        #                 "acidity",
        #                 "acute_liver_failure",
        #                 "altered_sensorium",
        #                 "anxiety",
        #                 "back_pain",
        #                 "belly_pain",
        #                 "blackheads",
        #                 "bladder_discomfort",
        #                 "blister",
        #                 "blood_in_sputum",
        #                 "bloody_stool",
        #                 "blurred_and_distorted_vision",
        #                 "breathlessness",
        #                 "brittle_nails",
        #                 "bruising",
        #                 "burning_micturition",
        #                 "chest_pain",
        #                 "chills",
        #                 "cold_hands_and_feets",
        #                 "coma",
        #                 "congestion",
        #                 "constipation",
        #                 "continuous_feel_of_urine",
        #                 "continuous_sneezing",
        #                 "cough",
        #                 "cramps",
        #                 "dark_urine",
        #                 "dehydration",
        #                 "depression",
        #                 "diarrhoea",
        #                 "dischromic _patches",
        #                 "distention_of_abdomen",
        #                 "dizziness",
        #                 "drying_and_tingling_lips",
        #                 "enlarged_thyroid",
        #                 "excessive_hunger",
        #                 "extra_marital_contacts",
        #                 "family_history",
        #                 "fast_heart_rate",
        #                 "fatigue",
        #                 "fluid_overload",
        #                 "fluid_overload",
        #                 "foul_smell_of urine",
        #                 "headache",
        #                 "high_fever",
        #                 "hip_joint_pain",
        #                 "history_of_alcohol_consumption",
        #                 "increased_appetite",
        #                 "indigestion",
        #                 "inflammatory_nails",
        #                 "internal_itching",
        #                 "irregular_sugar_level",
        #                 "irritability",
        #                 "irritation_in_anus",
        #                 "itching",
        #                 "joint_pain",
        #                 "knee_pain",
        #                 "lack_of_concentration",
        #                 "lethargy",
        #                 "loss_of_appetite",
        #                 "loss_of_balance",
        #                 "loss_of_smell",
        #                 "malaise",
        #                 "mild_fever",
        #                 "mood_swings",
        #                 "movement_stiffness",
        #                 "mucoid_sputum",
        #                 "muscle_pain",
        #                 "muscle_wasting",
        #                 "muscle_weakness",
        #                 "nausea",
        #                 "neck_pain",
        #                 "nodal_skin_eruptions",
        #                 "obesity",
        #                 "pain_behind_the_eyes",
        #                 "pain_during_bowel_movements",
        #                 "pain_in_anal_region",
        #                 "painful_walking",
        #                 "palpitations",
        #                 "passage_of_gases",
        #                 "patches_in_throat",
        #                 "phlegm",
        #                 "polyuria",
        #                 "prognosis",
        #                 "prominent_veins_on_calf",
        #                 "puffy_face_and_eyes",
        #                 "pus_filled_pimples",
        #                 "receiving_blood_transfusion",
        #                 "receiving_unsterile_injections",
        #                 "red_sore_around_nose",
        #                 "red_spots_over_body",
        #                 "redness_of_eyes",
        #                 "restlessness",
        #                 "runny_nose",
        #                 "rusty_sputum",
        #                 "scurring",
        #                 "shivering",
        #                 "silver_like_dusting",
        #                 "sinus_pressure",
        #                 "skin_peeling",
        #                 "skin_rash",
        #                 "slurred_speech",
        #                 "small_dents_in_nails",
        #                 "spinning_movements",
        #                 "spotting_ urination",
        #                 "stiff_neck",
        #                 "stomach_bleeding",
        #                 "stomach_pain",
        #                 "sunken_eyes",
        #                 "sweating",
        #                 "swelled_lymph_nodes",
        #                 "swelling_joints",
        #                 "swelling_of_stomach",
        #                 "swollen_blood_vessels",
        #                 "swollen_extremeties",
        #                 "swollen_legs",
        #                 "throat_irritation",
        #                 "toxic_look_(typhos)",
        #                 "ulcers_on_tongue",
        #                 "unsteadiness",
        #                 "visual_disturbances",
        #                 "vomiting",
        #                 "watering_from_eyes",
        #                 "weakness_in_limbs",
        #                 "weakness_of_one_body_side",
        #                 "weight_gain",
        #                 "weight_loss",
        #                 "yellow_crust_ooze",
        #                 "yellow_urine",
        #                 "yellowing_of_eyes",
        #                 "yellowish_skin"
        # ]

#Structure of SymptomsDict
        # SymptomsDict = {
        #         "abdominal_pain" : 0 ,
        #         "abnormal_menstruation" : 0 ,
        #         "acidity" : 0 ,
        #         "acute_liver_failure" : 0 ,
        #         "altered_sensorium" : 0 ,
        #         "anxiety" : 0 ,
        #         "back_pain" : 0 ,
        #         "belly_pain" : 0 ,
        #         "blackheads" : 0 ,
        #         "bladder_discomfort" : 0 ,
        #         "blister" : 0 ,
        #         "blood_in_sputum" : 0 ,
        #         "bloody_stool" : 0 ,
        #         "blurred_and_distorted_vision" : 0 ,
        #         "breathlessness" : 0 ,
        #         "brittle_nails" : 0 ,
        #         "bruising" : 0 ,
        #         "burning_micturition" : 0 ,
        #         "chest_pain" : 0 ,
        #         "chills" : 0 ,
        #         "cold_hands_and_feets" : 0 ,
        #         "coma" : 0 ,
        #         "congestion" : 0 ,
        #         "constipation" : 0 ,
        #         "continuous_feel_of_urine" : 0 ,
        #         "continuous_sneezing" : 0 ,
        #         "cough" : 0 ,
        #         "cramps" : 0 ,
        #         "dark_urine" : 0 ,
        #         "dehydration" : 0 ,
        #         "depression" : 0 ,
        #         "diarrhoea" : 0 ,
        #         "dischromic _patches" : 0 ,
        #         "distention_of_abdomen" : 0 ,
        #         "dizziness" : 0 ,
        #         "drying_and_tingling_lips" : 0 ,
        #         "enlarged_thyroid" : 0 ,
        #         "excessive_hunger" : 0 ,
        #         "extra_marital_contacts" : 0 ,
        #         "family_history" : 0 ,
        #         "fast_heart_rate" : 0 ,
        #         "fatigue" : 0 ,
        #         "fluid_overload" : 0 ,
        #         "fluid_overload" : 0 ,
        #         "foul_smell_of urine" : 0 ,
        #         "headache" : 0 ,
        #         "high_fever" : 0 ,
        #         "hip_joint_pain" : 0 ,
        #         "history_of_alcohol_consumption" : 0 ,
        #         "increased_appetite" : 0 ,
        #         "indigestion" : 0 ,
        #         "inflammatory_nails" : 0 ,
        #         "internal_itching" : 0 ,
        #         "irregular_sugar_level" : 0 ,
        #         "irritability" : 0 ,
        #         "irritation_in_anus" : 0 ,
        #         "itching" : 0 ,
        #         "joint_pain" : 0 ,
        #         "knee_pain" : 0 ,
        #         "lack_of_concentration" : 0 ,
        #         "lethargy" : 0 ,
        #         "loss_of_appetite" : 0 ,
        #         "loss_of_balance" : 0 ,
        #         "loss_of_smell" : 0 ,
        #         "malaise" : 0 ,
        #         "mild_fever" : 0 ,
        #         "mood_swings" : 0 ,
        #         "movement_stiffness" : 0 ,
        #         "mucoid_sputum" : 0 ,
        #         "muscle_pain" : 0 ,
        #         "muscle_wasting" : 0 ,
        #         "muscle_weakness" : 0 ,
        #         "nausea" : 0 ,
        #         "neck_pain" : 0 ,
        #         "nodal_skin_eruptions" : 0 ,
        #         "obesity" : 0 ,
        #         "pain_behind_the_eyes" : 0 ,
        #         "pain_during_bowel_movements" : 0 ,
        #         "pain_in_anal_region" : 0 ,
        #         "painful_walking" : 0 ,
        #         "palpitations" : 0 ,
        #         "passage_of_gases" : 0 ,
        #         "patches_in_throat" : 0 ,
        #         "phlegm" : 0 ,
        #         "polyuria" : 0 ,
        #         "prognosis" : 0 ,
        #         "prominent_veins_on_calf" : 0 ,
        #         "puffy_face_and_eyes" : 0 ,
        #         "pus_filled_pimples" : 0 ,
        #         "receiving_blood_transfusion" : 0 ,
        #         "receiving_unsterile_injections" : 0 ,
        #         "red_sore_around_nose" : 0 ,
        #         "red_spots_over_body" : 0 ,
        #         "redness_of_eyes" : 0 ,
        #         "restlessness" : 0 ,
        #         "runny_nose" : 0 ,
        #         "rusty_sputum" : 0 ,
        #         "scurring" : 0 ,
        #         "shivering" : 0 ,
        #         "silver_like_dusting" : 0 ,
        #         "sinus_pressure" : 0 ,
        #         "skin_peeling" : 0 ,
        #         "skin_rash" : 0 ,
        #         "slurred_speech" : 0 ,
        #         "small_dents_in_nails" : 0 ,
        #         "spinning_movements" : 0 ,
        #         "spotting_ urination" : 0 ,
        #         "stiff_neck" : 0 ,
        #         "stomach_bleeding" : 0 ,
        #         "stomach_pain" : 0 ,
        #         "sunken_eyes" : 0 ,
        #         "sweating" : 0 ,
        #         "swelled_lymph_nodes" : 0 ,
        #         "swelling_joints" : 0 ,
        #         "swelling_of_stomach" : 0 ,
        #         "swollen_blood_vessels" : 0 ,
        #         "swollen_extremeties" : 0 ,
        #         "swollen_legs" : 0 ,
        #         "throat_irritation" : 0 ,
        #         "toxic_look_(typhos)" : 0 ,
        #         "ulcers_on_tongue" : 0 ,
        #         "unsteadiness" : 0 ,
        #         "visual_disturbances" : 0 ,
        #         "vomiting" : 0 ,
        #         "watering_from_eyes" : 0 ,
        #         "weakness_in_limbs" : 0 ,
        #         "weakness_of_one_body_side" : 0 ,
        #         "weight_gain" : 0 ,
        #         "weight_loss" : 0 ,
        #         "yellow_crust_ooze" : 0 ,
        #         "yellow_urine" : 0 ,
        #         "yellowing_of_eyes" : 0 ,
        #         "yellowish_skin" : 0 
        # }
#Json
        # @csrf_exempt
        # def putSymptoms(request):
        #     data = json.loads(request.body)
        #     Syms = data['symps']
        #     print(Syms)
        #     for i in Syms:
        #         S = Symptoms(Sym_name = i)
        #         S.save()
        #     return JsonResponse({'results': 'Done'})
        # loading
        # Symptoms = json.loads(getSymptoms())
        # Symptoms = Symptoms["results"]
        # dumping
        # json.dumps({'results': response})
# valid = True
# r = df.shape[0]
# c = df.shape[1]
# col = 0
# for i in df.items():
#     if type(i[0]) != str:
#         valid = False
#     if col != (c-1):
#         print(i[0],end = " ")
#         for j in i[1]:
#             print(j,end = "      ")
#         print()
#     else:
#         print(i[0],end = " ")
#         for j in i[1]:
#             print(type(j),end = "      ")
#         print()   
#     col+=1

# q = FileModel.objects.all()
# dataset = pd.read_csv(q[0].upload)
# attr = dataset.drop('prognosis', axis=1)                                                  
# target = dataset['prognosis']                                                                   
# X_train, X_test, Y_train, Y_test = train_test_split(attr, target, test_size=0.15)  
# dc_tree = DecisionTreeClassifier()    
# dc_tree.fit(X_train, Y_train)   
# prediction = dc_tree.predict(X_test)  
# print(prediction)
# print(Y_test)
# url = settings.MEDIA_ROOT + "\models\model1.pkl"
# joblib.dump(dc_tree, url)  

# def uploadFile(request):
#     print("Executing...")
#     if request.method == 'POST':
#         print("\n\n*********** RSA Encrypted Text ***********\n\n")
#         ciphertext = request.POST['reqdata']
#         for i in range(math.floor(len(ciphertext)*0.01)):
#             if i%133 == 0:
#                 print()
#             print(ciphertext[i],end="")
#         print("......")
#         print("\n\n*********** RSA Decrypted text ***********\n")
#         plaintxt = decrypt(ciphertext)
#         for i in range(math.floor(len(plaintxt)*0.01)):
#             if i%133 == 0:
#                 print()
#             print(plaintxt[i],end="")
#         print("......\n")
#         csvText = ""
#         for i in range(len(plaintxt)):
#             csvText+=plaintxt[i]
#         newdf = pd.read_csv(StringIO(csvText))
#         print("\n\n\n")
#         print(newdf)
#         attr = newdf.drop('prognosis', axis=1)
#         target = newdf['prognosis']  
#         d = {}
#         for key in target:
#             d[key] = d.get(key, 0) + 1

#         #print count of each disease
#         for i in d:
#             print(i, d[i])

#         smote = SMOTE()
#         x_smote, y_smote = smote.fit_resample(attr, target)
#         # print(x_smote)
#         # print(y_smote)
#         x_smote['prognosis'] = y_smote
        
#         print('\n\nOriginal dataset shape', Counter(target))
#         print('\n\nResample dataset shape', Counter(y_smote))
#         resstr = x_smote.to_string(index=False)
#         print(resstr)
#         # f = open(settings.MEDIA_ROOT+"\\test\\UploadTest.csv", 'w')
#         f = open(settings.MEDIA_ROOT+"\\test\\UploadTest.csv", 'a+', newline='')
#         f.seek(0) # Ensure you're at the start of the file..
#         first_char = f.read(1) # Get the first character
#         exist = False
#         if not first_char:
#             print("file is empty") 
#         else:
#             exist = True
#         writer = csv.writer(f)
#         for i in csvText.split('\n'):
#             if exist == True:
#                 exist = False
#             else:
#                 row = i.split(",")
#                 writer.writerow(row)
#         f.close()
#         return JsonResponse({'Upload Status': 'Success'})
#     else:
#         print("return exec")
#         return render(request,"hospital.html")